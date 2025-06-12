# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
Reference:
 - [graphrag](https://github.com/microsoft/graphrag)
 - [LightRag](https://github.com/HKUDS/LightRAG)
"""

import html
import json
import logging
import re
import time
from collections import defaultdict
from hashlib import md5
from typing import Any, Callable
import os
import trio
from typing import Set, Tuple

import networkx as nx
import numpy as np
import xxhash
from networkx.readwrite import json_graph
import dataclasses

from api import settings
from api.utils import get_uuid
from rag.nlp import search, rag_tokenizer
from rag.utils.doc_store_conn import OrderByExpr
from rag.utils.redis_conn import REDIS_CONN

GRAPH_FIELD_SEP = "<SEP>"

ErrorHandlerFn = Callable[[BaseException | None, str | None, dict | None], None]

chat_limiter = trio.CapacityLimiter(int(os.environ.get('MAX_CONCURRENT_CHATS', 10)))

@dataclasses.dataclass
class GraphChange:
    removed_nodes: Set[str] = dataclasses.field(default_factory=set)
    added_updated_nodes: Set[str] = dataclasses.field(default_factory=set)
    removed_edges: Set[Tuple[str, str]] = dataclasses.field(default_factory=set)
    added_updated_edges: Set[Tuple[str, str]] = dataclasses.field(default_factory=set)

def perform_variable_replacements(
    input: str, history: list[dict] | None = None, variables: dict | None = None
) -> str:
    """Perform variable replacements on the input string and in a chat log."""
    if history is None:
        history = []
    if variables is None:
        variables = {}
    result = input

    def replace_all(input: str) -> str:
        result = input
        for k, v in variables.items():
            result = result.replace(f"{{{k}}}", str(v))
        return result

    result = replace_all(result)
    for i, entry in enumerate(history):
        if entry.get("role") == "system":
            entry["content"] = replace_all(entry.get("content") or "")

    return result


def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\"\x00-\x1f\x7f-\x9f]", "", result)


def dict_has_keys_with_types(
    data: dict, expected_fields: list[tuple[str, type]]
) -> bool:
    """Return True if the given dictionary has the given keys with the given types."""
    for field, field_type in expected_fields:
        if field not in data:
            return False

        value = data[field]
        if not isinstance(value, field_type):
            return False
    return True


def get_llm_cache(llmnm, txt, history, genconf):
    hasher = xxhash.xxh64()
    hasher.update(str(llmnm).encode("utf-8"))
    hasher.update(str(txt).encode("utf-8"))
    hasher.update(str(history).encode("utf-8"))
    hasher.update(str(genconf).encode("utf-8"))

    k = hasher.hexdigest()
    bin = REDIS_CONN.get(k)
    if not bin:
        return
    return bin


def set_llm_cache(llmnm, txt, v, history, genconf):
    hasher = xxhash.xxh64()
    hasher.update(str(llmnm).encode("utf-8"))
    hasher.update(str(txt).encode("utf-8"))
    hasher.update(str(history).encode("utf-8"))
    hasher.update(str(genconf).encode("utf-8"))

    k = hasher.hexdigest()
    REDIS_CONN.set(k, v.encode("utf-8"), 24*3600)


def get_embed_cache(llmnm, txt):
    hasher = xxhash.xxh64()
    hasher.update(str(llmnm).encode("utf-8"))
    hasher.update(str(txt).encode("utf-8"))

    k = hasher.hexdigest()
    bin = REDIS_CONN.get(k)
    if not bin:
        return
    return np.array(json.loads(bin))


def set_embed_cache(llmnm, txt, arr):
    hasher = xxhash.xxh64()
    hasher.update(str(llmnm).encode("utf-8"))
    hasher.update(str(txt).encode("utf-8"))

    k = hasher.hexdigest()
    arr = json.dumps(arr.tolist() if isinstance(arr, np.ndarray) else arr)
    REDIS_CONN.set(k, arr.encode("utf-8"), 24*3600)


def get_tags_from_cache(kb_ids):
    hasher = xxhash.xxh64()
    hasher.update(str(kb_ids).encode("utf-8"))

    k = hasher.hexdigest()
    bin = REDIS_CONN.get(k)
    if not bin:
        return
    return bin


def set_tags_to_cache(kb_ids, tags):
    hasher = xxhash.xxh64()
    hasher.update(str(kb_ids).encode("utf-8"))

    k = hasher.hexdigest()
    REDIS_CONN.set(k, json.dumps(tags).encode("utf-8"), 600)

def tidy_graph(graph: nx.Graph, callback):
    """
    Ensure all nodes and edges in the graph have some essential attribute.
    """
    def is_valid_node(node_attrs: dict) -> bool:
        valid_node = True
        for attr in ["description", "source_id"]:
            if attr not in node_attrs:
                valid_node = False
                break
        return valid_node
    purged_nodes = []
    for node, node_attrs in graph.nodes(data=True):
        if not is_valid_node(node_attrs):
            purged_nodes.append(node)
    for node in purged_nodes:
        graph.remove_node(node)
    if purged_nodes and callback:
        callback(msg=f"Purged {len(purged_nodes)} nodes from graph due to missing essential attributes.")

    purged_edges = []
    for source, target, attr in graph.edges(data=True):
        if not is_valid_node(attr):
            purged_edges.append((source, target))
        if "keywords" not in attr:
            attr["keywords"] = []
    for source, target in purged_edges:
        graph.remove_edge(source, target)
    if purged_edges and callback:
        callback(msg=f"Purged {len(purged_edges)} edges from graph due to missing essential attributes.")

def get_from_to(node1, node2):
    if node1 < node2:
        return (node1, node2)
    else:
        return (node2, node1)

def graph_merge(g1: nx.Graph, g2: nx.Graph, change: GraphChange):
    """Merge graph g2 into g1 in place."""
    for node_name, attr in g2.nodes(data=True):
        change.added_updated_nodes.add(node_name)
        if not g1.has_node(node_name):
            g1.add_node(node_name, **attr)
            continue
        node = g1.nodes[node_name]
        node["description"] += GRAPH_FIELD_SEP + attr["description"]
        # A node's source_id indicates which chunks it came from.
        node["source_id"] += attr["source_id"]

    for source, target, attr in g2.edges(data=True):
        change.added_updated_edges.add(get_from_to(source, target))
        edge = g1.get_edge_data(source, target)
        if edge is None:
            g1.add_edge(source, target, **attr)
            continue
        edge["weight"] += attr.get("weight", 0)
        edge["description"] += GRAPH_FIELD_SEP + attr["description"]
        edge["keywords"] += attr["keywords"]
        # A edge's source_id indicates which chunks it came from.
        edge["source_id"] += attr["source_id"]

    for node_degree in g1.degree:
        g1.nodes[str(node_degree[0])]["rank"] = int(node_degree[1])
    # A graph's source_id indicates which documents it came from.
    if "source_id" not in g1.graph:
        g1.graph["source_id"] = []
    g1.graph["source_id"] += g2.graph.get("source_id", [])
    return g1

def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name.upper(),
        entity_type=entity_type.upper(),
        description=entity_description,
        source_id=entity_source_id,
    )


def handle_single_relationship_extraction(record_attributes: list[str], chunk_key: str):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    pair = sorted([source.upper(), target.upper()])
    return dict(
        src_id=pair[0],
        tgt_id=pair[1],
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        metadata={"created_at": time.time()},
    )


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def chunk_id(chunk):
    return xxhash.xxh64((chunk["content_with_weight"] + chunk["kb_id"]).encode("utf-8")).hexdigest()


async def graph_node_to_chunk(kb_id, embd_mdl, ent_name, meta, chunks):
    chunk = {
        "id": get_uuid(),
        "important_kwd": [ent_name],
        "title_tks": rag_tokenizer.tokenize(ent_name),
        "entity_kwd": ent_name,
        "knowledge_graph_kwd": "entity",
        "entity_type_kwd": meta["entity_type"],
        "content_with_weight": json.dumps(meta, ensure_ascii=False),
        "content_ltks": rag_tokenizer.tokenize(meta["description"]),
        "source_id": meta["source_id"],
        "kb_id": kb_id,
        "available_int": 0
    }
    chunk["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(chunk["content_ltks"])
    ebd = get_embed_cache(embd_mdl.llm_name, ent_name)
    if ebd is None:
        ebd, _ = await trio.to_thread.run_sync(lambda: embd_mdl.encode([ent_name]))
        ebd = ebd[0]
        set_embed_cache(embd_mdl.llm_name, ent_name, ebd)
    assert ebd is not None
    chunk["q_%d_vec" % len(ebd)] = ebd
    chunks.append(chunk)


def get_relation(tenant_id, kb_id, from_ent_name, to_ent_name, size=1):
    ents = from_ent_name
    if isinstance(ents, str):
        ents = [from_ent_name]
    if isinstance(to_ent_name, str):
        to_ent_name = [to_ent_name]
    ents.extend(to_ent_name)
    ents = list(set(ents))
    conds = {
        "fields": ["content_with_weight"],
        "size": size,
        "from_entity_kwd": ents,
        "to_entity_kwd": ents,
        "knowledge_graph_kwd": ["relation"]
    }
    res = []
    es_res = settings.retrievaler.search(conds, search.index_name(tenant_id), [kb_id] if isinstance(kb_id, str) else kb_id)
    for id in es_res.ids:
        try:
            if size == 1:
                return json.loads(es_res.field[id]["content_with_weight"])
            res.append(json.loads(es_res.field[id]["content_with_weight"]))
        except Exception:
            continue
    return res


async def graph_edge_to_chunk(kb_id, embd_mdl, from_ent_name, to_ent_name, meta, chunks):
    chunk = {
        "id": get_uuid(),
        "from_entity_kwd": from_ent_name,
        "to_entity_kwd": to_ent_name,
        "knowledge_graph_kwd": "relation",
        "content_with_weight": json.dumps(meta, ensure_ascii=False),
        "content_ltks": rag_tokenizer.tokenize(meta["description"]),
        "important_kwd": meta["keywords"],
        "source_id": meta["source_id"],
        "weight_int": int(meta["weight"]),
        "kb_id": kb_id,
        "available_int": 0
    }
    chunk["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(chunk["content_ltks"])
    # 从 meta 中获取实体类型信息
    from_entity_type = meta.get("from_entity_type", "")
    to_entity_type = meta.get("to_entity_type", "")

    # 构建包含实体类型的关系描述
    txt = ""
    if from_entity_type and to_entity_type:
        txt = f" {from_ent_name}{from_entity_type}->{to_ent_name}{to_entity_type}"
    else:
        txt = f" {from_ent_name}->{to_ent_name}"

    # 添加日志
    logging.info("----------------------test------------------------------")
    logging.info(f"Vector content for edge {from_ent_name}->{to_ent_name}:")
    logging.info(f"Base text: {txt}")
    if meta.get("description"):
        logging.info(f"With description: {txt}: {meta['description']}")
    logging.info("----------------------------------------------------")
    ebd = get_embed_cache(embd_mdl.llm_name, txt)
    if ebd is None:
        ebd, _ = await trio.to_thread.run_sync(lambda: embd_mdl.encode([txt+f": {meta['description']}"]))
        ebd = ebd[0]
        set_embed_cache(embd_mdl.llm_name, txt, ebd)
    assert ebd is not None
    chunk["q_%d_vec" % len(ebd)] = ebd
    chunks.append(chunk)

async def does_graph_contains(tenant_id, kb_id, doc_id):
    # Get doc_ids of graph
    fields = ["source_id"]
    condition = {
        "knowledge_graph_kwd": ["graph"],
        "removed_kwd": "N",
    }
    res = await trio.to_thread.run_sync(lambda: settings.docStoreConn.search(fields, [], condition, [], OrderByExpr(), 0, 1, search.index_name(tenant_id), [kb_id]))
    fields2 = settings.docStoreConn.getFields(res, fields)
    graph_doc_ids = set()
    for chunk_id in fields2.keys():
        graph_doc_ids = set(fields2[chunk_id]["source_id"])
    return doc_id in graph_doc_ids

async def get_graph_doc_ids(tenant_id, kb_id) -> list[str]:
    conds = {
        "fields": ["source_id"],
        "removed_kwd": "N",
        "size": 1,
        "knowledge_graph_kwd": ["graph"]
    }
    res = await trio.to_thread.run_sync(lambda: settings.retrievaler.search(conds, search.index_name(tenant_id), [kb_id]))
    doc_ids = []
    if res.total == 0:
        return doc_ids
    for id in res.ids:
        doc_ids = res.field[id]["source_id"]
    return doc_ids


async def get_graph(tenant_id, kb_id, exclude_rebuild=None):
    conds = {
        "fields": ["content_with_weight", "removed_kwd", "source_id"],
        "size": 1,
        "knowledge_graph_kwd": ["graph"]
    }
    res = await trio.to_thread.run_sync(lambda: settings.retrievaler.search(conds, search.index_name(tenant_id), [kb_id]))
    if not res.total == 0:
        for id in res.ids:
            try:
                if res.field[id]["removed_kwd"] == "N":
                    g = json_graph.node_link_graph(json.loads(res.field[id]["content_with_weight"]), edges="edges")
                    if "source_id" not in g.graph:
                        g.graph["source_id"] = res.field[id]["source_id"]
                else:
                    g = await rebuild_graph(tenant_id, kb_id, exclude_rebuild)
                return g
            except Exception:
                continue
    result = None
    return result


async def set_graph(tenant_id: str, kb_id: str, embd_mdl, graph: nx.Graph, change: GraphChange, callback):
    start = trio.current_time()

    # 清理相关的实体类型缓存
    try:
        from rag.nlp import search
        idxnms = [search.index_name(tenant_id)]
        clear_entity_type_cache(idxnms, [kb_id])
        if callback:
            callback(msg=f"已清理知识库 {kb_id} 的实体类型缓存")
    except Exception as e:
        logging.warning(f"清理缓存失败: {e}")

    await trio.to_thread.run_sync(lambda: settings.docStoreConn.delete({"knowledge_graph_kwd": ["graph", "subgraph"]}, search.index_name(tenant_id), kb_id))

    if change.removed_nodes:
        await trio.to_thread.run_sync(lambda: settings.docStoreConn.delete({"knowledge_graph_kwd": ["entity"], "entity_kwd": sorted(change.removed_nodes)}, search.index_name(tenant_id), kb_id))

    if change.removed_edges:
        async with trio.open_nursery() as nursery:
            for from_node, to_node in change.removed_edges:
                 nursery.start_soon(lambda from_node=from_node, to_node=to_node: trio.to_thread.run_sync(lambda: settings.docStoreConn.delete({"knowledge_graph_kwd": ["relation"], "from_entity_kwd": from_node, "to_entity_kwd": to_node}, search.index_name(tenant_id), kb_id)))
    now = trio.current_time()
    if callback:
        callback(msg=f"set_graph removed {len(change.removed_nodes)} nodes and {len(change.removed_edges)} edges from index in {now - start:.2f}s.")
    start = now

    chunks = [{
        "id": get_uuid(),
        "content_with_weight": json.dumps(nx.node_link_data(graph, edges="edges"), ensure_ascii=False),
        "knowledge_graph_kwd": "graph",
        "kb_id": kb_id,
        "source_id": graph.graph.get("source_id", []),
        "available_int": 0,
        "removed_kwd": "N"
    }]
    
    # generate updated subgraphs
    for source in graph.graph["source_id"]:
        subgraph = graph.subgraph([n for n in graph.nodes if source in graph.nodes[n]["source_id"]]).copy()
        subgraph.graph["source_id"] = [source]
        for n in subgraph.nodes:
            subgraph.nodes[n]["source_id"] = [source]
        chunks.append({
            "id": get_uuid(),
            "content_with_weight": json.dumps(nx.node_link_data(subgraph, edges="edges"), ensure_ascii=False),
            "knowledge_graph_kwd": "subgraph",
            "kb_id": kb_id,
            "source_id": [source],
            "available_int": 0,
            "removed_kwd": "N"
        })
    
    async with trio.open_nursery() as nursery:
        for node in change.added_updated_nodes:
            node_attrs = graph.nodes[node]
            nursery.start_soon(graph_node_to_chunk, kb_id, embd_mdl, node, node_attrs, chunks)
        for from_node, to_node in change.added_updated_edges:
            edge_attrs = graph.get_edge_data(from_node, to_node)
            if not edge_attrs:
                # added_updated_edges could record a non-existing edge if both from_node and to_node participate in nodes merging.
                continue
                       # 获取节点的类型信息
            from_node_attrs = graph.nodes[from_node]
            to_node_attrs = graph.nodes[to_node]
            edge_attrs["from_entity_type"] = from_node_attrs.get("entity_type", "")
            edge_attrs["to_entity_type"] = to_node_attrs.get("entity_type", "")
            nursery.start_soon(lambda: graph_edge_to_chunk(kb_id, embd_mdl, from_node, to_node, edge_attrs, chunks))
    now = trio.current_time()
    if callback:
        callback(msg=f"set_graph converted graph change to {len(chunks)} chunks in {now - start:.2f}s.")
    start = now

    es_bulk_size = 4
    for b in range(0, len(chunks), es_bulk_size):
        doc_store_result = await trio.to_thread.run_sync(lambda: settings.docStoreConn.insert(chunks[b:b + es_bulk_size], search.index_name(tenant_id), kb_id))
        if doc_store_result:
            error_message = f"Insert chunk error: {doc_store_result}, please check log file and Elasticsearch/Infinity status!"
            raise Exception(error_message)
    now = trio.current_time()
    if callback:
        callback(msg=f"set_graph added/updated {len(change.added_updated_nodes)} nodes and {len(change.added_updated_edges)} edges from index in {now - start:.2f}s.")

    # 自动预热缓存
    try:
        await check_and_warmup_cache_on_kg_update(tenant_id, kb_id)
    except Exception as e:
        logging.warning(f"知识图谱更新后自动预热缓存失败: {e}")


def is_continuous_subsequence(subseq, seq):
    def find_all_indexes(tup, value):
        indexes = []
        start = 0
        while True:
            try:
                index = tup.index(value, start)
                indexes.append(index)
                start = index + 1
            except ValueError:
                break
        return indexes

    index_list = find_all_indexes(seq,subseq[0])
    for idx in index_list:
        if idx!=len(seq)-1:
            if seq[idx+1]==subseq[-1]:
                return True
    return False


def merge_tuples(list1, list2):
    result = []
    for tup in list1:
        last_element = tup[-1]
        if last_element in tup[:-1]:
            result.append(tup)
        else:
            matching_tuples = [t for t in list2 if t[0] == last_element]
            already_match_flag = 0
            for match in matching_tuples:
                matchh = (match[1], match[0])
                if is_continuous_subsequence(match, tup) or is_continuous_subsequence(matchh, tup):
                    continue
                already_match_flag = 1
                merged_tuple = tup + match[1:]
                result.append(merged_tuple)
            if not already_match_flag:
                result.append(tup)
    return result


async def get_entity_type2sampels(idxnms, kb_ids: list):
    es_res = await trio.to_thread.run_sync(lambda: settings.retrievaler.search({"knowledge_graph_kwd": "ty2ents", "kb_id": kb_ids,
                                       "size": 10000,
                                       "fields": ["content_with_weight"]},
                                      idxnms, kb_ids))

    res = defaultdict(list)
    for id in es_res.ids:
        smp = es_res.field[id].get("content_with_weight")
        if not smp:
            continue
        try:
            smp = json.loads(smp)
        except Exception as e:
            logging.exception(e)

        for ty, ents in smp.items():
            res[ty].extend(ents)
    return res


def flat_uniq_list(arr, key):
    res = []
    for a in arr:
        a = a[key]
        if isinstance(a, list):
            res.extend(a)
        else:
            res.append(a)
    return list(set(res))


async def rebuild_graph(tenant_id, kb_id, exclude_rebuild=None):
    graph = nx.Graph()
    flds = ["knowledge_graph_kwd", "content_with_weight", "source_id"]
    bs = 256
    for i in range(0, 1024*bs, bs):
        es_res = await trio.to_thread.run_sync(lambda: settings.docStoreConn.search(flds, [],
                                 {"kb_id": kb_id, "knowledge_graph_kwd": ["subgraph"]},
                                 [],
                                 OrderByExpr(),
                                 i, bs, search.index_name(tenant_id), [kb_id]
                                 ))
        # tot = settings.docStoreConn.getTotal(es_res)
        es_res = settings.docStoreConn.getFields(es_res, flds)

        if len(es_res) == 0:
            break

        for id, d in es_res.items():
            assert d["knowledge_graph_kwd"] == "subgraph"
            if isinstance(exclude_rebuild, list):
                if sum([n in d["source_id"] for n in exclude_rebuild]):
                    continue
            elif exclude_rebuild in d["source_id"]:
                continue
            
            next_graph = json_graph.node_link_graph(json.loads(d["content_with_weight"]), edges="edges")
            merged_graph = nx.compose(graph, next_graph)
            merged_source = {
                n: graph.nodes[n]["source_id"] + next_graph.nodes[n]["source_id"]
                for n in graph.nodes & next_graph.nodes
            }
            nx.set_node_attributes(merged_graph, merged_source, "source_id")
            if "source_id" in graph.graph:
                merged_graph.graph["source_id"] = graph.graph["source_id"] + next_graph.graph["source_id"]
            else:
                merged_graph.graph["source_id"] = next_graph.graph["source_id"]
            graph = merged_graph

    if len(graph.nodes) == 0:
        return None
    graph.graph["source_id"] = sorted(graph.graph["source_id"])
    return graph

def get_entity_type2sampels_cache(idxnms, kb_ids):
    """获取实体类型样本的缓存"""
    hasher = xxhash.xxh64()
    hasher.update(str(sorted(idxnms)).encode("utf-8"))
    hasher.update(str(sorted(kb_ids)).encode("utf-8"))
    hasher.update("entity_type2sampels".encode("utf-8"))  # 添加特定前缀避免键冲突

    k = hasher.hexdigest()
    
    # 添加调试日志
    logging.info(f"🔍 查找缓存键: {k}，idxnms: {idxnms}，kb_ids: {kb_ids}")
    
    bin = REDIS_CONN.get(k)
    if not bin:
        logging.info(f"❌ 缓存未命中，键: {k}")
        return None
    try:
        # Redis配置了decode_responses=True，所以bin已经是字符串，不需要decode
        if isinstance(bin, bytes):
            result = json.loads(bin.decode("utf-8"))
        else:
            result = json.loads(bin)
        logging.info(f"✅ 缓存命中，键: {k}，数据类型数: {len(result)}")
        return result
    except Exception as e:
        logging.warning(f"⚠️ 缓存数据解析失败，键: {k}，错误: {e}")
        return None


def set_entity_type2sampels_cache(idxnms, kb_ids, data, expire_seconds=3600):
    """设置实体类型样本的缓存，默认缓存1小时"""
    hasher = xxhash.xxh64()
    hasher.update(str(sorted(idxnms)).encode("utf-8"))
    hasher.update(str(sorted(kb_ids)).encode("utf-8"))
    hasher.update("entity_type2sampels".encode("utf-8"))

    k = hasher.hexdigest()
    
    # 添加调试日志
    logging.info(f"💾 设置缓存键: {k}，idxnms: {idxnms}，kb_ids: {kb_ids}，数据类型数: {len(data)}")
    
    try:
        REDIS_CONN.set(k, json.dumps(data, ensure_ascii=False).encode("utf-8"), expire_seconds)
        logging.info(f"✅ 缓存设置成功，键: {k}，过期时间: {expire_seconds}秒")
    except Exception as e:
        logging.error(f"❌ 缓存设置失败，键: {k}，错误: {e}")


async def get_entity_type2sampels_realtime(idxnms, kb_ids: list, use_cache: bool = True, cache_expire: int = 3600):
    """
    实时从知识库中获取实体类型到实体样本的映射
    返回格式: {entity_type: [entity1, entity2, ...]}
    
    Args:
        idxnms: 索引名称列表
        kb_ids: 知识库ID列表
        use_cache: 是否使用缓存，默认True
        cache_expire: 缓存过期时间（秒），默认3600秒（1小时）
    """
    # 如果启用缓存，先尝试从缓存获取
    if use_cache:
        cached_data = get_entity_type2sampels_cache(idxnms, kb_ids)
        if cached_data is not None:
            logging.info(f"🚀 从缓存获取实体类型样本，kb_ids: {kb_ids}，包含 {len(cached_data)} 种类型")
            return cached_data
    
    logging.info(f"📡 实时查询实体类型样本，kb_ids: {kb_ids}")
    
    # 查询所有实体
    es_res = await trio.to_thread.run_sync(lambda: settings.retrievaler.search({
        "knowledge_graph_kwd": "entity",
        "kb_id": kb_ids,
        "size": 10000,
        "fields": ["entity_kwd", "entity_type_kwd", "rank_flt"]
    }, idxnms, kb_ids))

    # 按类型分组实体，使用集合去重
    ty2ents = defaultdict(set)
    for id in es_res.ids:
        entity = es_res.field[id]
        entity_name = entity.get("entity_kwd")
        entity_type = entity.get("entity_type_kwd")
        rank_raw = entity.get("rank_flt", 0)
        
        # 确保rank是数字类型，处理可能的字符串情况
        try:
            if isinstance(rank_raw, str):
                rank = float(rank_raw) if rank_raw else 0.0
            else:
                rank = float(rank_raw) if rank_raw is not None else 0.0
        except (ValueError, TypeError):
            rank = 0.0

        if not entity_name or not entity_type:
            continue

        # 使用集合存储实体，自动去重
        ty2ents[entity_type].add((entity_name, rank))

    # 对每个类型的实体按rank排序，并只保留前12个
    result = defaultdict(list)
    for ty in ty2ents:
        # 转换为列表并按rank排序
        sorted_ents = sorted(ty2ents[ty], key=lambda x: x[1], reverse=True)
        # 只保留前12个实体
        result[ty] = [ent[0] for ent in sorted_ents[:12]]

    # 如果启用缓存，将结果写入缓存
    if use_cache:
        set_entity_type2sampels_cache(idxnms, kb_ids, dict(result), cache_expire)
    
    return result

async def warmup_entity_type_cache(tenant_ids=None, kb_ids=None):
    """
    预热实体类型缓存
    
    Args:
        tenant_ids: 指定的租户ID列表，如果为None则查询所有租户
        kb_ids: 指定的知识库ID列表，如果为None则查询所有知识库
    """
    from api.db.services.knowledgebase_service import KnowledgebaseService
    import trio
    
    logging.info("开始预热实体类型缓存...")
    
    try:
        # 如果没有指定租户和知识库，查询所有活跃的
        if tenant_ids is None:
            # 获取所有租户
            tenant_ids = []
            try:
                from api.db.services.user_service import TenantService
                # 获取所有租户
                tenants = TenantService.get_all()
                tenant_ids = [t.id for t in tenants if t and hasattr(t, 'id')]
                logging.info(f"找到 {len(tenant_ids)} 个租户")
            except Exception as e:
                logging.warning(f"获取租户列表失败: {e}")
                # 如果获取租户失败，尝试从知识库直接获取
                try:
                    kbs = await trio.to_thread.run_sync(lambda: KnowledgebaseService.get_all())
                    tenant_ids = list(set([kb['tenant_id'] for kb in kbs if kb and 'tenant_id' in kb]))
                    logging.info(f"从知识库获取到 {len(tenant_ids)} 个租户")
                except Exception as e2:
                    logging.warning(f"从知识库获取租户也失败: {e2}")
                    return
        
        if kb_ids is None:
            # 获取指定租户下的所有知识库
            kb_ids = []
            try:
                # 使用更简单的方法获取所有知识库ID
                all_kb_ids = await trio.to_thread.run_sync(lambda: KnowledgebaseService.get_all_ids())
                kb_ids = all_kb_ids
                logging.info(f"找到 {len(kb_ids)} 个知识库")
            except Exception as e:
                logging.warning(f"获取知识库列表失败: {e}")
                # 如果失败，尝试按租户获取
                try:
                    for tenant_id in tenant_ids:
                        kbs = await trio.to_thread.run_sync(lambda: KnowledgebaseService.get_by_tenant_id(tenant_id))
                        kb_ids.extend([kb['id'] for kb in kbs if kb and 'id' in kb])
                    logging.info(f"通过租户找到 {len(kb_ids)} 个知识库")
                except Exception as e2:
                    logging.warning(f"通过租户获取知识库也失败: {e2}")
                    return
        
        # 如果还是没有知识库，直接返回
        if not kb_ids:
            logging.info("没有找到知识库，跳过缓存预热。这是正常的，如果系统中还没有创建任何知识库的话。")
            return
        
        # 为每个租户预热缓存
        warmed_count = 0
        for tenant_id in tenant_ids:
            try:
                from rag.nlp import search
                idxnms = [search.index_name(tenant_id)]
                
                # 获取该租户下的知识库
                tenant_kb_ids = []
                try:
                    # 使用正确的方法获取租户下的知识库ID
                    all_tenant_kb_ids = await trio.to_thread.run_sync(lambda: KnowledgebaseService.get_kb_ids(tenant_id))
                    # 只保留在总的知识库列表中的ID
                    tenant_kb_ids = [kb_id for kb_id in all_tenant_kb_ids if kb_id in kb_ids]
                    logging.info(f"租户 {tenant_id} 下找到 {len(tenant_kb_ids)} 个相关知识库")
                except Exception as e:
                    logging.warning(f"获取租户 {tenant_id} 的知识库失败: {e}")
                    continue
                
                if tenant_kb_ids:
                    # 为每个知识库单独预热缓存
                    warmed_kb_count = 0
                    for kb_id in tenant_kb_ids:
                        try:
                            result = await get_entity_type2sampels_realtime(idxnms, [kb_id], use_cache=True)
                            if result:
                                entity_types = len(result)
                                total_entities = sum(len(ents) for ents in result.values())
                                warmed_kb_count += 1
                                logging.info(f"已预热知识库 {kb_id}，包含 {entity_types} 种类型，{total_entities} 个实体")
                            else:
                                logging.info(f"知识库 {kb_id} 暂无实体数据")
                        except Exception as single_cache_e:
                            logging.warning(f"预热知识库 {kb_id} 缓存失败: {single_cache_e}")
                    
                    if warmed_kb_count > 0:
                        warmed_count += 1
                        logging.info(f"租户 {tenant_id} 共预热 {warmed_kb_count} 个知识库的缓存")
                else:
                    logging.info(f"租户 {tenant_id} 没有相关的知识库，跳过")
                    
            except Exception as e:
                logging.warning(f"预热租户 {tenant_id} 缓存失败: {e}")
                continue
        
        logging.info(f"缓存预热完成，共预热 {warmed_count} 个租户的缓存")
        
    except Exception as e:
        logging.error(f"缓存预热过程中出现错误: {e}")


def warmup_entity_type_cache_sync(tenant_ids=None, kb_ids=None):
    """
    同步版本的缓存预热函数，供主线程调用
    """
    import trio
    try:
        trio.run(lambda: warmup_entity_type_cache(tenant_ids, kb_ids))
    except Exception as e:
        logging.error(f"同步预热缓存失败: {e}")


def clear_entity_type_cache(idxnms=None, kb_ids=None):
    """
    清理实体类型缓存
    
    Args:
        idxnms: 索引名称列表，如果为None则清理所有
        kb_ids: 知识库ID列表，如果为None则清理所有
    """
    if idxnms is None or kb_ids is None:
        # 如果没有指定参数，这里可以实现清理所有缓存的逻辑
        # 但需要小心，避免影响其他缓存
        logging.warning("未指定具体参数，跳过缓存清理")
        return
    
    try:
        hasher = xxhash.xxh64()
        hasher.update(str(sorted(idxnms)).encode("utf-8"))
        hasher.update(str(sorted(kb_ids)).encode("utf-8"))
        hasher.update("entity_type2sampels".encode("utf-8"))
        
        k = hasher.hexdigest()
        REDIS_CONN.delete(k)
        logging.info(f"已清理实体类型缓存: {kb_ids}")
    except Exception as e:
        logging.error(f"清理缓存失败: {e}")

async def check_and_warmup_cache_on_kg_update(tenant_id, kb_id):
    """
    在知识图谱更新后检查并预热缓存
    
    这个函数在知识图谱更新后自动调用，确保缓存是最新的
    """
    try:
        from rag.nlp import search
        
        idxnms = [search.index_name(tenant_id)]
        
        # 先清理旧缓存
        clear_entity_type_cache(idxnms, [kb_id])
        
        # 异步预热新缓存
        result = await get_entity_type2sampels_realtime(idxnms, [kb_id], use_cache=True, cache_expire=3600)
        
        if result:
            entity_types = len(result)
            total_entities = sum(len(ents) for ents in result.values())
            logging.info(f"知识图谱更新后，已自动预热知识库 {kb_id} 的缓存，包含 {entity_types} 种实体类型，{total_entities} 个实体")
        
    except Exception as e:
        logging.warning(f"知识图谱更新后预热缓存失败: {e}")
