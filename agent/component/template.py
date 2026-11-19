#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import re
from agent.component.base import ComponentBase, ComponentParamBase
from jinja2.sandbox import SandboxedEnvironment


class TemplateParam(ComponentParamBase):
    """
    Define the Generate component parameters.
    """

    def __init__(self):
        super().__init__()
        self.content = ""
        self.parameters = []

    def check(self):
        self.check_empty(self.content, "[Template] Content")
        return True


class Template(ComponentBase):
    component_name = "Template"

    def get_dependent_components(self):
        inputs = self.get_input_elements()
        cpnts = set([i["key"] for i in inputs if i["key"].lower().find("answer") < 0 and i["key"].lower().find("begin") < 0])
        return list(cpnts)

    def get_input_elements(self):
        key_set = set([])
        res = []
        for r in re.finditer(r"\{([a-z]+[:@][a-z0-9_-]+)\}", self._param.content, flags=re.IGNORECASE):
            cpn_id = r.group(1)
            if cpn_id in key_set:
                continue
            if cpn_id.lower().find("begin@") == 0:
                cpn_id, key = cpn_id.split("@")
                for p in self._canvas.get_component(cpn_id)["obj"]._param.query:
                    if p["key"] != key:
                        continue
                    res.append({"key": r.group(1), "name": p["name"]})
                    key_set.add(r.group(1))
                continue
            cpn_nm = self._canvas.get_component_name(cpn_id)
            if not cpn_nm:
                continue
            res.append({"key": cpn_id, "name": cpn_nm})
            key_set.add(cpn_id)
        return res

    def _run(self, history, **kwargs):
        content = self._param.content

        self._param.inputs = []
        for para in self.get_input_elements():
            if para["key"].lower().find("begin@") == 0:
                cpn_id, key = para["key"].split("@")
                for p in self._canvas.get_component(cpn_id)["obj"]._param.query:
                    if p["key"] == key:
                        value = p.get("value", "")
                        self.make_kwargs(para, kwargs, value)
                        break
                else:
                    assert False, f"Can't find parameter '{key}' for {cpn_id}"
                continue

            component_id = para["key"]
            cpn = self._canvas.get_component(component_id)["obj"]
            if cpn.component_name.lower() == "answer":
                hist = self._canvas.get_history(1)
                if hist:
                    hist = hist[0]["content"]
                else:
                    hist = ""
                self.make_kwargs(para, kwargs, hist)
                continue

            _, out = cpn.output(allow_partial=False)

            result = ""
            if "content" in out.columns:
                result = "\n".join(
                    [o if isinstance(o, str) else str(o) for o in out["content"]]
                )

            self.make_kwargs(para, kwargs, result)

        # 上游修复：PR #7905 / merge commit 20ab6aad4ab7（“Fix: patch SSTI vulnerability in
        # template rendering”，已核它只改本文件）。本实例的 RCE 入口就是这个组件。
        # 注：上游已公布的两份 SSTI 公告都不覆盖本组件，且在本基线（tag_2025_05_18）里无可补点：
        # CVE-2026-28797(GHSA-vvwj-fvwh-4whx) 指 StringTransform/Message（string_transform.py 本树不存在，
        # message.py 存在但无 Jinja2 渲染点）；CVE-2026-45312(GHSA-wpg4-h5g2-jxm6) 指
        # rag/prompts/generator.py 的 citation_prompt()（本树不存在）。全树 grep Jinja2Template/Environment(
        # 只剩本文件这一处渲染点。
        # 本组件的 content 是用户可控的 Jinja2 模板，非沙箱 Environment 可以
        # ``{{ cycler.__init__.__globals__.os.popen(...).read() }}`` 直接拿进程执行（本实例 11-17~11-19 的 RCE 入口）。
        # 与本实例有意的差异：上游还传了 autoescape=True / undefined=StrictUndefined，
        # 那会改变输出（文本被 HTML 转义；未定义变量从“空串”变成抛异常后整段模板原样留下）。
        # 挡住 RCE 的是 sandbox，不是这两个参数，所以这里保持旧渲染语义。
        env = SandboxedEnvironment()
        template = env.from_string(content)

        try:
            content = template.render(kwargs)
        except Exception:
            pass

        for n, v in kwargs.items():
            if not isinstance(v, str):
                try:
                    v = json.dumps(v, ensure_ascii=False)
                except Exception:
                    pass
            content = re.sub(
                r"\{%s\}" % re.escape(n), v, content
            )
            content = re.sub(
                r"(#+)", r" \1 ", content
            )

        return Template.be_output(content)

    def make_kwargs(self, para, kwargs, value):
        self._param.inputs.append(
            {"component_id": para["key"], "content": value}
        )
        try:
            value = json.loads(value)
        except Exception:
            pass
        kwargs[para["key"]] = value
