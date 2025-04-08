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
from flask import request

from api.utils.api_utils import validate_request, apikey_required
import logging
from .dify_retrieval import retrieval

@manager.route('/dify/kg/retrieval', methods=['POST'])  # noqa: F821
@apikey_required
@validate_request("knowledge_id", "query")
def kg_retrieval(tenant_id):
    req = request.json
    req["use_kg"] = True  # 设置use_kg为True
    logging.info(f"Using knowledge graph retrieval mode, request params: {req}")  # 添加日志，包含请求参数
    #在 dify_retrieval.py 中，retrieval 函数已经通过装饰器 @apikey_required 和 @validate_request 接收了 tenant_id 参数，当我们再次传入 tenant_id 时，就造成了参数重复
    return retrieval()  # 不传入tenant_id，让装饰器处理
