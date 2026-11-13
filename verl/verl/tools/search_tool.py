# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

import httpx

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class SearchTool(BaseTool):
    """Search a local retrieval service and return the top passages as text."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.retrieval_service_url = config["retrieval_service_url"]
        self.topk = int(config.get("topk", 3))
        self.timeout = float(config.get("timeout", 60))
        self.semaphore = asyncio.Semaphore(int(config.get("max_concurrency", 128)))

    @staticmethod
    def format_documents(documents: list[dict]) -> str:
        """Render documents as ``Doc i(Title: ...) ...`` lines, one per document."""
        passages = []
        for i, document in enumerate(documents, 1):
            title, _, body = document["contents"].partition("\n")
            passages.append(f"Doc {i}(Title: {title}) {body}")
        return "\n".join(passages) or "No matching documents found."

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResponse(text="Search requires a non-empty query."), 0.0, {}

        async with self.semaphore, httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.retrieval_service_url,
                json={"queries": [query.strip()], "topk": self.topk, "return_scores": False},
            )
            response.raise_for_status()
            documents = response.json()["result"][0]
        return ToolResponse(text=self.format_documents(documents)), 0.0, {"search_calls": 1}
