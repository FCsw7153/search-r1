"""Search-R1 retrieval adapter; no answer labels are exposed to the tool."""
import asyncio
import httpx
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

class SearchR1Tool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.url = config["retrieval_service_url"]
        self.topk = int(config.get("topk", 3))
        self.timeout = float(config.get("timeout", 30))
        self.limit = asyncio.Semaphore(int(config.get("max_concurrency", 8)))

    async def execute(self, instance_id, parameters, **kwargs):
        query = parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResponse(text="Search requires a non-empty query."), 0.0, {"invalid_query": 1}
        query = query.strip()
        if len(query) > 2048:
            return ToolResponse(text="Search query is too long; shorten it."), 0.0, {"invalid_query": 1}
        # Propagate service failures; ToolAgentLoop reports them to the model as tool errors.
        async with self.limit:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, json={
                    "queries": [query], "topk": self.topk, "return_scores": True,
                })
                response.raise_for_status()
                groups = response.json()["result"]
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError("Retriever must return one result group for one query")
        passages = []
        for i, item in enumerate(groups[0][:self.topk], 1):
            content = item["document"]["contents"]
            if not isinstance(content, str):
                raise ValueError("document.contents must be text")
            title, _, body = content.partition("\n")
            passages.append(f"Doc {i}(Title: {title}) {body}")
        text = "\n".join(passages) or "No matching documents found."
        return ToolResponse(text=text), 0.0, {"search_calls": 1}
