import asyncio
import json
from typing import Any, Dict, Optional

import aiohttp


class MCPClient:
    """Simple MCP client using JSON-RPC 2.0 over HTTP."""

    def __init__(self, endpoint: str, session: Optional[aiohttp.ClientSession] = None) -> None:
        self.endpoint = endpoint
        self._session = session or aiohttp.ClientSession()
        self._request_id = 0

    async def close(self) -> None:
        await self._session.close()

    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }
        async with self._session.post(self.endpoint, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def main():
    client = MCPClient("http://localhost:8080/jsonrpc")
    try:
        result = await client.send_request("ping")
        print(json.dumps(result, indent=2))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
