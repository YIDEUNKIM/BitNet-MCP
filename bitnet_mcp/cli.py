import asyncio
import json
import sys

from .client import MCPClient
from .context import ContextManager


async def run(endpoint: str) -> None:
    ctx = ContextManager()
    client = MCPClient(endpoint)

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            ctx.add_user_message(user_input)
            prompt = ctx.to_prompt()
            result = await client.send_request("chat", {"prompt": prompt})
            answer = result.get("result", "")
            print(f"Assistant: {answer}")
            ctx.add_assistant_message(answer)
    finally:
        await client.close()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m bitnet_mcp.cli <MCP endpoint>")
        raise SystemExit(1)
    endpoint = sys.argv[1]
    asyncio.run(run(endpoint))


if __name__ == "__main__":
    main()
