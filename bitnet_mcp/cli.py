import argparse
import asyncio
import sys

from .client import MCPClient
from .context import ContextManager


async def run(endpoint: str, model_path: str) -> None:
    """Interactive session forwarding context to an MCP server and BitNet."""
    ctx = ContextManager()
    client = MCPClient(endpoint)
    from .model import BitNetModel

    model = BitNetModel(model_path)

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            ctx.add_user_message(user_input)
            prompt = ctx.to_prompt()
            # Query the MCP server first for any tool calls
            server_resp = await client.send_request("chat", {"prompt": prompt})
            tool_answer = server_resp.get("result")
            if tool_answer:
                print(f"[Server] {tool_answer}")
                ctx.add_assistant_message(tool_answer)

            # Generate final answer with BitNet
            answer = model.generate(ctx.to_prompt())
            print(f"Assistant: {answer}")
            ctx.add_assistant_message(answer)
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="BitNet MCP client")
    parser.add_argument("endpoint", help="MCP server endpoint")
    parser.add_argument("model", help="Path or name of the BitNet model")
    args = parser.parse_args()

    asyncio.run(run(args.endpoint, args.model))


if __name__ == "__main__":
    main()
