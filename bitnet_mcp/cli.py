import argparse
import asyncio
import sys
from .client import MCPClient
from .context import ContextManager

async def run(endpoint: str, model_path: str = None) -> None:
    """Interactive session forwarding context to an MCP server and optionally BitNet."""
    ctx = ContextManager()
    client = MCPClient(endpoint)
    
    # Initialize BitNet model if model_path is provided
    model = None
    if model_path:
        try:
            from .model import BitNetModel
            model = BitNetModel(model_path)
            print(f"BitNet model loaded: {model_path}")
        except ImportError:
            print("Warning: BitNet model dependencies not available. Install 'transformers' and 'torch'.")
            print("Falling back to MCP-only mode.")
        except Exception as e:
            print(f"Warning: Failed to load BitNet model: {e}")
            print("Falling back to MCP-only mode.")
    
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            
            ctx.add_user_message(user_input)
            prompt = ctx.to_prompt()
            
            if model:
                # Hybrid mode: Query MCP server first for tool calls, then BitNet for final answer
                try:
                    server_resp = await client.send_request("chat", {"prompt": prompt})
                    tool_answer = server_resp.get("result")
                    if tool_answer:
                        print(f"[Server] {tool_answer}")
                        ctx.add_assistant_message(tool_answer)
                    
                    # Generate final answer with BitNet
                    answer = model.generate(ctx.to_prompt())
                except Exception as e:
                    print(f"Error in hybrid mode: {e}")
                    # Fallback to MCP-only
                    result = await client.send_request("chat", {"prompt": prompt})
                    answer = result.get("result", "")
            else:
                # MCP-only mode
                result = await client.send_request("chat", {"prompt": prompt})
                answer = result.get("result", "")
            
            print(f"Assistant: {answer}")
            ctx.add_assistant_message(answer)
            
    finally:
        await client.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="BitNet MCP client")
    parser.add_argument("endpoint", help="MCP server endpoint")
    parser.add_argument("model", nargs="?", help="Path or name of the BitNet model (optional)")
    parser.add_argument("--model-path", help="Alternative way to specify model path")
    
    args = parser.parse_args()
    
    # Support both positional and optional model argument for backward compatibility
    model_path = args.model or args.model_path
    
    if not args.endpoint:
        print("Usage: python -m bitnet_mcp.cli <MCP endpoint> [model]")
        print("   or: python -m bitnet_mcp.cli <MCP endpoint> --model-path <model>")
        raise SystemExit(1)
    
    print(f"Connecting to MCP server: {args.endpoint}")
    if model_path:
        print(f"BitNet model: {model_path}")
    else:
        print("Running in MCP-only mode")
    
    asyncio.run(run(args.endpoint, model_path))

if __name__ == "__main__":
    main()