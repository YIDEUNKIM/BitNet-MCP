# BitNet-MCP

BitNet-MCP Hybrid System for Efficient Query processing.

This repository provides a simple MCP client that connects the BitNet language
model to external MCP servers using JSON-RPC 2.0. The client maintains
conversation context and can generate replies locally with BitNet while also
forwarding the conversation to an MCP server for tool invocation.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the CLI and provide the MCP endpoint and the BitNet model name or path:

```bash
python -m bitnet_mcp.cli http://localhost:8080/jsonrpc microsoft/BitNet-b1.58-2B-4T
```

The example above assumes the model weights are available locally or via
HuggingFace Hub. Install `transformers` and `torch` to enable BitNet inference.
Type messages and the client will use BitNet to generate a reply. Use `exit` or
`quit` to stop the session.
