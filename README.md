# BitNet-MCP

BitNet-MCP Hybrid System for Efficient Query processing.

This repository provides a minimal MCP client that communicates with an MCP
server using JSON-RPC 2.0. The client keeps a conversation context and sends it
as a prompt to the server.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the CLI and provide the MCP endpoint:

```bash
python -m bitnet_mcp.cli http://localhost:8080/jsonrpc
```

Type messages and the client will forward the conversation to the server.
Use `exit` or `quit` to stop the session.
