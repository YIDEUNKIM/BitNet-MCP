import argparse
import subprocess
import os
import platform
import json
import asyncio
import re
import ast
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastmcp import Client
from fastmcp.exceptions import ClientError, ToolError, NotFoundError


# --- AppConfig, get_llama_cli_path, and query_local_slm functions are the same as before ---
@dataclass
class AppConfig:
    """Manages application settings."""
    server_url: str = "http://localhost:8001/sse"
    default_model_path: str = "./models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    default_n_predict: int = 256
    default_threads: int = 8
    default_ctx_size: int = 2048
    default_temperature: float = 0.8
    default_n_gpu_layers: int = 0
    default_system_prompt: str = "You are an intelligent AI agent assistant."

def get_llama_cli_path():
    """Constructs the full path to the llama-cli executable for the current OS."""
    build_dir = "build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"llama-cli not found at {main_path}. Please build the project first.")
    return main_path


def query_local_slm(prompt: str, args, response_mode: str = "full") -> str:
    """Invokes llama-cli as a subprocess and sanitizes the response."""
    try:
        llama_cli_path = get_llama_cli_path()
        command = [
            f'{llama_cli_path}', '-m', args.model, '-p', prompt, '-n', str(args.n_predict),
            '-t', str(args.threads), '-c', str(args.ctx_size), '--temp', str(args.temperature),
            '--no-display-prompt'
        ]
        if hasattr(args, 'n_gpu_layers') and args.n_gpu_layers > 0:
            command.extend(['-ngl', str(args.n_gpu_layers)])
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding='utf-8'
        )
        response = result.stdout.strip()
        response = re.sub(r'\[.*?\]', '', response).strip()
        if response_mode == "single_line":
            return response.split('\n')[0].strip()
        return response
    except FileNotFoundError as e:
        return str(e)
    except subprocess.CalledProcessError as e:
        return f"SLM call failed: {e}"


class SLMToolClassifier:
    """Uses SLM as a 'tool classifier' and corrects incomplete responses with code."""

    def __init__(self, slm_func):
        self.slm_func = slm_func

    def select_tool(self, user_query: str, available_tools: List[Dict]) -> str:
        """
        Uses SLM as a 'tool classifier' and reliably selects the most suitable tool name
        through a clear prompt and intelligent correction logic.
        """
        # 1. Create a clearly separated list of tools to show the SLM.
        tool_descriptions_list = []
        for tool in available_tools:
            # Maximize clarity using "Tool Name:" and "Description:" labels.
            tool_entry = f"- tool name: {tool['name']}\n  description: {tool.get('description', '')}"
            tool_descriptions_list.append(tool_entry)

        tools_text = "\n".join(tool_descriptions_list)

        # 2. Design an enhanced prompt.
        prompt = f"""[System Role]
    You are a tool-selection expert. Your ONLY job is to analyze the user's query and the tool descriptions to identify the single best tool.
    Respond with ONLY the `tool name`. Do not provide any other text or explanation.

    [available tools]
    - Tool Name: chat
      Description: Use for general conversation, greetings, or when no other tool fits.
    {tools_text}

    [user query]
    {user_query}

    Your one-word response (the exact Tool Name):"""

        # 3. Call the SLM to get a (potentially incomplete) response.
        slm_output = self.slm_func(prompt, response_mode="single_line").strip().split()[0]
        print("initial SLM tool select output: "+slm_output)

        # 4. Apply intelligent correction logic.
        valid_tool_names = [tool['name'] for tool in available_tools] + ['chat']

        # 4a. If the SLM's response is already perfectly valid, return it immediately.
        if slm_output in valid_tool_names:
            return slm_output

        # 4b. If the SLM's response is incomplete (e.g., 'get_user_'),
        #     find a name in the list of valid tool names that starts with that response.
        for tool_name in valid_tool_names:
            if tool_name.startswith(slm_output):
                print(f"[DEBUG] Corrected incomplete response '{slm_output}' to '{tool_name}'")
                return tool_name

        # 5. If correction also fails, handle it as 'chat' for safety.
        return "chat"


class PythonParameterExtractor:
    """Dynamically reads the tool schema and extracts parameters with Python code."""

    def extract(self, user_query: str, tool: Dict) -> Dict:
        params_to_extract = tool.get('parameters', {}).get('properties', {})
        if not params_to_extract:
            return {}

        query_lower = user_query.lower()
        extracted_params = {}

        # Dynamically create a list of 'stop words'.
        stop_words = [
            'what', 'is', 'was', 'are', 'the', 'a', 'an', 'in', 'for', 'of', 'about',
            'explain', 'give', 'show', 'me', 'tell', 'can', 'you', 'my', 'i',
            'please', 'provide', 'with', 'latest', 'current', 'how', 'like', 'information'
        ]
        # Also add the tool name itself (e.g., 'get', 'weather') to the stop words.
        stop_words.extend(tool['name'].split('_'))

        query_words = query_lower.split()
        # Remove stop words and the parameter names themselves to leave only pure value candidates.
        candidates = [word for word in query_words if word not in stop_words and word not in params_to_extract.keys()]

        # Assign the last remaining candidate word to the first required parameter (most stable).
        if candidates:
            param_name = list(params_to_extract.keys())[0]  # Assume the first parameter is the target.
            extracted_params[param_name] = candidates[-1]

        return extracted_params


class ToolRouter:
    """Manages dynamic tool selection and execution using a hybrid approach."""

    def __init__(self, config: AppConfig, args):
        self.config = config
        self.args = args
        self.tool_classifier = SLMToolClassifier(self._call_slm)
        self.param_extractor = PythonParameterExtractor()
        self._tools_cache = None

    def _call_slm(self, prompt: str, response_mode: str = "full") -> str:
        return query_local_slm(prompt, self.args, response_mode)

    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Dynamically fetches the list of available tools from the server."""
        if self._tools_cache: return self._tools_cache
        try:
            async with Client(self.config.server_url) as client:
                tools = await client.list_tools()
                print(f"\n[DEBUG] Dynamically fetched tools from server:\n{tools}\n")
                self._tools_cache = [
                    {"name": t.name, "description": t.description, "parameters": getattr(t, 'parameters', {})} for t in
                    tools]
                return self._tools_cache
        except Exception as e:
            print(f"Failed to retrieve tool list from MCP server: {e}")
            return []

    async def _execute_tool(self, tool_name: str, tool_params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with Client(self.config.server_url) as client:
                print(f"Executing tool '{tool_name}' with parameters: {tool_params}")
                result = await client.call_tool(tool_name, tool_params)
                print(f"\n[DEBUG] Raw tool result from server for '{tool_name}':\n{result}\n")
                output = str(getattr(result, 'data', result))
                return {"success": True, "output": output}
        except Exception as e:
            return {"success": False, "error": f"Tool execution failed: {e}"}

    def _generate_natural_response(self, query: str, tool_output: str, tool_name: str) -> str:
        try:
            data_dict = ast.literal_eval(tool_output)
            readable_results = ", ".join([f"{key}: {value}" for key, value in data_dict.items()])
        except (ValueError, SyntaxError):
            readable_results = tool_output
        prompt = f"""[System Role]
You are a helpful assistant who summarizes tool results into natural, user-friendly language.
[User's Original Question]
{query}

[Tool Used]
{tool_name}

[Tool Execution Result (pre-processed for clarity)]
{readable_results}

[Instruction]
Based on the provided tool execution result, answer the user's original question in a complete and friendly English sentence.
For example, if the result is "location: seoul, temperature: 25, condition: clean", your response should be like "Currently, the weather in Seoul is clean and the temperature is 25 degrees."

[Final Answer]

"""
        return self._call_slm(prompt, response_mode="single_line")

    def _handle_general_chat(self, query: str) -> str:
        prompt = f"""[System Role]
{self.config.default_system_prompt}
[User's Original Question]
{query}

[Final Answer]

"""
        return self._call_slm(prompt, response_mode="single_line")

    async def route_query(self, user_query: str):
        available_tools = await self._get_available_tools()
        if not available_tools:
            response = self._handle_general_chat(user_query);
            print(f"Answer: {response}");
            return

        print("Step 1: Selecting the best tool with SLM Classifier...")
        selected_tool_name = self.tool_classifier.select_tool(user_query, available_tools)
        print(f"[DEBUG] Selected tool: '{selected_tool_name}'")

        if selected_tool_name != "chat":
            tool_to_execute = next((t for t in available_tools if t['name'] == selected_tool_name), None)
            if tool_to_execute:
                print(f"Step 2: Extracting parameters for '{selected_tool_name}' with Python Logic...")
                tool_params = self.param_extractor.extract(user_query, tool_to_execute)
                print(f"[DEBUG] Extracted parameters: {tool_params}")

                tool_result = await self._execute_tool(selected_tool_name, tool_params)
                if tool_result.get("success"):
                    response = self._generate_natural_response(user_query, tool_result["output"], selected_tool_name)
                    print(f"\n[Tool Used: {selected_tool_name}]")
                else:
                    response = self._handle_general_chat(user_query)
                    print(f"\n[Tool execution failed, switching to general chat]")
            else:
                response = self._handle_general_chat(user_query)
                print(f"\n[Tool '{selected_tool_name}' not found, switching to general chat]")
        else:
            response = self._handle_general_chat(user_query)

        print(f"Answer: {response}")


# --- main_async and main functions are the same as before ---
async def main_async():
    config = AppConfig()
    parser = argparse.ArgumentParser(description='Final Hybrid Dynamic MCP Client')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", default=config.default_model_path)
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict",
                        default=config.default_n_predict)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", default=config.default_threads)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the prompt context",
                        default=config.default_ctx_size)
    parser.add_argument("-temp", "--temperature", type=float, help="Temperature for sampling",
                        default=config.default_temperature)
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, help="Number of layers to offload to GPU",
                        default=config.default_n_gpu_layers)
    parser.add_argument("-q", "--query", type=str, help="User query to process", default=None)
    args = parser.parse_args()
    print("======================================================")
    print("  MCP Client - Hybrid Dynamic Tool Selection (Final Version)")
    print("======================================================")
    router = ToolRouter(config, args)
    user_input = args.query or input("\nEnter your question: ")
    try:
        await router.route_query(user_input)
    except Exception as e:
        print(f"A critical error occurred during processing: {e}")
    print("\nExecution complete.")


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"An error occurred during program execution: {e}")


if __name__ == "__main__":
    main()
