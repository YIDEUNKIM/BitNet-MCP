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
    default_ctx_size: int = 2048  # 2048
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

        # SLMToolClassifier 클래스 내부의 select_tool 함수를 이 코드로 교체하세요.

    def select_tool(self, user_query: str, available_tools: List[Dict]) -> str:
        """
        Uses SLM as a 'tool classifier' and reliably selects the most suitable tool name
        through a clear prompt and robust, fuzzy matching logic.
        """
        tool_descriptions_list = []
        for tool in available_tools:
            tool_entry = f"- tool name: {tool['name']}\n  description: {tool.get('description', '')}"
            tool_descriptions_list.append(tool_entry)

        tools_text = "\n".join(tool_descriptions_list)

        prompt = f"""[System Role]
You are a tool-selection expert. Your ONLY job is to analyze the user's query and the tool descriptions to identify the single best tool.
Respond with ONLY the `tool name`. If there is no appropriate tool, return only "chat".
[available tools]
- Tool Name: chat
  Description: Use for general conversation, greetings, or when no other tool fits.
{tools_text}
[user query]
{user_query}
Your one-word response (the exact Tool Name):"""

        slm_output = self.slm_func(prompt, response_mode="single_line").strip().lower()
        print(f"initial SLM tool select output: {slm_output}")

        if slm_output.startswith("slm call failed"):
            print(f"[DEBUG] SLM tool selection failed. Falling back to chat.")
            return "chat"

        valid_tool_names = [tool['name'] for tool in available_tools] + ['chat']

        # ✨ --- 새로운 관대한(Fuzzy) 매칭 로직 --- ✨
        # 1순위: 정확히 일치하는지 먼저 확인 (가장 빠르고 정확)
        if slm_output in valid_tool_names:
            return slm_output

        # 2순위: 밑줄(_)을 기준으로 "관대한" 매칭 시도
        # 예: "solve__equation"을 "solve_equation"으로 매칭
        for tool_name in valid_tool_names:
            if "_" in tool_name:
                # 각 이름을 밑줄로 분리하여 공백이 아닌 부분만 세트(set)로 만듭니다.
                tool_parts = set(part for part in tool_name.split('_') if part)
                slm_parts = set(part for part in slm_output.split('_') if part)

                # 도구 이름의 모든 구성요소가 SLM 출력에 포함되는지 확인
                if tool_parts and tool_parts.issubset(slm_parts):
                    print(f"   - Fuzzy match successful: mapped '{slm_output}' to '{tool_name}'")
                    return tool_name

        # 3순위: 기존의 포함(substring) 관계 확인 (최후의 보루)
        for tool_name in valid_tool_names:
            if tool_name in slm_output:
                return tool_name

        # 모든 매칭 실패 시 chat으로 fallback
        return "chat"


class SLMParameterExtractor:
    """Uses an SLM to extract parameters for a given tool based on its schema."""

    def __init__(self, slm_func):
        self.slm_func = slm_func

    def extract(self, user_query: str, tool: Dict) -> Dict:
        params_schema = tool.get('parameters', {}).get('properties', {})
        if not params_schema: return {}
        schema_text = "\n".join(
            [f"- {name} ({details.get('type', 'N/A')}): {details.get('description', 'No description.')}" for
             name, details in params_schema.items()])
        examples = """
[Examples]
- User Query: "what is 5 * 3?"
- Extracted Parameters: {"expression": "5 * 3"}
- User Query: "can you solve x**2 - 4 = 0 for me?"
- Extracted Parameters: {"equation": "x**2 - 4 = 0"}
"""
        prompt = f"""[System Role]
You are an expert data extraction assistant. Your task is to analyze the user's query and extract the required parameters for the given tool. Respond ONLY with a single, valid, minified JSON object.
[Tool Name]
{tool['name']}
[Parameter Schema]
{schema_text}
{examples}
[User Query to Process]
{user_query}
[Extracted Parameters (JSON only)]
"""
        slm_response = self.slm_func(prompt, response_mode="full")
        if slm_response.startswith("slm call failed"): return {}
        try:
            json_match = re.search(r'\{.*\}', slm_response, re.DOTALL)
            return json.loads(json_match.group(0).replace("'", '"')) if json_match else {}
        except (json.JSONDecodeError, SyntaxError, AttributeError):
            return {}


class ToolRouter:
    """Manages dynamic tool selection and execution using a hybrid approach."""

    def __init__(self, config: AppConfig, args):
        self.config = config
        self.args = args
        self.tool_classifier = SLMToolClassifier(self._call_slm)
        self.param_extractor = SLMParameterExtractor(self._call_slm)
        self._tools_cache = None

    def _call_slm(self, prompt: str, response_mode: str = "full") -> str:
        return query_local_slm(prompt, self.args, response_mode)

    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        if self._tools_cache: return self._tools_cache
        try:
            async with Client(self.config.server_url) as client:
                tools = await client.list_tools()
                self._tools_cache = [{"name": t.name, "description": (t.description or '').split('\n')[0],
                                      "parameters": getattr(t, 'parameters', {})} for t in tools]
                return self._tools_cache
        except Exception as e:
            print(f"Failed to retrieve tool list from MCP server: {e}")
            return []

    async def _execute_tool(self, tool_name: str, tool_params: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[DEBUG] CLIENT IS SENDING -> Tool: {tool_name}, Parameters: {tool_params}\n")
        try:
            async with Client(self.config.server_url) as client:
                result = await client.call_tool(tool_name, tool_params)
                return {"success": True, "output": str(getattr(result, 'data', result))}
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

    def _extract_math_expression(self, query: str) -> Optional[str]:
        math_token_pattern = re.compile(
            r"[\d\.]+|\*\*|[\+\-\*\/=]|\b(?:sin|cos|tan|log|sqrt|exp|pi|e|x)\b|[()]",
            re.IGNORECASE
        )
        matches = list(math_token_pattern.finditer(query))
        if not matches: return None

        sequences = []
        if matches:
            current_sequence = [matches[0]]
            for i in range(1, len(matches)):
                prev_match, current_match = matches[i - 1], matches[i]
                gap = query[prev_match.end():current_match.start()]
                if gap.isspace() or not gap:
                    current_sequence.append(current_match)
                else:
                    sequences.append(current_sequence)
                    current_sequence = [current_match]
            sequences.append(current_sequence)

        candidates = [query[seq[0].start():seq[-1].end()] for seq in sequences if seq]
        if not candidates: return None

        best_candidate = max(candidates, key=len)
        if re.search(r"[\d\+\-\*\/=]|\*\*", best_candidate):
            return best_candidate.strip()
        return None

        # ToolRouter 클래스 내부의 route_query 함수를 이 코드로 교체하세요.

    async def route_query(self, user_query: str):
        available_tools = await self._get_available_tools()
        if not available_tools:
            response = self._handle_general_chat(user_query);
            print(f"Answer: {response}");
            return

        print("Step 1: Selecting the best tool with SLM Classifier...")
        processed_query = user_query.replace('^', '**')
        selected_tool_name = self.tool_classifier.select_tool(processed_query, available_tools)
        print(f"[DEBUG] Selected tool: '{selected_tool_name}'")

        if selected_tool_name == "chat":
            response = self._handle_general_chat(user_query)
        else:
            tool_to_execute = next((t for t in available_tools if t['name'] == selected_tool_name), None)
            if not tool_to_execute:
                response = self._handle_general_chat(user_query)
                print(f"\n[Tool '{selected_tool_name}' not found, switching to general chat]")
            else:
                param_names = list(tool_to_execute.get('parameters', {}).get('properties', {}).keys())
                tool_params = None

                print("Step 2: Attempting parameter extraction...")

                # ✨ --- 최종 수정: 모든 단일 파라미터 도구를 안정적으로 처리하도록 확장 --- ✨
                # [1순위] 명시적 명령어 형식: "get_weather seoul"
                if processed_query.lower().startswith(selected_tool_name.lower()):
                    print("   - Strategy: Detected command-like query. Applying rule-based extraction.")
                    argument_str = processed_query[len(selected_tool_name):].strip()

                    # 서버가 파라미터 정보를 주지 않아도, 단일 파라미터 도구라고 가정하고 처리
                    # 이 경우, 파라미터 이름을 알아내야 하지만, 대부분의 단일 파라미터 도구는
                    # 파라미터 이름이 'location', 'username', 'expression' 등으로 예측 가능함.
                    # 가장 안정적인 방법은 아래 2순위에서처럼 맵을 사용하는 것.
                    # 여기서는 우선 SLM으로 넘겨서 처리하게 둔다.

                # [2순위] 규칙 기반 엔티티 추출: 자연어 속에 포함된 특정 패턴을 안정적으로 추출
                if tool_params is None:
                    # ✨ 확장 가능한 맵: 여기에 새 도구를 추가하면 안정성이 크게 향상됩니다.
                    RELIABLE_SINGLE_PARAM_TOOLS = {
                        # 수학 도구
                        'calculate': {'param': 'expression', 'extractor': self._extract_math_expression},
                        'solve_equation': {'param': 'equation', 'extractor': self._extract_math_expression},
                        'expand': {'param': 'expression', 'extractor': self._extract_math_expression},
                        'factorize': {'param': 'expression', 'extractor': self._extract_math_expression},

                        # ✨ 새로운 도구 추가 ✨
                        'get_weather': {'param': 'location',
                                        'extractor': lambda q: q.split(" in ")[-1].split(" ")[0]},
                        # "weather in Seoul" -> "Seoul"
                        'get_user_info': {'param': 'username', 'extractor': lambda q: q.split("'s ")[0]}
                        # "yideun's info" -> "yideun"
                    }

                    if selected_tool_name in RELIABLE_SINGLE_PARAM_TOOLS:
                        tool_info = RELIABLE_SINGLE_PARAM_TOOLS[selected_tool_name]
                        print(
                            f"   - Strategy: Reliable tool '{selected_tool_name}' detected. Attempting specialized extraction.")

                        # 명령어 부분만 제거하고 순수 쿼리 전달
                        query_for_extraction = re.sub(f'^{selected_tool_name}\\s*', '', processed_query, count=1,
                                                      flags=re.IGNORECASE).strip()

                        # 도구별 맞춤 추출기(extractor) 사용
                        extracted_value = tool_info['extractor'](query_for_extraction)

                        if extracted_value:
                            tool_params = {tool_info['param']: extracted_value}

                # [3순위] SLM 기반 추출 (최후의 보루)
                if tool_params is None:
                    print("   - Strategy: No rules matched. Falling back to SLM-based extraction.")
                    tool_params = self.param_extractor.extract(processed_query, tool_to_execute)

                print(f"[DEBUG] Extracted parameters: {tool_params}")

                if param_names and not tool_params and selected_tool_name not in RELIABLE_SINGLE_PARAM_TOOLS:
                    response = self._handle_general_chat(user_query)
                    print(f"\n[Parameter extraction failed for '{selected_tool_name}', switching to general chat]")
                else:
                    tool_result = await self._execute_tool(selected_tool_name, tool_params or {})
                    if tool_result.get("success"):
                        response = self._generate_natural_response(user_query, tool_result["output"],
                                                                   selected_tool_name)
                        print(f"\n[Tool Used: {selected_tool_name}]")
                    else:
                        response = self._handle_general_chat(user_query)
                        print(f"\n[Tool execution failed, switching to general chat]")

        print(f"Answer: {response}")

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
