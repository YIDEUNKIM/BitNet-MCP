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
from fastmcp import Client
from fastmcp.exceptions import ClientError, ToolError, NotFoundError

"""
workflow
:  1. 요청 분석: 사용자의 질문을 받으면, 먼저 mcp_server에 어떤 전문가(Tool)들이 있는지 확인합니다.
   2. 전문가 선택: 언어 모델(SLM)에게 질문을 보여주고, 이 일을 처리할 최적의 전문가(Tool)를 단 한 명
      고름.
   3. 정보 준비:
       * 빠른 처리: 만약 전문가가 필요한 정보(Parameter)가 간단한 수학식처럼 규칙(Regex)으로 바로
         뽑을 수 있는 것이면, 즉시 추출.
       * 심층 분석: 규칙으로 안 되면, 언어 모델(SLM)에게 다시 요청하여 질문에서 복잡한 정보를 JSON
         형태로 뽑아냄.
   4. 업무 지시 및 보고: 준비된 정보를 전문가에게 전달하여 일을 처리시키고, 그 결과를 받아 사람이
      이해하기 쉬운 자연스러운 문장으로 바꿔 최종 보고.
"""


# --- AppConfig, get_llama_cli_path, and query_local_slm are the same ---
@dataclass
class AppConfig:
    server_url: str = "http://localhost:8001/sse"
    default_model_path: str = "./models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    default_n_predict: int = 256
    default_threads: int = 8
    default_ctx_size: int = 2048
    default_temperature: float = 0.8
    default_n_gpu_layers: int = 0
    default_system_prompt: str = "You are an intelligent AI agent assistant."


def get_llama_cli_path():
    build_dir = "build"
    main_path = os.path.join(build_dir, "bin", "llama-cli.exe") if platform.system() == "Windows" else os.path.join(
        build_dir, "bin", "llama-cli")
    if platform.system() == "Windows" and not os.path.exists(main_path):
        main_path = os.path.join(build_dir, "bin", "llama-cli")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"llama-cli not found at {main_path}. Please build the project first.")
    return main_path


def query_local_slm(prompt: str, args, response_mode: str = "full") -> str:
    try:
        llama_cli_path = get_llama_cli_path()
        command = [f'{llama_cli_path}', '-m', args.model, '-p', prompt, '-n', str(args.n_predict), '-t',
                   str(args.threads), '-c', str(args.ctx_size), '--temp', str(args.temperature), '--no-display-prompt']
        if hasattr(args, 'n_gpu_layers') and args.n_gpu_layers > 0: command.extend(['-ngl', str(args.n_gpu_layers)])
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        response = result.stdout.strip()
        response = re.sub(r'\[.*?\]', '', response).strip()
        return response.split('\n')[0].strip() if response_mode == "single_line" else response
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        return f"SLM call failed: {e}"


class SLMToolClassifier:
    def __init__(self, slm_func):
        self.slm_func = slm_func

    def select_tool(self, user_query: str, available_tools: List[Dict]) -> str:
        tools_text = "\n".join(
            [f"- tool name: {tool['name']}\n  description: {tool.get('description', '')}" for tool in available_tools])
        prompt = f"""[System Role]
You are a tool-selection expert. Respond with ONLY the `tool name`. If no tool fits, return "chat".
[available tools]
- Tool Name: chat
  Description: For general conversation.
{tools_text}
[user query]
{user_query}
Your one-word response (the exact Tool Name):"""
        slm_output = self.slm_func(prompt, response_mode="single_line").strip().lower()
        print(f"initial SLM tool select output: {slm_output}")
        if slm_output.startswith("slm call failed"): return "chat"
        valid_tool_names = [tool['name'] for tool in available_tools] + ['chat']
        if slm_output in valid_tool_names: return slm_output
        for tool_name in valid_tool_names:
            if "_" in tool_name:
                tool_parts = set(part for part in tool_name.split('_') if part)
                slm_parts = set(part for part in slm_output.split('_') if part)
                if tool_parts and tool_parts.issubset(slm_parts):
                    print(f"   - Fuzzy match successful: mapped '{slm_output}' to '{tool_name}'")
                    return tool_name
        for tool_name in valid_tool_names:
            if tool_name in slm_output: return tool_name
        return "chat"


class SLMParameterExtractor:
    def __init__(self, slm_func):
        self.slm_func = slm_func

    def extract(self, user_query: str, tool: Dict) -> Dict:
        # 이 함수는 이제 복잡한 다중 파라미터 도구를 위한 최후의 보루 역할만 합니다.
        params_schema = tool.get('parameters', {}).get('properties', {})
        if not params_schema: return {}
        schema_text = "\n".join(
            [f"- {param_name} ({details.get('type', 'N/A')}): {details.get('description', 'No description.')}" for
             param_name, details in params_schema.items()])
        examples = """[Examples]
- User Query: "differentiate x**3 with respect to x"
- Extracted Parameters: {"expression": "x**3", "variable": "x"}"""
        prompt = f"""[System Role]
You are an expert data extraction assistant. Respond ONLY with a single, valid, minified JSON object.
[Tool Name]
{tool['name']}
[Parameter Schema]
{schema_text}
{examples}
[User Query to Process]
{user_query}
[Extracted Parameters (JSON only)]"""
        slm_response = self.slm_func(prompt, response_mode="full")
        if slm_response.startswith("slm call failed"): return {}
        try:
            json_match = re.search(r'\{.*\}', slm_response, re.DOTALL)
            return json.loads(json_match.group(0).replace("'", '"')) if json_match else {}
        except (json.JSONDecodeError, SyntaxError, AttributeError):
            return {}


class ToolRouter:
    def __init__(self, config: AppConfig, args):
        self.config = config
        self.args = args
        self.tool_classifier = SLMToolClassifier(self._call_slm)
        self.param_extractor = SLMParameterExtractor(self._call_slm)
        self._tools_cache = None

        # ✨ --- 클라이언트의 "스킬셋": 어떤 타입의 힌트를 어떤 전문가가 처리할지 정의 --- ✨
        self.strategy_by_type = {
            'equation': self._extract_equation,
            'math_expression': self._extract_math_expression,
        }

    def _call_slm(self, prompt: str, response_mode: str = "full") -> str:
        return query_local_slm(prompt, self.args, response_mode)

    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        if self._tools_cache: return self._tools_cache
        try:
            async with Client(self.config.server_url) as client:
                tools = await client.list_tools()

                self._tools_cache = []
                for t in tools:
                    full_description = t.description or ''
                    # 간결한 설명을 위해 첫 줄만 사용 (프롬프트 과부하 방지)
                    concise_desc = full_description.strip().split('\n')[0]

                    # ✨ --- '자기소개서'를 파싱하여 구조화된 파라미터 정보 생성 --- ✨
                    parsed_params = {}
                    # `[param: 이름, type: 종류]` 태그를 모두 찾음
                    param_tags = re.findall(r'\[param:\s*(\w+),\s*type:\s*(\w+)\]', full_description)
                    for name, type in param_tags:
                        parsed_params[name] = {"type": type}

                    tool_data = {
                        "name": t.name,
                        "description": concise_desc,
                        "parameters": {"properties": parsed_params}  # fastmcp가 줬어야 할 정보를 직접 생성
                    }
                    self._tools_cache.append(tool_data)

                print(f"[DEBUG] Loaded and structured tools: {json.dumps(self._tools_cache, indent=2)}")
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
You are a helpful assistant who summarizes tool results into natural language.
[User's Original Question]
{query}
[Tool Used]
{tool_name}
[Tool Execution Result]
{readable_results}
[Final Answer]"""
        return self._call_slm(prompt, response_mode="single_line")

    def _handle_general_chat(self, query: str) -> str:
        prompt = f"""[System Role]
{self.config.default_system_prompt}
[User's Original Question]
{query}
[Final Answer]"""
        return self._call_slm(prompt, response_mode="single_line")

    def _extract_math_expression(self, query: str) -> Optional[str]:
        # 이 함수는 변경되지 않음
        math_token_pattern = re.compile(r"[\d\.]+|\*\*|[\+\-\*\/=]|\b(?:sin|cos|tan|log|sqrt|exp|pi|e|x)\b|[()]",
                                        re.IGNORECASE)
        matches = list(math_token_pattern.finditer(query))
        if not matches: return None
        sequences, current_sequence = [], []
        if matches:
            current_sequence = [matches[0]]
            for i in range(1, len(matches)):
                prev_match, current_match = matches[i - 1], matches[i]
                gap = query[prev_match.end():current_match.start()]
                if gap.isspace() or not gap:
                    current_sequence.append(current_match)
                else:
                    sequences.append(current_sequence); current_sequence = [current_match]
            sequences.append(current_sequence)
        candidates = [query[seq[0].start():seq[-1].end()] for seq in sequences if seq]
        if not candidates: return None
        best_candidate = max(candidates, key=len)
        if re.search(r"[\d\+\-\*\/=]|\*\*", best_candidate): return best_candidate.strip()
        return None

    def _extract_equation(self, query: str) -> Optional[str]:
        # 이 함수는 변경되지 않음
        if '=' not in query: return None
        valid_char_pattern = r"[\d\w\.\s\(\)\*\+\-\/\=]"
        candidates = re.findall(f"({valid_char_pattern}+)", query)
        equation_candidates = [c for c in candidates if '=' in c]
        if not equation_candidates: return None
        return max(equation_candidates, key=len).strip()

    async def route_query(self, user_query: str):
        available_tools = await self._get_available_tools()
        if not available_tools:
            response = self._handle_general_chat(user_query);
            print(f"Answer: {response}");
            return

        print("Step 1: Selecting the best tool with SLM Classifier...")
        processed_query = user_query.replace('^', '**')
        processed_query = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', processed_query)
        selected_tool_name = self.tool_classifier.select_tool(processed_query, available_tools)
        print(f"[DEBUG] Selected tool: '{selected_tool_name}'")

        if selected_tool_name == "chat":
            response = self._handle_general_chat(user_query)
        else:
            tool_to_execute = next((t for t in available_tools if t['name'] == selected_tool_name), None)
            if not tool_to_execute:
                response = self._handle_general_chat(user_query);
                print(f"\n[Tool '{selected_tool_name}' not found, switching to general chat]")
            else:
                tool_params = None
                print("Step 2: Attempting parameter extraction...")

                # ✨ --- 최종 동적 로직 --- ✨
                parameters = tool_to_execute.get('parameters', {}).get('properties', {})

                # 규칙 기반 추출이 가능한 파라미터가 있는지 확인
                extracted_rule_params = {}
                requires_slm = False

                for param_name, param_details in parameters.items():
                    param_type = param_details.get("type")
                    if param_type in self.strategy_by_type:
                        extractor_func = self.strategy_by_type[param_type]
                        print(
                            f"   - Strategy: Found hint for '{param_name}' (type: {param_type}). Using extractor: {extractor_func.__name__}")
                        extracted_value = extractor_func(processed_query)
                        if extracted_value:
                            extracted_rule_params[param_name] = extracted_value
                    else:
                        # 컨트롤 타워에 없는 타입은 SLM 처리가 필요하다고 표시
                        requires_slm = True

                # 규칙으로 모든 파라미터를 찾았고, SLM이 필요 없다면 바로 사용
                if extracted_rule_params and not requires_slm:
                    tool_params = extracted_rule_params
                else:
                    # 규칙으로 일부만 찾았거나, SLM이 필요한 파라미터가 있다면 SLM 호출
                    print("   - Strategy: No suitable rule for all params found. Falling back to SLM.")
                    slm_params = self.param_extractor.extract(processed_query, tool_to_execute)
                    # 규칙 기반 결과와 SLM 결과를 합침 (SLM 결과를 우선)
                    tool_params = {**extracted_rule_params, **slm_params}

                print(f"[DEBUG] Extracted parameters: {tool_params}")

                if not tool_params and parameters:
                    response = self._handle_general_chat(user_query);
                    print(f"\n[Parameter extraction failed for '{selected_tool_name}', switching to general chat]")
                else:
                    tool_result = await self._execute_tool(selected_tool_name, tool_params or {})
                    if tool_result.get("success"):
                        response = self._generate_natural_response(user_query, tool_result["output"],
                                                                   selected_tool_name)
                        print(f"\n[Tool Used: {selected_tool_name}]")
                    else:
                        response = self._handle_general_chat(user_query);
                        print(f"\n[Tool execution failed, switching to general chat]")

        print(f"Answer: {response}")


# --- main_async and main functions are the same ---
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
