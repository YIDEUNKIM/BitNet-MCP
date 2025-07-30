import argparse
import asyncio
import json
import ast
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastmcp import Client

# --- 설정 클래스 ---
@dataclass
class AppConfig:
    """transformers 기반 클라이언트의 설정을 관리합니다."""
    server_url: str = "http://localhost:8000/sse"
    model_id: str = "microsoft/bitnet-b1.58-2B-4T"
    system_prompt: str = "You are a Intelligent AI Agent."
    # torch_dtype은 bfloat16이 권장되나, 환경에 따라 float16 또는 float32로 조정
    torch_dtype: torch.dtype = torch.bfloat16

# --- Transformers 모델 래퍼 클래스 ---
class TransformersSLM:
    """Hugging Face Transformers 모델의 로딩 및 추론을 관리합니다."""
    def __init__(self, model_id: str, dtype: torch.dtype):
        print(f"'{model_id}' 모델과 토크나이저를 로딩합니다...")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto"  # GPU 자동 할당
        )
        print("✅ 모델 로딩 완료.")

    def query(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        """주어진 프롬프트로 텍스트를 생성하고, 첫 줄만 반환하여 후속 생성을 방지합니다."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )

        full_response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

        # 생성된 전체 텍스트에서 첫 번째 줄바꿈까지만 결과로 인정
        first_line = full_response.split('\n')[0]
        return first_line.strip()

# --- 도구 선택 클래스 (기존과 거의 동일) ---
class SLMToolClassifier:
    """SLM을 사용하여 사용자 쿼리에 가장 적합한 도구를 선택합니다."""
    def __init__(self, slm_query_func):
        self.slm_query_func = slm_query_func

    def select_tool(self, user_query: str, available_tools: List[Dict]) -> str:
        tool_descriptions = "\n".join([
            f"- tool name: {tool['name']}\n  description: {tool.get('description', '')}"
            for tool in available_tools
        ])

        prompt = f"""[System Role]
You are a tool-selection expert. Your ONLY job is to analyze the user's query and the tool descriptions to identify the single best tool.
Respond with ONLY the exact 'tool name'. If no tool fits, return "chat". Do not provide any explanation.

[Available Tools]
- tool name: chat
  description: Use for general conversation or when no other tool is appropriate.
{tool_descriptions}

[User Query]
{user_query}

Your one-word response (the exact Tool Name):"""

        slm_output = self.slm_query_func(prompt, max_new_tokens=20).strip().split()[0]
        print(f"Initial SLM tool selection: '{slm_output}'")

        valid_tool_names = [tool['name'] for tool in available_tools] + ['chat']
        if slm_output in valid_tool_names:
            return slm_output

        for name in valid_tool_names:
            if name.startswith(slm_output):
                print(f"[DEBUG] Corrected incomplete response '{slm_output}' to '{name}'")
                return name

        return "chat"

# --- 업그레이드된 파라미터 추출 클래스 ---
class SLMParameterExtractor:
    """SLM을 사용하여 도구에 필요한 파라미터를 JSON 형식으로 추출합니다."""
    def __init__(self, slm_query_func):
        self.slm_query_func = slm_query_func

    def extract(self, user_query: str, tool: Dict) -> Dict:
        params_schema = tool.get('parameters', {})
        if not params_schema.get('properties'):
            return {}

        prompt = f"""[System Role]
You are a parameter extraction expert. Your task is to extract the values for the parameters required by a tool from the user's query.
Respond with ONLY a valid JSON object containing the extracted parameters. If a parameter cannot be found, do not include it in the JSON.

[Tool Name]
{tool['name']}

[Tool Description]
{tool.get('description', '')}

[Parameter Schema (JSON)]
{json.dumps(params_schema, indent=2)}

[User Query]
{user_query}

Your response (JSON object only):

"""

        # 모델이 JSON을 생성할 충분한 토큰을 할당
        raw_response = self.slm_query_func(prompt, max_new_tokens=100)

        # 모델이 생성한 텍스트에서 JSON 객체만 정확히 추출
        try:
            # 모델 응답에서 ```json ... ``` 코드 블록 찾기
            match = ast.literal_eval(raw_response)
            if isinstance(match, dict):
                return match
        except (ValueError, SyntaxError, TypeError):
            print(f"[DEBUG] Failed to parse SLM response as JSON: '{raw_response}'")
            return {}

        return {}


# --- 메인 로직을 담당하는 라우터 클래스 ---
class ToolRouter:
    """Transformers 모델을 사용하여 동적 도구 선택 및 실행을 관리합니다."""
    def __init__(self, config: AppConfig, slm: TransformersSLM):
        self.config = config
        self.slm = slm
        self.tool_classifier = SLMToolClassifier(self.slm.query)
        self.param_extractor = SLMParameterExtractor(self.slm.query)
        self._tools_cache = None

    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        if self._tools_cache:
            return self._tools_cache
        try:
            async with Client(self.config.server_url) as client:
                tools = await client.list_tools()
                self._tools_cache = [
                    {"name": t.name, "description": t.description, "parameters": getattr(t, 'parameters', {})}
                    for t in tools
                ]
                print(f"\n[DEBUG] Fetched tools from server: {self._tools_cache}\n")
                return self._tools_cache
        except Exception as e:
            print(f"MCP 서버에서 도구 목록을 가져오는 데 실패했습니다: {e}")
            return []

    async def _execute_tool(self, tool_name: str, tool_params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with Client(self.config.server_url) as client:
                print(f"Executing tool '{tool_name}' with params: {tool_params}")
                result = await client.call_tool(tool_name, tool_params)
                output = str(getattr(result, 'data', result))
                return {"success": True, "output": output}
        except Exception as e:
            return {"success": False, "error": f"Tool execution failed: {e}"}

    def _generate_natural_response(self, query: str, tool_output: str, tool_name: str) -> str:
        prompt = f"""[System]
You are a helpful assistant who summarizes tool results into natural, user-friendly language.

[User Question]
{query}

[Tool Used]
{tool_name}

[Tool Execution Result]
{tool_output}

[Instruction]
Based on the provided tool execution result, answer the user's original question in a complete and friendly English sentence.
For example, if the result is "location: seoul, temperature: 25, condition: clean", your response should be like "Currently, the weather in Seoul is clean and the temperature is 25 degrees."

[Final Answer]

"""
        return self.slm.query(prompt, max_new_tokens=150)

    def _handle_general_chat(self, query: str) -> str:
        prompt = f"""[System]
    {self.config.system_prompt}
    [User's Original Question]
    {query}

    [Final Answer]
    
    """
        response = self.slm.query(prompt, max_new_tokens=60)
        # 첫 줄만 반환 (중복 인사/텍스트 방지)
        return response.split('\n')[0].strip()

    async def route_query(self, user_query: str):
        available_tools = await self._get_available_tools()
        response = None  # 답변 변수 미리 선언

        if not available_tools:
            response = self._handle_general_chat(user_query)
        else:
            selected_tool_name = self.tool_classifier.select_tool(user_query, available_tools)
            if selected_tool_name != "chat":
                tool_to_execute = next((t for t in available_tools if t['name'] == selected_tool_name), None)
                if tool_to_execute:
                    tool_params = self.param_extractor.extract(user_query, tool_to_execute)
                    tool_result = await self._execute_tool(selected_tool_name, tool_params)
                    if tool_result.get("success"):
                        response = self._generate_natural_response(user_query, tool_result["output"],
                                                                   selected_tool_name)
                    else:
                        response = self._handle_general_chat(user_query)
                else:
                    response = self._handle_general_chat(user_query)
            else:
                response = self._handle_general_chat(user_query)

        # ↑ 위 논리의 실행 결과로 response에 한 번만 답변 저장
        print(f"\nFinal Answer: {response}")  # 단 한 번만 출력

# --- 메인 실행 함수 ---
async def main_async():
    parser = argparse.ArgumentParser(description='MCP Client with Transformers-based SLM')
    parser.add_argument("-q", "--query", type=str, help="User query to process", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  MCP Client - Dynamic Tool Selection with Transformers SLM")
    print("=" * 60)

    try:
        config = AppConfig()
        slm = TransformersSLM(config.model_id, config.torch_dtype)
        router = ToolRouter(config, slm)

        user_input = args.query or input("\nEnter your question: ")
        await router.route_query(user_input)

    except Exception as e:
        print(f"A critical error occurred: {e}")

    print("\nExecution complete.")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()