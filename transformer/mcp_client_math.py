import argparse
import asyncio
import json
import ast
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastmcp import Client
import csv
from datasets import load_dataset

# --- Prompts (Copied verbatim from gsm8k_tester.py) ---

QUERY_TO_EXPRESSION_PROMPT = '''[System]
You are an expert at solving grade school math problems.
Below are some examples that show a step-by-step reasoning process that leads to the final Python expression.
Your final output MUST be ONLY the Python expression.

**CRITICAL: Output ONLY the final Python expression. DO NOT include any analysis, variables, explanations, or additional text. AVOID REPETITION in the expression – use efficient math like sums or loops if needed, but keep it concise.**

---

[Example 1]
User Problem: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
Reasoning: The total is the sum of clips sold in April (48) and May. Clips sold in May were half of April's, which is 48 / 2. So the expression is the sum of 48 and (48/2).
Final Python Expression: 48 + (48 / 2)

---

[Example 2]
User Problem: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
Reasoning: To find the earnings for 50 minutes, we first need the per-minute rate. The hourly rate of $12 is divided by 60 minutes. Then, this per-minute rate is multiplied by 50.
Final Python Expression: (12 / 60) * 50

---

[Example 3]
User Problem: "A deep-sea monster has a hoard of 350 gold coins. It sinks a ship carrying 130 gold coins and another ship carrying 80 gold coins. How many gold coins does the monster have in its hoard now?"
Reasoning: The monster's new hoard is the sum of its initial coins (350), the coins from the first ship (130), and the coins from the second ship (80).
Final Python Expression: 350 + 130 + 80

---

[Example 4]
User Problem: "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
Reasoning: The target is $100. Betty has half, which is 50. Her parents give her 15. Her grandparents give twice that, which is 15 * 2 = 30. To find how much more she needs, we subtract all the money she has (50, 15, and 30) from the target price of 100.
Final Python Expression: 100 - 50 - 15 - 30

---

[User Problem]
{query}
Final Python Expression:'''



# SYNTHESIZE_ANSWER_PROMPT is no longer needed in the recommended version.

# --- 설정 클래스 ---
@dataclass
class AppConfig:
    """transformers 기반 클라이언트의 설정을 관리합니다."""
    server_url: str = "http://localhost:8001/sse"
    model_id: str = "microsoft/bitnet-b1.58-2B-4T"
    system_prompt: str = "You are a Intelligent AI Agent."
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
            device_map="cuda"
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
        first_line = full_response.split('\n')[0]
        return first_line.strip()


# --- 도구 선택 클래스 ---
class SLMToolClassifier:
    """SLM을 사용하여 사용자 쿼리에 가장 적합한 도구를 선택합니다."""

    def __init__(self, slm_query_func):
        self.slm_query_func = slm_query_func

    def select_tool(self, user_query: str, available_tools: List[Dict]) -> str:
        tool_descriptions = "\n".join([
            f"- tool name: {tool['name']}\n description: {tool.get('description', '')}"
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


# --- 파라미터 추출 클래스 ---
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
        raw_response = self.slm_query_func(prompt, max_new_tokens=100)
        try:
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
        return response.split('\n')[0].strip()

    async def route_query(self, user_query: str):
        available_tools = await self._get_available_tools()
        response = None
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
        print(f"\nFinal Answer: {response}")

    # ===== ★★★★★ 수정된 부분 ★★★★★ =====
    async def solve_math(self, query: str) -> Optional[float]:
        # Step 1: Query to Python expression using SLM
        prompt = QUERY_TO_EXPRESSION_PROMPT.format(query=query)
        expression = self.slm.query(prompt, max_new_tokens=256, temperature=0.1).strip()

        # LLM이 이상한 응답을 했을 경우를 대비한 방어 코드
        if not expression or not re.match(r'^[0-9\(\s][0-9\s\.\+\-\*\/\(\)]+[0-9\)\s]$', expression, re.MULTILINE):
            print(f" - Invalid or empty expression from LLM: '{expression}'")
            return None

        # Step 2: Execute calculation via MCP tool
        print(f" - Calling MCP tool server to execute: {expression}")
        calc_result_dict = await self._execute_tool("calculate", {"expression": expression})

        if "error" in calc_result_dict or "success" not in calc_result_dict or not calc_result_dict["success"]:
            print(f" - Calculation failed: {calc_result_dict.get('error', 'Unknown error')}")
            return None

        # Step 3: Directly use the calculation result
        result_value_str = calc_result_dict.get('output')
        try:
            # MCP 서버의 응답이 {'result': 18.0} 같은 딕셔너리 형태의 문자열일 수 있음
            parsed_output = ast.literal_eval(result_value_str.strip())
            if isinstance(parsed_output, dict) and 'result' in parsed_output:
                predicted_value = float(parsed_output['result'])
            else:
                # 단순 숫자 문자열일 경우
                predicted_value = float(result_value_str.strip())

            print(f" - MCP Result Parsed: {predicted_value}")
            return predicted_value

        except (ValueError, SyntaxError, TypeError) as e:
            print(f" - Invalid MCP result format: {result_value_str} (Error: {e})")
            return None

    # ========================================

    async def test_gsm8k(self, max_samples: int = 100, csv_filename: str = "gsm8k_results.csv"):
        gsm8k = load_dataset("gsm8k", "main")
        test_set = gsm8k["test"]
        results = []

        for i, sample in enumerate(test_set):
            if i >= max_samples:
                break

            print(f"\n--- Processing sample {i + 1}/{max_samples} ---")
            question = sample["question"]
            match = re.search(r"####[ ]*(-?[0-9,]+\.?[0-9]*)", sample["answer"], re.IGNORECASE | re.DOTALL)
            ground_truth = float(match.group(1).replace(",", "")) if match else None

            predicted = await self.solve_math(question)
            is_correct = (predicted is not None and ground_truth is not None and abs(predicted - ground_truth) < 1e-3)

            results.append({
                "sample_id": i + 1,
                "question": question,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": is_correct
            })
            print(
                f"Result for sample {i + 1}: Correct={is_correct}, Predicted={predicted}, Ground Truth={ground_truth}")

        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "question", "predicted", "ground_truth", "correct"])
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to {csv_filename}")
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = (correct_count / len(results) * 100) if results else 0
        print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(results)})")


# --- 메인 실행 함수 ---
async def main_async():
    parser = argparse.ArgumentParser(description='MCP Client with Transformers-based SLM')
    parser.add_argument("-q", "--query", type=str, help="User query to process", default=None)
    parser.add_argument("--test-gsm8k", action="store_true", help="Run GSM8K test and save results to CSV")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum samples to process for GSM8K test")
    parser.add_argument("--csv-filename", type=str, default="gsm8k_results_simplified.csv",
                        help="Filename for GSM8K results CSV")
    args = parser.parse_args()

    print("=" * 60)
    print(" MCP Client - Dynamic Tool Selection with Transformers SLM")
    print("=" * 60)

    try:
        config = AppConfig()
        slm = TransformersSLM(config.model_id, config.torch_dtype)
        router = ToolRouter(config, slm)

        if args.test_gsm8k:
            await router.test_gsm8k(max_samples=args.max_samples, csv_filename=args.csv_filename)
        elif args.query:
            # 간단한 키워드 체크로 수학 문제인지 판별
            if "math" in args.query.lower() or any(c.isdigit() for c in args.query):
                result = await router.solve_math(args.query)
                print(f"\nMath Result: {result}")
            else:
                await router.route_query(args.query)
        else:
            user_input = input("\nEnter your question: ")
            if "math" in user_input.lower() or any(c.isdigit() for c in user_input):
                result = await router.solve_math(user_input)
                print(f"\nMath Result: {result}")
            else:
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

