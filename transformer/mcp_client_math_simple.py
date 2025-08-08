import argparse
import asyncio
import re
import ast
import time
import csv
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastmcp import Client
from datasets import load_dataset

# --- 프롬프트 (기존과 동일) ---
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


# --- 설정 클래스 (기존과 동일) ---
@dataclass
class AppConfig:
    server_url: str = "http://localhost:8001/sse"
    model_id: str = "microsoft/bitnet-b1.58-2B-4T"
    torch_dtype: torch.dtype = torch.bfloat16


# --- Transformers 모델 래퍼 클래스 (기존과 동일) ---
class TransformersSLM:
    def __init__(self, model_id: str, dtype: torch.dtype):
        print(f"'{model_id}' 모델과 토크나이저를 로딩합니다...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available, but this setup requires a GPU.")
        print("CUDA 감지됨. GPU에 모델을 로딩합니다...")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
        print("✅ 모델 로딩 완료.")

    # 이름은 query_batch이지만, 이제 한 번에 하나씩 처리하는 데 사용됩니다.
    def query_one(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(
            self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                                      pad_token_id=self.tokenizer.pad_token_id)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        decoded_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_responses[0].split('\n')[0].strip()


# --- 핵심 로직 클래스 (동기/단일 처리 방식으로 변경) ---
class MathSolver:
    def __init__(self, config: AppConfig, slm: TransformersSLM):
        self.config = config
        self.slm = slm

    # ★★★ 비동기에서 동기 방식으로 변경된 도구 실행 함수 ★★★
    def _execute_tool_sync(self, tool_name: str, tool_params: Dict[str, Any]) -> Dict[str, Any]:
        # 내부적으로 비동기 라이브러리를 동기적으로 호출하기 위해 asyncio.run 사용
        async def _call():
            try:
                async with Client(self.config.server_url) as client:
                    result = await client.call_tool(tool_name, tool_params)
                    output = str(getattr(result, 'data', result))
                    return {"success": True, "output": output}
            except Exception as e:
                return {"success": False, "error": f"Tool execution failed: {e}"}

        return asyncio.run(_call())

    # ★★★ 배치 및 비동기 로직이 제거된 새로운 테스트 메소드 ★★★
    def test_gsm8k_sync(self, max_samples: int, csv_filename: str):
        gsm8k = load_dataset("gsm8k", "main")
        num_samples_to_process = min(max_samples, len(gsm8k["test"]))
        test_set = gsm8k["test"].select(range(num_samples_to_process))

        all_results = []
        start_time = time.time()

        fieldnames = ["sample_id", "question", "expression", "predicted", "ground_truth", "correct", "status"]

        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            print(f"CSV 파일 '{csv_filename}'이 초기화되었습니다.")

            # 한 번에 한 샘플씩 순차적으로 처리하는 루프
            for i, sample in enumerate(test_set):
                if i >= num_samples_to_process:
                    break

                sample_id = i + 1
                question = sample["question"]
                print(f"\n{'=' * 20} Processing sample {sample_id}/{num_samples_to_process} {'=' * 20}")

                # 1. LLM을 통해 표현식 생성
                prompt = QUERY_TO_EXPRESSION_PROMPT.format(query=question)
                raw_expression = self.slm.query_one(prompt)

                predicted, status, is_correct = None, "", False

                # mcp_client_math_simple.py 파일의 test_gsm8k_sync 함수 내부를 수정하세요.

                # 2. 표현식 유효성 검사 및 도구 실행 (동기)
                if raw_expression and re.match(r'^[0-9\(\s][0-9\s\.\+\-\*\/\(\)]+[0-9\)\s]$', raw_expression,
                                               re.MULTILINE):
                    tool_res = self._execute_tool_sync("calculate", {"expression": raw_expression})

                    if tool_res.get("success"):
                        try:
                            parsed_output = ast.literal_eval(tool_res['output'].strip())

                            # Case 1: 응답이 딕셔너리 형태일 경우
                            if isinstance(parsed_output, dict):
                                # 'result' 키가 있는지 먼저 "확인"하고, 있을 때만 값을 가져옵니다.
                                if 'result' in parsed_output:
                                    predicted = float(parsed_output['result'])
                                    status = "SUCCESS"
                                else:
                                    # 'result' 키가 없는 딕셔너리는 처리 실패로 간주합니다.
                                    status = f"FAIL (Key 'result' not in response: {parsed_output})"

                            # Case 2: 응답이 딕셔너리가 아닌 일반 숫자나 문자열일 경우
                            else:
                                predicted = float(parsed_output)
                                status = "SUCCESS"

                            # 성공적으로 값을 얻었을 때만 정답 여부를 확인합니다.
                            if status == "SUCCESS":
                                gt_match = re.search(r"####[ ]*(-?[0-9,]+\.?[0-9]*)", sample["answer"])
                                ground_truth = float(gt_match.group(1).replace(",", "")) if gt_match else None
                                if ground_truth is not None:
                                    is_correct = abs(predicted - ground_truth) < 1e-3

                        except (ValueError, SyntaxError, TypeError):
                            status = f"FAIL (Could not parse server response: {tool_res['output']})"
                        # ==========================================================
                    else:
                        status = f"FAIL (Tool Execution Error: {tool_res.get('error', 'Unknown')})"
                else:
                    status = "FAIL (Invalid Expression)"

                # --- 터미널에 상세 로그 출력 ---
                gt_match = re.search(r"####[ ]*(-?[0-9,]+\.?[0-9]*)", sample["answer"])
                ground_truth = float(gt_match.group(1).replace(",", "")) if gt_match else None
                print("-" * 60)
                print(f"Question: {question[:150]}...")
                print(f"LLM Expression: '{raw_expression}'")
                print(f"Predicted: {predicted} | Ground Truth: {ground_truth}")
                print(f"Status: {status} | Correct: {is_correct}")

                result_row = {"sample_id": sample_id, "question": question, "expression": raw_expression,
                              "predicted": predicted, "ground_truth": ground_truth, "correct": is_correct,
                              "status": status}
                all_results.append(result_row)
                writer.writerow(result_row)  # 한 줄씩 바로 쓰기

        # Final summary
        end_time = time.time()
        print(f"\n--- Test Finished ---")
        actual_processed_samples = len(all_results)
        if actual_processed_samples > 0:
            correct_count = sum(1 for r in all_results if r['correct'])
            accuracy = (correct_count / actual_processed_samples * 100)
            print(f"Final results are fully saved to '{csv_filename}'")
            print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{actual_processed_samples})")
            print(f"Total time for {actual_processed_samples} samples: {end_time - start_time:.2f} seconds")
            print(f"Samples per second: {actual_processed_samples / (end_time - start_time):.2f}")
        else:
            print("No samples were processed.")


# --- 메인 실행 함수 (동기 방식으로 변경) ---
def main_sync():
    parser = argparse.ArgumentParser(description="MCP Client for GSM8K (Synchronous, Single-process mode).")
    parser.add_argument("--test-gsm8k", action="store_true", help="Run GSM8K test and save results to CSV")
    parser.add_argument("--max-samples", type=int, default=1319, help="Maximum samples to process for GSM8K test")
    parser.add_argument("--csv-filename", type=str, default="gsm8k_results_sync_single.csv",
                        help="Filename for GSM8K results")
    # 배치 사이즈 인자 제거
    args = parser.parse_args()

    print("=" * 60)
    print(" MCP Client - Math Solver (Synchronous, Single-process mode)")
    print("=" * 60)

    try:
        config = AppConfig()
        slm = TransformersSLM(config.model_id, config.torch_dtype)
        solver = MathSolver(config, slm)

        if args.test_gsm8k:
            solver.test_gsm8k_sync(max_samples=args.max_samples, csv_filename=args.csv_filename)
        else:
            print("No action specified. Use --test-gsm8k to run the benchmark.")

    except Exception as e:
        print(f"\nA critical error occurred: {e}")
        import traceback
        traceback.print_exc()


def main():
    try:
        main_sync()  # asyncio.run 제거
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")


if __name__ == "__main__":
    main()