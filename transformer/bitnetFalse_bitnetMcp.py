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


# --- Transformers 모델 래퍼 클래스 (배치 처리 기능 추가) ---
class TransformersSLM:
    def __init__(self, model_id: str, dtype: torch.dtype):
        print(f"'{model_id}' 모델과 토크나이저를 로딩합니다...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available, but this setup requires a GPU.")
        print("CUDA 감지됨. GPU에 모델을 로딩합니다...")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
        print("✅ 모델 로딩 완료.")

    def query_batch(self, prompts: List[str], max_new_tokens: int = 512, temperature: float = 0.1) -> List[str]:
        """ 여러 프롬프트를 배치로 처리하여 속도를 최적화합니다. """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(
            self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                                      pad_token_id=self.tokenizer.pad_token_id)
        # 생성된 텍스트 부분만 잘라냅니다.
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        decoded_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # 각 응답에서 첫 줄만 사용하고 공백을 제거합니다.
        return [resp.split('\n')[0].strip() for resp in decoded_responses]


# --- 핵심 로직 클래스 (기존과 동일) ---
class MathSolver:
    def __init__(self, config: AppConfig, slm: TransformersSLM):
        self.config = config
        self.slm = slm

    def _execute_tool_sync(self, tool_name: str, tool_params: Dict[str, Any]) -> Dict[str, Any]:
        async def _call():
            try:
                async with Client(self.config.server_url) as client:
                    result = await client.call_tool(tool_name, tool_params)
                    output = str(getattr(result, 'data', result))
                    return {"success": True, "output": output}
            except Exception as e:
                return {"success": False, "error": f"Tool execution failed: {e}"}

        return asyncio.run(_call())

    # ★★★ 배치 처리 로직으로 최적화된 테스트 메소드 ★★★
    def test_gsm8k_sync(self, max_samples: int, csv_filename: str, batch_size: int):
        input_csv_path = 'pure_bitnet_false.csv'
        test_samples = []
        try:
            with open(input_csv_path, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    test_samples.append(row)
            print(f"'{input_csv_path}' 파일에서 {len(test_samples)}개의 샘플을 로드했습니다.")
        except FileNotFoundError:
            print(f"에러: 입력 파일 '{input_csv_path}'을 찾을 수 없습니다.")
            return

        num_samples_to_process = min(max_samples, len(test_samples))

        all_results = []
        start_time = time.time()

        fieldnames = ["sample_id", "question", "expression", "predicted", "ground_truth", "correct", "status"]

        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            print(f"CSV 결과 파일 '{csv_filename}'이 초기화되었습니다. (배치 크기: {batch_size})")

            # 배치 단위로 루프 실행
            for i in range(0, num_samples_to_process, batch_size):
                batch_samples = test_samples[i:i + batch_size]
                if not batch_samples:
                    continue

                print(
                    f"\n{'=' * 20} Processing batch {i // batch_size + 1} (samples {i + 1}-{i + len(batch_samples)}) {'=' * 20}")

                # 1. 배치 프롬프트 생성 및 LLM을 통해 표현식 일괄 생성
                batch_prompts = [QUERY_TO_EXPRESSION_PROMPT.format(query=s.get("question", "")) for s in batch_samples]
                raw_expressions = self.slm.query_batch(batch_prompts)

                # 2. 배치 결과 순회하며 채점
                for sample, raw_expression in zip(batch_samples, raw_expressions):
                    sample_id = sample.get("simple_id", "N/A")
                    question = sample.get("question", "")
                    predicted, status, is_correct = None, "", False
                    ground_truth = None

                    if raw_expression and re.match(r'^[0-9\(\s][0-9\s\.\+\-\*\/\(\)]+[0-9\)\s]$', raw_expression,
                                                   re.MULTILINE):
                        tool_res = self._execute_tool_sync("calculate", {"expression": raw_expression})

                        if tool_res.get("success"):
                            try:
                                parsed_output = ast.literal_eval(tool_res['output'].strip())

                                if isinstance(parsed_output, dict):
                                    if 'result' in parsed_output:
                                        predicted = float(parsed_output['result'])
                                        status = "SUCCESS"
                                    else:
                                        status = f"FAIL (Key 'result' not in response: {parsed_output})"
                                else:
                                    predicted = float(parsed_output)
                                    status = "SUCCESS"

                                if status == "SUCCESS":
                                    try:
                                        ground_truth = float(sample.get("ground_truth"))
                                        is_correct = abs(predicted - ground_truth) < 1e-3
                                    except (ValueError, TypeError):
                                        status = "FAIL (Invalid Ground Truth in CSV)"

                            except (ValueError, SyntaxError, TypeError) as e:
                                status = f"FAIL (Could not parse server response: {tool_res['output']}) - {e}"
                        else:
                            status = f"FAIL (Tool Execution Error: {tool_res.get('error', 'Unknown')})"
                    else:
                        status = "FAIL (Invalid Expression)"

                    if ground_truth is None:
                        try:
                            ground_truth = float(sample.get("ground_truth"))
                        except (ValueError, TypeError):
                            ground_truth = "N/A"

                    print(
                        f"  - Sample {sample_id}: Predicted={predicted}, GT={ground_truth}, Correct={is_correct}, Status={status}")

                    result_row = {"sample_id": sample_id, "question": question, "expression": raw_expression,
                                  "predicted": predicted, "ground_truth": ground_truth, "correct": is_correct,
                                  "status": status}
                    all_results.append(result_row)
                    writer.writerow(result_row)

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


# --- 메인 실행 함수 (배치 크기 인자 추가) ---
def main_sync():
    parser = argparse.ArgumentParser(description="MCP Client for Math Problems (Synchronous, Batch-processed mode).")
    parser.add_argument("--test-problems", action="store_true", help="Run test on custom CSV and save results")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples to process from CSV")
    parser.add_argument("--csv-filename", type=str, default="results_pure_bitnet_false_rerun.csv",
                        help="Filename for the output results")
    # ★★★ 속도 최적화를 위한 배치 크기 인자 추가 ★★★
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for model inference")
    args = parser.parse_args()

    print("=" * 60)
    print(" MCP Client - Math Solver (Synchronous, Batch-processed mode)")
    print("=" * 60)

    try:
        config = AppConfig()
        slm = TransformersSLM(config.model_id, config.torch_dtype)
        solver = MathSolver(config, slm)

        if args.test_problems:
            solver.test_gsm8k_sync(max_samples=args.max_samples, csv_filename=args.csv_filename,
                                   batch_size=args.batch_size)
        else:
            print("No action specified. Use --test-problems to run the benchmark.")

    except Exception as e:
        print(f"\nA critical error occurred: {e}")
        import traceback
        traceback.print_exc()


def main():
    try:
        main_sync()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")


if __name__ == "__main__":
    main()