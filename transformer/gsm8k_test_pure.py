import argparse
import csv
import re
from dataclasses import dataclass
from typing import Optional, List, Dict
import time  # 시간 측정을 위해 추가

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# --- 1. 설정 클래스 (batch_size 추가) ---
@dataclass
class TestConfig:
    """테스트 실행을 위한 모든 설정을 관리합니다."""
    model_id: str = "microsoft/bitnet-b1.58-2B-4T"
    torch_dtype: torch.dtype = torch.bfloat16
    max_samples: int = 1319
    output_csv: str = "gsm8k_results_pure_4shot_batch.csv"
    batch_size: int = 16  # 배치 사이즈 추가
    # 생성 관련 파라미터
    max_new_tokens: int = 512
    temperature: float = 0.1


# --- 2. 프롬프트 구성 (변경 없음) ---
def get_few_shot_examples() -> List[Dict[str, str]]:
    # ... (내용 동일)
    return [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": "Natalia sold 48 clips in April. In May, she sold half as many clips as in April, which is 48/2 = 24 clips. Therefore, in total, she sold 48 + 24 = 72 clips. #### 72"},
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": "Weng earns $12 per hour, which is equivalent to 60 minutes. So, her earning per minute is $12/60 = $0.2. For 50 minutes of babysitting, she earned 50 * $0.2 = $10. #### 10"},
        {
            "question": "A deep-sea monster rises from the waters once every hundred years to feast on a ship and adds the ships' treasure to its hoard. Before it awakens, it has a hoard of 350 gold coins. It sinks a ship carrying 130 gold coins and another ship carrying 80 gold coins. How many gold coins does the monster have in its hoard now?",
            "answer": "The monster starts with 350 gold coins. It adds 130 gold coins from the first ship, so it has 350 + 130 = 480 coins. Then it adds 80 gold coins from the second ship, so it has 480 + 80 = 560 coins. #### 560"},
        {
            "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
            "answer": "The wallet costs $100. Betty has half of the money she needs, which is $100 / 2 = $50. Her parents give her $15. Her grandparents give her twice as much as her parents, which is $15 * 2 = $30. In total, Betty has $50 + $15 + $30 = $95. She needs $100 - $95 = $5 more. #### 5"}
    ]


def create_prompt(examples: List[Dict[str, str]], test_question: str) -> str:
    # ... (내용 동일)
    prompt_parts = [
        "Solve the following grade school math problems. Show your reasoning step-by-step and end your answer with '#### <number>'."]
    for ex in examples: prompt_parts.append(f"\n\nQuestion: {ex['question']}\nAnswer: {ex['answer']}")
    prompt_parts.append(f"\n\nQuestion: {test_question}\nAnswer:")
    return "".join(prompt_parts)


# --- 3. 핵심 로직 및 헬퍼 함수 (수정됨) ---
def extract_answer(text: str) -> Optional[float]:
    # ... (내용 동일)
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", str(text))
    if match:
        try:
            cleaned_number = match.group(1).replace(",", "")
            return float(cleaned_number)
        except (ValueError, TypeError):
            return None
    return None


class PureBitnetTester:
    def __init__(self, config: TestConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"'{self.config.model_id}' 모델과 토크나이저를 로딩합니다...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, padding_side='left')
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available. This script requires a GPU.")
        print("CUDA 감지됨. 모델의 내장 양자화 설정을 사용하여 GPU에 로딩합니다...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, torch_dtype=self.config.torch_dtype, device_map="cuda"
        )
        print("✅ 모델 로딩 완료.")

    # ===== ★★★★★ 배치 처리를 위한 새로운 메소드 ★★★★★ =====
    def run_test_batch(self):
        """GSM8K 테스트를 '배치'로 실행하여 성능을 극대화합니다."""
        gsm8k = load_dataset("gsm8k", "main")
        test_set = gsm8k["test"].select(range(self.config.max_samples))
        few_shot_examples = get_few_shot_examples()
        results = []

        print(f"\n--- 총 {len(test_set)}개의 샘플에 대해 '배치 모드(batch_size={self.config.batch_size})'로 테스트를 시작합니다 ---")
        start_time = time.time()

        # 배치 단위로 루프 실행
        for i in tqdm(range(0, len(test_set), self.config.batch_size), desc="GSM8K 배치 테스트 진행 중"):
            batch = test_set[i: i + self.config.batch_size]
            batch_questions = batch["question"]

            # 1. 배치 전체에 대한 프롬프트 생성
            batch_prompts = [create_prompt(few_shot_examples, q) for q in batch_questions]

            # 2. 모델 배치 추론
            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )
            print(outputs)
            # 3. 배치 결과 처리
            full_model_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for j, full_output in enumerate(full_model_outputs):
                answer_part = full_output[len(batch_prompts[j]):].strip()
                predicted_answer = extract_answer(answer_part)
                ground_truth_answer = extract_answer(batch["answer"][j])

                is_correct = (predicted_answer is not None and
                              ground_truth_answer is not None and
                              abs(predicted_answer - ground_truth_answer) < 1e-3)
                print(answer_part)
                results.append({
                    "question": batch_questions[j], "full_model_output": answer_part,
                    "predicted": predicted_answer, "ground_truth": ground_truth_answer,
                    "correct": is_correct
                })

        end_time = time.time()
        print(f"\n총 처리 시간: {end_time - start_time:.2f} 초")
        self._save_results(results, end_time - start_time)

    def _save_results(self, results: List[Dict], duration: float = 0.0):
        """테스트 결과를 CSV 파일로 저장하고 정확도 및 성능을 출력합니다."""
        if not results:
            print("처리된 결과가 없습니다.")
            return

        with open(self.config.output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["question", "full_model_output", "predicted", "ground_truth",
                                                   "correct"])
            writer.writeheader()
            writer.writerows(results)

        correct_count = sum(1 for r in results if r['correct'])
        total_samples = len(results)
        accuracy = (correct_count / total_samples * 100) if total_samples > 0 else 0

        print("\n--- 테스트 완료 ---")
        print(f"결과가 '{self.config.output_csv}' 파일에 저장되었습니다.")
        print(f"정확도: {accuracy:.2f}% ({correct_count}/{total_samples})")
        if duration > 0:
            print(f"초당 처리 샘플 수: {total_samples / duration:.2f} samples/sec")


# --- 4. 메인 실행 블록 (수정됨) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure Transformers-based GSM8K 4-shot tester for BitNet.")
    parser.add_argument("--max_samples", type=int, default=1319, help="테스트할 최대 샘플 수")
    parser.add_argument("--batch_size", type=int, default=16, help="추론을 위한 배치 사이즈")
    parser.add_argument("--output_csv", type=str, default="gsm8k_results_pure_4shot_batch.csv",
                        help="결과를 저장할 CSV 파일 이름")
    args = parser.parse_args()

    config = TestConfig(
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_csv=args.output_csv
    )

    try:
        tester = PureBitnetTester(config)
        # 배치 처리 메소드 호출
        tester.run_test_batch()
    except Exception as e:
        print(f"\n치명적인 오류가 발생했습니다: {e}")
        import traceback

        traceback.print_exc()

