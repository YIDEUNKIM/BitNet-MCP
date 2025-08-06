# ==============================================================================
# GSM8K Tester - Optimized with llama-server (Direct Expression Mode, aiohttp)
#
# Description:
# This script is a testing harness for GSM8K that faithfully mirrors the logic
# and structure of mcp_client_math_optimal2.py, including the MathAgent class
# and its asynchronous methods. It requires TWO running servers:
# 1. The llama-server (started by this script).
# 2. The MCP tool server (must be started manually by the user).
#
# Modifications for alignment with benchmark_gsm8k_pure.py:
# - Temperature set to 0.1 for deterministic responses.
# - n_predict increased to 256 for longer outputs.
# - Flexible regex for answer extraction (handles commas, decimals).
#
# Dependencies:
# pip install datasets fastmcp aiohttp
# Ensure 'llama-server' is built and accessible in 'build/bin/'.
# ==============================================================================

import subprocess
import re
import time
import os
import csv
import json
import platform
import socket
import asyncio
import aiohttp
from urllib import error
from datasets import load_dataset, load_from_disk
from dataclasses import dataclass
from typing import Any, Dict, Optional
from fastmcp import Client  # Import Client for real MCP calls

# --- 1. Prompts (Copied verbatim from mcp_client_math_optimal2.py) ---
QUERY_TO_EXPRESSION_PROMPT = '''[System]
You are an expert at solving grade school math problems.
Below are some examples. Please show your reasoning step-by-step. At the end, provide the final numerical answer in the format #### <answer>.
**CRITICAL: Output ONLY the final Python expression. DO NOT include any analysis, variables, explanations, or additional text. AVOID REPETITION in the expression – use efficient math like sums or loops if needed, but keep it concise.**
---
[Example 1]
User Problem: "A runner finished a 5km race. The next day, they ran 50% further. What was the total distance they ran over both days?"
Final Python Expression: 5 + (5 * 1.5)
---
[Example 2]
User Problem: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
Final Python Expression: (12 / 60) * 50
---
[Example 3]
User Problem: "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
Final Python Expression: 100 - 50 - 30 - 15
---
[Example 4]
User Problem: "Anna has a garden with flowers of two colors. She has 20 red flowers, and there are 50% more blue flowers than red flowers. How many flowers does Anna have in total?"
Final Python Expression: 20 + (20 * 1.5)
---
[Example 5]
User Problem: "Tom has a collection of toys. He has 8 cars, 12 trucks, and there are only 25% as many planes as there are cars and trucks combined. How many toys does Tom have?"
Final Python Expression: 8 + 12 + ((8 + 12) * 0.25)
---
[Example 6]
User Problem: "Sarah has a garden with flowers. She has 10 yellow flowers, and there are 80% more purple flowers than yellow flowers. There are 25% as many green flowers as there are yellow and purple flowers combined. How many flowers does Sarah have in total?"
Final Python Expression: 10 + (10 * 1.8) + ((10 + (10 * 1.8)) * 0.25)
---
[User Problem]
{query}
Final Python Expression:'''

SYNTHESIZE_ANSWER_PROMPT = '''[System]
You are a helpful assistant who provides the final answer in a specific format for loose EM evaluation.
Output ONLY the final answer in the format: #### [result]
Do not include any reasoning, explanations, or other text. Follow the examples exactly and always use this format.
Here are examples:

[Question1]
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

[Answer]
In April, Natalia sold 48 clips.
In May, she sold half as many, which is 48 / 2 = 24 clips.
In total, she sold 48 + 24 = 72 clips.
#### 72

[Question2]
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

[Answer]
Weng earns $12 per hour, which is 60 minutes.
So, her rate per minute is $12 / 60 = $0.20 per minute.
For 50 minutes of babysitting, she earned 50 * $0.20 = $10.
#### 10

[Question3]
At a flea market, Hillary sells handmade crafts for 12 dollars per craft. Today, Hillary sells 3 crafts and is given an extra 7 dollars from an appreciative customer. Later on, Hillary deposits 18 dollars from today's profits into her bank account. How many dollars is Hillary left with after making the deposit?

[Answer]
Hillary sells 3 crafts for 12 dollars each, for a total of 3 crafts * $12/craft = $<<3*12=36>>36
She receives an extra 7 dollars from a customer, increasing the total to $36 + $7 = $<<36+7=43>>43
She then deposits 18 dollars in the bank, leaving her with $43 - $18 = $25
#### 25

[Question4]
James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?

[Answer]
He writes each friend 3*2=<<3*2=6>>6 pages a week
So he writes 6*2=<<6*2=12>>12 pages every week
That means he writes 12*52=<<12*52=624>>624 pages a year
#### 624

[Question]
{query}

[Answer]
'''

# --- 2. Configuration (Aligned with benchmark_gsm8k_pure.py) ---
@dataclass
class AppConfig:
    model_path: str = os.path.join(os.path.dirname(__file__), "models", "BitNet-b1.58-2B-4T", "ggml-model-i2_s.gguf")
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    n_predict: int = 256  # Increased to match pure benchmark
    threads: int = 8
    ctx_size: int = 2048
    temperature: float = 0.1  # Lowered to match pure benchmark for determinism
    n_gpu_layers: int = 0
    mcp_server_url: str = "http://localhost:8001/sse"

# --- 3. SLM & Tool Interaction ---
def get_server_path() -> str:
    """Finds the llama-server executable."""
    build_dir = "build"
    exe_name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    main_path = os.path.join(build_dir, "bin", exe_name)
    if not os.path.exists(main_path):
        main_path = os.path.join("..", build_dir, "bin", exe_name)
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"llama-server not found at '{os.path.abspath(main_path)}'. Please build the project first.")
    return os.path.abspath(main_path)

async def call_slm_async(prompt: str, config: AppConfig) -> str:
    """Sends an async query to the running llama-server using aiohttp."""
    url = f"http://{config.server_host}:{config.server_port}/completion"
    data = {
        "prompt": prompt,
        "n_predict": config.n_predict,
        "temperature": config.temperature,
        "stop": ["---", "[User Problem]", "Final Python Expression:"]
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, timeout=120) as response:
                result = await response.json()
                return result.get("content", "").strip()
    except Exception as e:
        print(f" - SLM call failed: {e}")
        return "SLM_CALL_FAILED"

# --- 4. Core Agent Logic (Focused on pure expression -> MCP -> result) ---
class MathAgent:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = Client(self.config.mcp_server_url)  # Client 재사용

    async def _query_to_expression(self, query: str) -> str:
        prompt = QUERY_TO_EXPRESSION_PROMPT.format(query=query)
        full_response = await call_slm_async(prompt, self.config)
        if full_response == "SLM_CALL_FAILED" or not full_response:
            return ""
        return full_response.strip()

    async def _execute_calculation(self, expression: str) -> Optional[Dict[str, Any]]:
        """Makes a real network call to the MCP tool server to execute the expression."""
        print(f" - Calling MCP tool server to execute: {expression}")
        try:
            async with self.client:  # 연결 재사용
                result = await asyncio.wait_for(
                    self.client.call_tool("calculate", {"expression": expression}),
                    timeout=30.0  # 타임아웃 증가
                )
                output = getattr(result, 'data', result)
                if isinstance(output, str):
                    try: output = json.loads(output)
                    except (ValueError, SyntaxError): pass
                return output
        except asyncio.TimeoutError:
            return {"error": "MCP server call timed out."}
        except Exception as e:
            return {"error": f"MCP tool call failed: {e}"}

    async def _synthesize_answer(self, query: str, expression: str, result: Any) -> str:
        # 프롬프트 제거하고 직접 형식화
        if result is not None:
            # 소수점/쉼표 처리 (기존 regex 로직 반영)
            cleaned = str(result).replace(",", "")  # 쉼표 제거
            return f"#### {cleaned}"
        else:
            return "#### None"  # 실패 시 대체

    async def solve(self, query: str) -> Optional[float]:
        expression = await self._query_to_expression(query)
        if not expression:
            return None
        calc_result_dict = await self._execute_calculation(expression)
        if calc_result_dict.get("error"):
            print(f" - Calculation failed: {calc_result_dict.get('error')}")
            return None
        result_value = calc_result_dict.get('result')
        final_answer_str = await self._synthesize_answer(query, expression, result_value)
        predicted = extract_final_answer(final_answer_str)
        print(f" - Expression: {expression} | Predicted: {predicted}")
        return predicted

# --- 5. Test Execution Logic (Aligned with pure benchmark) ---
def wait_for_server_ready(config: AppConfig, timeout: int = 60):
    """Waits for the llama-server to become available."""
    start_time = time.time()
    url = f"http://{config.server_host}:{config.server_port}/health"
    print(f"\nWaiting for llama-server at {url}...", end="")
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((config.server_host, config.server_port), timeout=1):
                print(" Server is ready!")
                return True
        except (OSError, ConnectionRefusedError):
            print(".", end="", flush=True)
            time.sleep(1)
    raise RuntimeError(f"llama-server did not become ready within {timeout} seconds.")

def extract_final_answer(output: str) -> Optional[float]:
    """Extracts the final numerical answer from a string (flexible like pure benchmark)."""
    if output is None: return None
    match = re.search(r"####[ ]*(-?[0-9,]+\.?[0-9]*)", str(output), re.IGNORECASE | re.DOTALL)
    if match:
        cleaned = match.group(1).replace(",", "")
        return float(cleaned)
    return None

def extract_ground_truth(answer: str) -> Optional[float]:
    """Extracts the ground truth number from the GSM8K answer string."""
    match = re.search(r"####[ ]*(-?[0-9]+\.?[0-9]*)", answer, re.IGNORECASE | re.DOTALL)
    return float(match.group(1)) if match else None

async def process_sample_async(sample_data, config, shared_results, shared_errors):
    """Processes a single sample from the GSM8K dataset asynchronously."""
    i, question, ground_truth = sample_data
    print(f"\nProcessing sample {i + 1}: {question[:100]}...")
    agent = MathAgent(config)
    predicted = await agent.solve(question)
    is_correct = False
    if predicted is not None:
        is_correct = abs(predicted - ground_truth) < 1e-3
    else:
        shared_errors.append(f"Failed to solve sample {i}")
    shared_results.append([i + 1, question, predicted, ground_truth, is_correct])

async def process_batch_async(batch_data, config, shared_results, shared_errors):
    """Processes a batch of samples asynchronously with concurrency limit."""
    sem = asyncio.Semaphore(5)  # 동시에 최대 5개 요청 제한 (시스템에 맞게 조정)
    async def limited_process(sample):
        async with sem:
            await process_sample_async(sample, config, shared_results, shared_errors)
    tasks = [limited_process(data) for data in batch_data]
    await asyncio.gather(*tasks)

# --- Main Execution (Async Wrapper) ---
async def main():
    config = AppConfig()
    max_samples = 1319
    batch_size = 10  # Adjust based on system resources
    print("Loading GSM8K test set from disk (for speed)...")
    cache_dir = "BitNet/gsm8k_cache"
    if not os.path.exists(cache_dir):
        print("Cache not found. Loading from HuggingFace and saving...")
        gsm8k = load_dataset("gsm8k", "main")
        gsm8k.save_to_disk(cache_dir)
    gsm8k = load_from_disk(cache_dir)
    test_set = gsm8k["test"]
    print(f"Loaded {len(test_set)} samples.")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "gsm8k_results_mcp_mode.csv")
    with open(results_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "predicted", "ground_truth", "correct"])
    data_to_process = []
    for i, sample in enumerate(test_set):
        if len(data_to_process) >= max_samples: break
        gt = extract_ground_truth(sample["answer"])
        if gt is not None:
            data_to_process.append((i, sample["question"], gt))
    print(f"\nSuccessfully prepared {len(data_to_process)} samples for processing.")
    server_process = None
    try:
        total = len(data_to_process)
        if total == 0:
            raise RuntimeError("No samples were prepared. Check extract_ground_truth function.")
        server_path = get_server_path()
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model file not found at: {config.model_path}")
        command = [
            server_path, "-m", config.model_path, "-c", str(config.ctx_size),
            "-t", str(config.threads), "--host", config.server_host,
            "--port", str(config.server_port), "-cb"
        ]
        if config.n_gpu_layers != 0:
            command.extend(["-ngl", str(config.n_gpu_layers)])
        print("\nStarting llama-server...")
        server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Server started with PID: {server_process.pid}", end="")
        wait_for_server_ready(config)
        print("\nIMPORTANT: Ensure the MCP tool server is running on port 8001 for this test to work.")
        correct = 0
        shared_errors = []
        shared_results = []
        for i in range(0, total, batch_size):
            batch_data = data_to_process[i:i + batch_size]
            print(f"\n--- Processing Batch {i//batch_size + 1}/{(total + batch_size - 1) // batch_size} ---")
            await process_batch_async(batch_data, config, shared_results, shared_errors)
            # 최신 배치 결과만 추출 (shared_results는 append 순서 유지)
            batch_results_list = shared_results[-len(batch_data):]
            correct += sum(1 for r in batch_results_list if r[4])
            with open(results_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(batch_results_list)
        accuracy = (correct / total * 100) if total > 0 else 0
        print("\n" + "="*50)
        print("Test Finished")
        print("="*50)
        print(f" - Total tested: {total}")
        print(f" - Correct: {correct}")
        print(f" - Accuracy: {accuracy:.2f}%")
        if shared_errors:
            print(f" - Errors ({len(shared_errors)}):")
            for error in shared_errors[:10]:
                print(f" - {error}")
        print(f"\nResults saved to: {results_file}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if server_process:
            print("\nTerminating llama-server...")
            server_process.terminate()
            server_process.wait()
            print("Server terminated.")

if __name__ == "__main__":
    asyncio.run(main())
