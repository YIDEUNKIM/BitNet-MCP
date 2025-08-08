# ==============================================================================
# GSM8K Benchmark for Pure BitNet Model (Optimized Version)
#
# Description:
# This script benchmarks a GGUF model on the GSM8K dataset using a direct,
# few-shot prompting approach. It uses asyncio for parallel HTTP requests to
# improve speed on I/O-bound tasks.
#
# Dependencies:
# pip install datasets aiohttp
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
from functools import partial
from dataclasses import dataclass
from typing import Optional

# --- 1. Prompts (Chain-of-Thought Few-Shot) ---
DIRECT_SOLVE_PROMPT = '''You are an expert at solving grade school math problems.
Below are some examples. Please show your reasoning step-by-step. At the end, provide the final numerical answer in the format #### <answer>.

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

# --- 2. Configuration ---
@dataclass
class AppConfig:
    model_path: str = os.path.join(os.path.dirname(__file__), "models", "BitNet-b1.58-2B-4T", "ggml-model-i2_s.gguf")
    server_host: str = "127.0.0.1"
    server_port: int = 8081
    n_predict: int = 256
    threads: int = 8
    ctx_size: int = 2048
    temperature: float = 0.1  # Low temperature for deterministic answers
    n_gpu_layers: int = 0

# --- 3. Helper Functions ---
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

async def call_slm_async(prompt: str, config: AppConfig, session: aiohttp.ClientSession) -> str:
    """Sends an async query to the running llama-server."""
    url = f"http://{config.server_host}:{config.server_port}/completion"
    data = {
        "prompt": prompt,
        "n_predict": config.n_predict,
        "temperature": config.temperature,
    }
    try:
        async with session.post(url, json=data, timeout=120) as response:
            result = await response.json()
            return result.get("content", "").strip()
    except Exception as e:
        print(f"  - SLM call failed: {e}")
        return ""

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
    """Extracts the final numerical answer from a string."""
    if output is None: return None
    match = re.search(r"####[ ]*(-?[0-9,]+\.?[0-9]*)", output)
    if match:
        number_str = match.group(1).replace(",", "")
        try:
            return float(number_str)
        except ValueError:
            return None
    return None

def extract_ground_truth(answer: str) -> Optional[float]:
    """Extracts the ground truth number from the GSM8K answer string."""
    match = re.search(r"####[ ]*(-?[0-9,]+\.?[0-9]*)", answer)
    if match:
        number_str = match.group(1).replace(",", "")
        try:
            return float(number_str)
        except ValueError:
            return None
    return None

# --- 4. Core Processing Logic ---
async def process_sample(sample_data, config, session, shared_results, shared_errors):
    """Processes a single sample from the GSM8K dataset asynchronously."""
    i, question, ground_truth = sample_data
    print(f"\nProcessing sample {i + 1}: {question[:100]}...")

    prompt = DIRECT_SOLVE_PROMPT.format(query=question)
    model_output = await call_slm_async(prompt, config, session)
    predicted = extract_final_answer(model_output)

    is_correct = False
    if predicted is not None:
        is_correct = abs(predicted - ground_truth) < 1e-3
    else:
        shared_errors.append(f"Failed to extract answer for sample {i}. Raw output: {model_output}")

    print(f"  - Predicted: {predicted} | Ground Truth: {ground_truth} | Correct: {is_correct}")
    shared_results.append([i + 1, question, model_output, predicted, ground_truth, is_correct])

# --- 5. Main Execution ---
if __name__ == "__main__":
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
    results_file = os.path.join(results_dir, "benchmark_gsm8k_pure_results.csv")
    with open(results_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "model_output", "predicted", "ground_truth", "correct"])

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

        correct = 0
        shared_errors = []
        shared_results = []

        async def process_batch(batch_data, config):
            async with aiohttp.ClientSession() as session:
                tasks = [process_sample(data, config, session, shared_results, shared_errors) for data in batch_data]
                await asyncio.gather(*tasks)

        for i in range(0, total, batch_size):
            batch_data = data_to_process[i:i + batch_size]
            print(f"\n--- Processing Batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} ---")
            asyncio.run(process_batch(batch_data, config))

            # 수정: 최신 배치 결과를 직접 슬라이싱으로 가져옴 (더 안전하고 효율적)
            batch_results_list = shared_results[-len(batch_data):]  # 마지막 len(batch_data) 개 항목

            # correct 카운트
            correct += sum(1 for r in batch_results_list if r[5])  # is_correct가 index 5라고 가정

            # CSV에 쓰기 (기존 코드 유지)
            with open(results_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(batch_results_list)

        accuracy = (correct / total * 100) if total > 0 else 0
        print("\n" + "="*50)
        print("Benchmark Finished")
        print("="*50)
        print(f" - Total tested: {total}")
        print(f" - Correct: {correct}")
        print(f" - Accuracy: {accuracy:.2f}%")
        if shared_errors:
            print(f" - Errors ({len(shared_errors)}):")
            for error in shared_errors[:10]:
                print(f"   - {error}")
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
