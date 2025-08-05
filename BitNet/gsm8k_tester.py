import subprocess
import re
import time
import os
import csv
from datasets import load_dataset
from multiprocessing import Pool, Manager
from functools import partial

# Load GSM8K test set
gsm8k = load_dataset("gsm8k", "main")
test_set = gsm8k["test"]
print(f"Loaded GSM8K test set with {len(test_set)} samples.")

# Function to extract final number from mcp_client_math.py output (loose EM: extract from "#### [number]")
def extract_final_answer(output):
    match = re.search(r"####\s*(\d+\.?\d*)", output, re.IGNORECASE | re.DOTALL)
    if match:
        return float(match.group(1))
    # If no match, log a snippet for debugging
    print("  - Warning: No '#### [number]' pattern found in output. Snippet:", output[-200:] if len(output) > 200 else output)
    return None

# Function to extract ground truth from GSM8K answer (loose EM: extract from "#### [number]")
def extract_ground_truth(answer):
    match = re.search(r"####\s*(\d+\.?\d*)", answer, re.IGNORECASE | re.DOTALL)
    if match:
        return float(match.group(1))
    return None

# Function to process a single sample (used in parallel)
def process_sample(sample_data, shared_results, shared_errors):
    i, question, ground_truth = sample_data
    print(f"\nProcessing sample {i + 1}/{max_samples}: {question[:100]}...")

    start_time = time.time()
    try:
        result = subprocess.run(
            ["python", "mcp_client_math_optimal.py", "-q", question],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        predicted = extract_final_answer(output)

        # Save output to a debug file for failed extractions
        if predicted is None:
            debug_file = os.path.join(results_dir, f"debug_output_{i+1}.txt")
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"  - Debug output saved to: {debug_file}")

        elapsed = time.time() - start_time
        print(f"  - Time taken: {elapsed:.2f}s")
        print(f"  - Predicted: {predicted}, Ground Truth: {ground_truth}")

        is_correct = predicted is not None and predicted == ground_truth
        shared_results.append([i + 1, question, predicted, ground_truth, is_correct])
        return is_correct
    except subprocess.TimeoutExpired:
        print("  - Timeout: Skipped (possibly due to model loading delay).")
        shared_errors.append(f"Timeout on sample {i} (model load delay?)")
        shared_results.append([i + 1, question, "TIMEOUT", ground_truth, False])
        return False
    except Exception as e:
        print(f"  - Error: {e}")
        shared_errors.append(f"Error on sample {i}: {e}")
        shared_results.append([i + 1, question, "ERROR", ground_truth, False])
        return False

# Test parameters
max_samples = 100  # 전체 테스트 시 len(test_set)으로 변경
num_processes = 2  # Reduced to avoid overload during parallel model loading
batch_size = 10  # Process in batches to manage load and save incrementally
correct = 0
total = 0

# Create results directory if not exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, "gsm8k_results.csv")

# CSV header (write once at the beginning)
with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sample_id", "question", "predicted", "ground_truth", "correct"])

# Prepare data for processing
data_to_process = []
for i, sample in enumerate(test_set):
    if total >= max_samples:
        break
    question = sample["question"]
    ground_truth = extract_ground_truth(sample["answer"])
    data_to_process.append((i, question, ground_truth))
    total += 1

# Process in batches to manage load and save incrementally
shared_errors = Manager().list()  # Global shared errors
batch_results = []  # To collect all results

for batch_start in range(0, len(data_to_process), batch_size):
    batch_data = data_to_process[batch_start:batch_start + batch_size]
    print(f"\nProcessing batch {batch_start // batch_size + 1} ({len(batch_data)} samples)...")

    # Use Manager per batch for shared lists
    manager = Manager()
    shared_results = manager.list()

    # Create a pool of workers for this batch
    pool = Pool(processes=num_processes)
    process_func = partial(process_sample, shared_results=shared_results, shared_errors=shared_errors)
    results = pool.map(process_func, batch_data)
    pool.close()
    pool.join()

    # Accumulate batch results
    batch_results.extend(shared_results)

    # Write batch results to CSV immediately (append mode)
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(shared_results)

    # Update correct count
    correct += sum(1 for r in results if r)

# Calculate accuracy
accuracy = (correct / total * 100) if total > 0 else 0
print(f"\nTest Results:")
print(f" - Total tested: {total}")
print(f" - Correct: {correct}")
print(f" - Accuracy: {accuracy:.2f}%")
print(f" - Errors: {len(shared_errors)} (details: {shared_errors})")
print(f" - Results saved to: {results_file}")
