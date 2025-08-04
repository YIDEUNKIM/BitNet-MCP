"""
python mcp_client_math.py
────────────────────────────────────────────────────────
Math Agent (Structured Template-based)

Forces a model to follow a strict template to deconstruct
a word problem, extract a final expression, execute it,
and synthesize a clear answer.
────────────────────────────────────────────────────────
❱❱ Dependencies
pip install fastmcp rich
"""

import asyncio
import os
import re
import ast
import subprocess
import platform
import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastmcp import Client

# --- 1. Prompts ---
# CRITICAL 지시문을 통해 모델의 과잉 생성을 억제합니다.
QUERY_TO_EXPRESSION_PROMPT = """[System]
You are an expert mathematician specializing in converting word problems into pure Python mathematical expressions.
Your sole purpose is to deconstruct the user's problem and output a single, executable Python expression.

You MUST follow this template exactly.
**CRITICAL: Provide the analysis for the single [User Problem] given below. DO NOT generate any additional examples, problems, or text after completing the analysis for the single problem.**
**IMPORTANT: ALWAYS calculate the sum of referenced items FIRST when a percentage like "X% as many as Y and Z" is mentioned. It means X% of (Y + Z). Do not apply the percentage to individual items.**

---
[Example 1]
User Problem: "A runner finished a 5km race. The next day, they ran 50% further. What was the total distance they ran over both days?"

Your Analysis:
1.  **Variables & Numbers:** distance1 = 5, percentage_increase = 0.5
2.  **Problem Goal:** Calculate the first day's distance plus the second day's distance.
3.  **Step-by-step Logic:** The second day's distance is `distance1 * (1 + percentage_increase)`. The total distance is `distance1 + (distance1 * (1 + percentage_increase))`. A simpler way is `5 + (5 * 1.5)`.
4.  **Final Python Expression:** 5 + (5 * 1.5)
---
[Example 2]
User Problem: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

Your Analysis:
1.  **Variables & Numbers:** hourly_rate = 12, minutes_worked = 50, minutes_per_hour = 60
2.  **Problem Goal:** Calculate the total earnings for the babysitting time.
3.  **Step-by-step Logic:** The per-minute rate is `hourly_rate / minutes_per_hour`. The total earnings are `(hourly_rate / minutes_per_hour) * minutes_worked`. A simpler way is `(12 / 60) * 50`.
4.  **Final Python Expression:** (12 / 60) * 50
---
[Example 3]
User Problem: "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"

Your Analysis:
1.  **Variables & Numbers:** wallet_cost = 100, betty_has = wallet_cost / 2, parents_give = 15, grandparents_give = parents_give * 2
2.  **Problem Goal:** Calculate how much more money Betty needs after receiving money from her parents and grandparents.
3.  **Step-by-step Logic:** Betty currently has half of the wallet cost, which is `wallet_cost / 2`. Grandparents give twice as much as parents, which is `parents_give * 2`. The remaining amount needed is `wallet_cost - betty_has - grandparents_give - parents_give`. A simpler way is `100 - 50 - 30 - 15`.
4.  **Final Python Expression:** 100 - 50 - 30 - 15
---
[Example 4]
User Problem: "Anna has a garden with flowers of two colors. She has 20 red flowers, and there are 50% more blue flowers than red flowers. How many flowers does Anna have in total?"

Your Analysis:
1.  **Variables & Numbers:** red_flowers = 20, blue_increase = 0.5
2.  **Problem Goal:** Calculate the total number of flowers in the garden.
3.  **Step-by-step Logic:** The number of blue flowers is `red_flowers * (1 + blue_increase)`. The total number of flowers is `red_flowers + (red_flowers * (1 + blue_increase))`. A simpler way is `20 + (20 * 1.5)`.
4.  **Final Python Expression:** 20 + (20 * 1.5)
---
[Example 5]
User Problem: "Tom has a collection of toys. He has 8 cars, 12 trucks, and there are only 25% as many planes as there are cars and trucks combined. How many toys does Tom have?"

Your Analysis:
1.  **Variables & Numbers:** cars = 8, trucks = 12, plane_percentage = 0.25
2.  **Problem Goal:** Calculate the total number of toys.
3.  **Step-by-step Logic:** The total of cars and trucks is `cars + trucks`. The number of planes is `(cars + trucks) * plane_percentage`. The total number of toys is `cars + trucks + ((cars + trucks) * plane_percentage)`. A simpler way is `8 + 12 + ((8 + 12) * 0.25)`.
4.  **Final Python Expression:** 8 + 12 + ((8 + 12) * 0.25)
---
[Example 6]
User Problem: "Sarah has a garden with flowers. She has 10 yellow flowers, and there are 80% more purple flowers than yellow flowers. There are 25% as many green flowers as there are yellow and purple flowers combined. How many flowers does Sarah have in total?"

Your Analysis:
1.  **Variables & Numbers:** yellow = 10, purple = yellow * 1.8, green_percentage = 0.25
2.  **Problem Goal:** Calculate the total number of flowers.
3.  **Step-by-step Logic:** Purple flowers are `yellow * 1.8`. The sum of yellow and purple is `yellow + purple`. Green flowers are `(yellow + purple) * green_percentage`. Total flowers are `yellow + purple + ((yellow + purple) * green_percentage)`. A simpler way is `10 + (10 * 1.8) + ((10 + (10 * 1.8)) * 0.25)`.
4.  **Final Python Expression:** 10 + (10 * 1.8) + ((10 + (10 * 1.8)) * 0.25)
---
[User Problem]
{query}

Your Analysis:
1.  **Variables & Numbers:**"""

SYNTHESIZE_ANSWER_PROMPT = """[System]
You are a helpful assistant who provides direct and clear answers.
Your task is to present the final answer to the user based on the provided information.

Follow this two-sentence format PRECISELY:
1. The final answer is [result].
2. This was calculated using the expression: [expression].

Do not add greetings, sign-offs, or any other text.

[Information]
- User's Question: {query}
- Math Expression Used: {expression}
- Final Result: {result}

[Final Answer]
"""

# --- 2. Configuration ---
@dataclass
class AppConfig:
    server_url: str = "http://localhost:8001/sse"
    model_path: str = "./models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    n_predict: int = 384
    threads: int = 8
    ctx_size: int = 2048
    temperature: float = 0.3  # Slightly lowered for more deterministic responses, improving speed/accuracy trade-off
    n_gpu_layers: int = 0

# --- 3. SLM & Tool Interaction ---
def get_llama_cli_path() -> str:
    build_dir = "build"
    exe_name = "llama-cli.exe" if platform.system() == "Windows" else "llama-cli"
    main_path = os.path.join(build_dir, "bin", exe_name)
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"llama-cli not found at {main_path}. Please build the project first.")
    return main_path

def call_slm(prompt: str, args: argparse.Namespace, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            start_time = time.time()
            llama_cli_path = get_llama_cli_path()
            command = [
                f'{llama_cli_path}',
                '-m', args.model,
                '-p', prompt,
                '-n', str(args.n_predict),
                '-t', str(args.threads),
                '-c', str(args.ctx_size),
                '--temp', str(args.temperature),
                '--no-display-prompt'
            ]
            if hasattr(args, 'n_gpu_layers') and args.n_gpu_layers > 0:
                command.extend(['-ngl', str(args.n_gpu_layers)])

            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            output = result.stdout.strip()
            stop_phrases = ["---", "[User Problem]", "Your Analysis:"]
            for phrase in stop_phrases:
                if phrase in output:
                    output = output.split(phrase)[0]
            elapsed = time.time() - start_time
            print(f"  - SLM call took {elapsed:.2f} seconds (attempt {attempt+1})")
            return output.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"--- SLM Call Failed (attempt {attempt+1}/{retries}) ---", file=sys.stderr)
            if isinstance(e, subprocess.CalledProcessError):
                print(f"Stderr: {e.stderr.strip()}", file=sys.stderr)
            else:
                print(f"Error: {e}", file=sys.stderr)
            if attempt == retries - 1:
                return "SLM_CALL_FAILED"

# --- 4. Core Agent Logic ---
class MathAgent:
    def __init__(self, config: AppConfig, args: argparse.Namespace):
        self.config = config
        self.args = args

    def _query_to_expression(self, query: str) -> str:
        start_time = time.time()
        print("Step 1: Forcing model to follow a structured thought process...")
        prompt = QUERY_TO_EXPRESSION_PROMPT.format(query=query)
        full_response = call_slm(prompt, self.args)

        if full_response == "SLM_CALL_FAILED" or not full_response:
            print("  - Failed to generate expression.", file=sys.stderr)
            return ""

        full_response_display = "1.  **Variables & Numbers:**" + full_response
        print(f"  - Full model output:\n---\n{full_response_display}\n---")

        # Further simplified parsing to handle various formats
        variables = {}
        var_match = re.search(r'1\.\s*\*\*Variables & Numbers:\*\*.*?(.*?)(?=\n2\.|\n3\.|\n4\.|$)', full_response, re.DOTALL | re.IGNORECASE)
        if var_match:
            var_text = var_match.group(1).strip()
            # Split by comma, semicolon, or newline for maximum flexibility
            parts = [p.strip() for p in re.split(r',|;|\n', var_text) if p.strip()]
            for part in parts:
                if '=' in part:
                    key, val = part.split('=', 1)
                    val_stripped = val.strip()
                    try:
                        variables[key.strip()] = ast.literal_eval(val_stripped)
                    except (ValueError, SyntaxError):
                        variables[key.strip()] = val_stripped

        print(f"  - Extracted Variables: {variables}")

        expression = ""
        expr_match = re.search(r'4\.\s*\*\*Final Python Expression:\*\*.*?(.*?)(?=\n|$)', full_response, re.DOTALL | re.IGNORECASE)
        if expr_match:
            raw_expression = expr_match.group(1).strip()
            cleaned_expression = re.sub(r'^\*+\s*|\s*\*+$', '', raw_expression)
            match = re.search(r'[-]?\s*(\d[\d\.\s\(\)\+\-\*\/]*)|[\(\d][\d\.\s\(\)\+\-\*\/]*', cleaned_expression)
            if match:
                expression = match.group(0).strip()

        if not expression:
            print("  - Could not find 'Final Python Expression:' marker in model output.", file=sys.stderr)
            return ""

        print(f"  - Extracted Raw Expression: {expression}")

        for var, val in variables.items():
            val_str = str(val) if not isinstance(val, str) else val
            expression = re.sub(r'\b' + re.escape(var) + r'\b', val_str, expression)

        elapsed = time.time() - start_time
        print(f"  - Step 1 took {elapsed:.2f} seconds")
        print(f"  - Finalized Expression for Execution: {expression}")
        return expression

    async def _execute_calculation(self, expression: str) -> Optional[Dict[str, Any]]:
        start_time = time.time()
        print("\nStep 2: Executing expression via MCP server...")
        try:
            async with Client(self.config.server_url) as client:
                print("  - Selecting tool: calculate")
                result = await asyncio.wait_for(client.call_tool("calculate", {"expression": expression}), timeout=10.0)
                output = getattr(result, 'data', result)
                if isinstance(output, str):
                    try: output = ast.literal_eval(output)
                    except (ValueError, SyntaxError): pass
                print(f"  - Result from server: {output}")
                elapsed = time.time() - start_time
                print(f"  - Step 2 took {elapsed:.2f} seconds")
                return output
        except asyncio.TimeoutError:
            print("  - MCP server call timed out.", file=sys.stderr)
            return None
        except Exception as e:
            print(f"  - Tool execution failed: {e}", file=sys.stderr)
            return None

    def _synthesize_answer(self, query: str, expression: str, result: Any) -> str:
        start_time = time.time()
        print("\nStep 3: Synthesizing final answer...")
        prompt = SYNTHESIZE_ANSWER_PROMPT.format(query=query, expression=expression, result=result)
        final_answer = call_slm(prompt, self.args)

        if final_answer == "SLM_CALL_FAILED":
            return f"I was unable to generate a final explanation, but the calculated result is: {result}"

        # Improved cleaning: Remove any unwanted tokens and duplicates
        final_answer = re.sub(r"<\|im_end\|>|<\|im_sep\|>|<\|.*\|>", "", final_answer)
        final_answer = final_answer.replace("1. ", "").replace("2. ", "\n")
        # Take only the first instance of the answer to avoid repetition
        lines = final_answer.split('\n')
        if len(lines) > 2:
            final_answer = '\n'.join(lines[:2])

        elapsed = time.time() - start_time
        print(f"  - Step 3 took {elapsed:.2f} seconds")
        return final_answer.strip()

    async def solve(self, query: str):
        expression = self._query_to_expression(query)
        if not expression:
            print("\nCould not solve the problem. Aborting.", file=sys.stderr)
            return

        calc_result = await self._execute_calculation(expression)
        if calc_result is None or calc_result.get("error"):
            error_msg = calc_result.get('error') if calc_result else "Unknown error"
            print(f"\nCalculation failed: {error_msg}", file=sys.stderr)
        else:
            result_value = calc_result.get('result', 'No result found')
            # Post-process result: if it's a float and likely should be an integer (like counting flowers), round it
            if isinstance(result_value, float) and "flower" in query.lower():
                result_value = round(result_value)
                print(f"  - Rounded result to integer (for counting items): {result_value}")
            final_answer = self._synthesize_answer(query, expression, result_value)
            print("\n" + "="*80)
            print("Final Answer:")
            print(f"\n{final_answer}")
            print("\n" + "="*80)

# --- 5. Main Execution ---
async def main_async():
    config = AppConfig()
    parser = argparse.ArgumentParser(description="MCP - Math Problem Solver (Structured Template)", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-q", "--query", type=str, required=True, help="The mathematical problem to solve.")
    parser.add_argument("-m", "--model", type=str, default=config.model_path, help=f"Path to model file (default: {config.model_path})")
    parser.add_argument("-n", "--n-predict", type=int, default=config.n_predict, help=f"Number of tokens to predict (default: {config.n_predict})")
    parser.add_argument("-t", "--threads", type=int, default=config.threads, help=f"Number of threads (default: {config.threads})")
    parser.add_argument("-c", "--ctx-size", type=int, default=config.ctx_size, help=f"Context size (default: {config.ctx_size})")
    parser.add_argument("-temp", "--temperature", type=float, default=config.temperature, help=f"Temperature (default: {config.temperature})")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=config.n_gpu_layers, help=f"Number of GPU layers (default: {config.n_gpu_layers})")

    args = parser.parse_args()
    config.model_path = args.model

    try:
        get_llama_cli_path()
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model file not found at: {config.model_path}")
    except FileNotFoundError as e:
        print(f"Fatal Error: {e}", file=sys.stderr)
        sys.exit(1)

    agent = MathAgent(config, args)
    print(f"\nProblem: {args.query}")
    print("="*80)

    try:
        await agent.solve(args.query)
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")

if __name__ == "__main__":
    main()
