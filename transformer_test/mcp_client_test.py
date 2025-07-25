# bitnet_client.py
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastmcp import Client
import psutil
import os

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"

# --- 메모리 측정 로직을 위한 헬퍼 함수 ---
def log_memory_usage(stage: str, process: psutil.Process):
    """지정된 프로세스의 메모리 사용량을 측정하고 출력합니다."""
    mem_info_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\n--- [메모리 측정] {stage}: {mem_info_mb:.2f} MB ---")


def load_model():
    """BitNet 모델과 토크나이저를 로드합니다."""
    try:
        print(f"'{MODEL_ID}' 모델과 토크나이저를 로딩합니다...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("모델 로딩 완료.")
        return model, tokenizer
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return None, None

def generate_response(model, tokenizer, system_prompt: str, user_prompt: str) -> str:
    """BitNet 모델을 사용하여 프롬프트에 대한 응답을 생성합니다."""
    if not model or not tokenizer:
        return "모델이 로드되지 않아 응답을 생성할 수 없습니다."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return response

async def run_mcp_client(model, tokenizer, process: psutil.Process):
    """FastMCP 클라이언트를 실행하며 각 추론 단계별 메모리를 측정합니다."""
    server_url = "http://localhost:8000/sse/"
    print(f"\n--- MCP 클라이언트 시작: {server_url}에 연결 ---")

    async with Client(server_url) as client:
        # 이 블록에 진입할 때: MCP 서버와 연결 설정
        # 블록을 벗어날 때: 연결 자동 해제
        print("MCP 서버에 성공적으로 연결되었습니다.")
        
        system_prompt = "당신은 주어진 '도구 실행 결과'를 바탕으로 '사용자 질문'에 대해 자연스러운 한국어 문장으로 답변하는 요약 전문가입니다. 절대 추가 정보를 만들지 말고, 주어진 사실만을 사용하여 간결하게 한 문장으로 답변하세요."
        
        # 시나리오 1: 날씨 정보 요청
        print("\n[시나리오 1: 날씨 도구 사용]")
        weather_result = await client.call_tool("get_weather", {"city": "파리"})
        tool_output_weather = weather_result.content[0].text
        print(f"서버로부터 받은 날씨 정보: {tool_output_weather}")

        user_prompt_weather = f"[사용자 질문]\n파리의 날씨는 어때?\n\n[도구 실행 결과]\n{tool_output_weather}\n\n[최종 답변]\n"
        
        # --- 수정된 부분: 첫 번째 추론 전/후 메모리 측정 ---
        log_memory_usage("첫 번째 추론 시작 전", process)
        final_answer_weather = generate_response(model, tokenizer, system_prompt, user_prompt_weather)
        log_memory_usage("첫 번째 추론 완료 후", process)
        
        print(f"\nBitNet 기반 최종 답변:\n{final_answer_weather.strip()}")

        # 시나리오 2: 계산 요청
        print("\n\n[시나리오 2: 계산기 도구 사용]")
        calc_result = await client.call_tool("calculate", {"a": 2048, "b": 16, "operation": "divide"})
        print(calc_result)
        tool_output_calc = calc_result.content[0].text
        print(f"서버로부터 받은 계산 결과: {tool_output_calc}")
        
        user_prompt_calc = f"[사용자 질문]\n2048을 16으로 나누면?\n\n[도구 실행 결과]\n{tool_output_calc}\n\n[최종 답변]\n"
        
        # --- 수정된 부분: 두 번째 추론 전/후 메모리 측정 ---
        log_memory_usage("두 번째 추론 시작 전", process)
        final_answer_calc = generate_response(model, tokenizer, system_prompt, user_prompt_calc)
        log_memory_usage("두 번째 추론 완료 후", process)
        
        print(f"\nBitNet 기반 최종 답변:\n{final_answer_calc.strip()}")


async def main(process: psutil.Process):
    """메인 실행 로직을 포함하며, 프로세스 객체를 전달합니다."""
    log_memory_usage("스크립트 시작", process)
    bitnet_model, bitnet_tokenizer = load_model()
    
    if bitnet_model and bitnet_tokenizer:
        log_memory_usage("모델 로딩 완료 후", process)
        await run_mcp_client(bitnet_model, bitnet_tokenizer, process)
    else:
        print("\n모델이 준비되지 않아 클라이언트를 실행할 수 없습니다.")


if __name__ == "__main__":
    main_process = psutil.Process(os.getpid())
    try:
        asyncio.run(main(main_process))
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {e}")
    finally:
        mem_info_mb = main_process.memory_info().rss / (1024 * 1024)
        print("\n" + "="*50)
        print("               최종 누적 메모리 사용량")
        print("="*50)
        print(f"스크립트 종료 시점의 메모리 사용량: {mem_info_mb:.2f} MB")
        print("="*50)
