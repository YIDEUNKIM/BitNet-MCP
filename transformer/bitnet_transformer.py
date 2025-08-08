import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import os
import gc

# --- 리소스 측정 및 출력 함수 ---
def measure_and_print_inference_usage(model, chat_input, max_new_tokens=50):
    """
    model.generate() 호출 전후의 메모리 변화를 측정하여
    순수 추론에 소요된 RAM과 VRAM 사용량을 출력합니다.
    """
    # 1. 추론 시작 전 메모리 상태 측정
    # -----------------------------------
    gc.collect() # 가비지 컬렉션으로 정확도 향상
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # GPU 캐시 비우기
        torch.cuda.reset_peak_memory_stats() # 최대 메모리 통계 초기화
        
    process = psutil.Process(os.getpid())
    ram_before_gb = process.memory_info().rss / (1024 ** 3)
    
    vram_allocated_before_gb = 0
    if torch.cuda.is_available():
        vram_allocated_before_gb = torch.cuda.memory_allocated() / (1024 ** 3)

    print("--- 추론 시작 전 ---")
    print(f"RAM 사용량: {ram_before_gb:.4f} GB")
    if torch.cuda.is_available():
        print(f"GPU VRAM 할당량: {vram_allocated_before_gb:.4f} GB")
    print("-" * 20)

    # 2. 추론 실행
    # -----------------------------------
    chat_outputs = model.generate(**chat_input, max_new_tokens=max_new_tokens)
    
    # 3. 추론 완료 후 메모리 상태 측정
    # -----------------------------------
    ram_after_gb = process.memory_info().rss / (1024 ** 3)
    
    vram_allocated_after_gb = 0
    vram_peak_gb = 0
    if torch.cuda.is_available():
        vram_allocated_after_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("--- 추론 완료 후 ---")
    print(f"RAM 사용량: {ram_after_gb:.4f} GB")
    if torch.cuda.is_available():
        print(f"GPU VRAM 할당량 (현재): {vram_allocated_after_gb:.4f} GB")
        print(f"GPU VRAM 할당량 (최대): {vram_peak_gb:.4f} GB")
    print("-" * 20)
    
    # 4. 순수 추론 리소스 사용량 계산 및 출력
    # -----------------------------------
    ram_consumed = ram_after_gb - ram_before_gb
    vram_consumed = vram_peak_gb # reset_peak_memory_stats 이후의 최댓값이 순수 추론 사용량
    
    print("✅ 순수 추론 단계 리소스 소모량")
    print(f"   - RAM 증가량: {ram_consumed:.4f} GB")
    if torch.cuda.is_available():
        print(f"   - VRAM 증가량 (Peak): {vram_consumed:.4f} GB")
    print("-" * 20)
    
    return chat_outputs


# --- 메인 코드 ---
# 모델 및 토크나이저 로드 (이 단계는 측정에서 제외)
model_id = "microsoft/bitnet-b1.58-2B-4T"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

# 프롬프트 준비
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)


# 추론 및 리소스 측정 함수 호출
# ----------------------------
print("\n🚀 추론 단계의 리소스 사용량 측정을 시작합니다...")
chat_outputs = measure_and_print_inference_usage(model, chat_input, max_new_tokens=512)


# 결과 디코딩 및 출력
response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True)
print("\nAssistant Response:", response)
