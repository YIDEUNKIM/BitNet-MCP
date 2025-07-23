import torch
import time
import psutil
import tracemalloc
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def format_bytes(bytes):
    """바이트를 읽기 쉬운 형태로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def get_gpu_memory():
    """GPU 메모리 사용량 반환"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated(),
            'reserved': torch.cuda.memory_reserved(),
            'max_allocated': torch.cuda.max_memory_allocated()
        }
    return None

def print_memory_info(stage):
    """메모리 정보 출력"""
    # CPU 메모리
    process = psutil.Process()
    cpu_memory = process.memory_info()
    
    print(f"\n=== {stage} 메모리 사용량 ===")
    print(f"CPU RAM: {format_bytes(cpu_memory.rss)}")
    print(f"CPU Virtual: {format_bytes(cpu_memory.vms)}")
    
    # GPU 메모리
    gpu_memory = get_gpu_memory()
    if gpu_memory:
        print(f"GPU Allocated: {format_bytes(gpu_memory['allocated'])}")
        print(f"GPU Reserved: {format_bytes(gpu_memory['reserved'])}")
        print(f"GPU Max Allocated: {format_bytes(gpu_memory['max_allocated'])}")
    else:
        print("GPU: 사용할 수 없음")

def main():
    # 메모리 추적 시작
    tracemalloc.start()
    
    # 시작 시간
    total_start_time = time.time()
    
    # 초기 메모리 상태
    print_memory_info("초기 상태")
    
    # GPU 메모리 초기화
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    model_id = "microsoft/bitnet-b1.58-2B-4T"
    
    # 1. 토크나이저 로딩
    print("\n🔄 토크나이저 로딩 중...")
    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_time = time.time() - tokenizer_start
    print(f"✅ 토크나이저 로딩 완료: {tokenizer_time:.2f}초")
    
    print_memory_info("토크나이저 로딩 후")
    
    # 2. 모델 로딩
    print("\n🔄 모델 로딩 중...")
    model_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )
    model_time = time.time() - model_start
    print(f"✅ 모델 로딩 완료: {model_time:.2f}초")
    
    print_memory_info("모델 로딩 후")
    
    # 3. 추론 준비
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How are you?"},
    ]
    
    # 입력 처리 시간 측정
    print("\n🔄 입력 처리 중...")
    input_start = time.time()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_time = time.time() - input_start
    print(f"✅ 입력 처리 완료: {input_time:.4f}초")
    
    print_memory_info("입력 처리 후")
    
    # 4. 추론 실행
    print("\n🔄 텍스트 생성 중...")
    inference_start = time.time()
    
    # 여러 번 실행하여 평균 시간 측정
    num_runs = 3
    total_inference_time = 0
    
    for i in range(num_runs):
        run_start = time.time()
        with torch.no_grad():  # 메모리 효율성 향상
            chat_outputs = model.generate(**chat_input, max_new_tokens=50)
        run_time = time.time() - run_start
        total_inference_time += run_time
        print(f"  Run {i+1}: {run_time:.4f}초")
    
    avg_inference_time = total_inference_time / num_runs
    print(f"✅ 평균 추론 시간: {avg_inference_time:.4f}초")
    
    print_memory_info("추론 완료 후")
    
    # 5. 결과 디코딩
    decode_start = time.time()
    response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True)
    decode_time = time.time() - decode_start
    print(f"✅ 디코딩 완료: {decode_time:.4f}초")
    
    # 전체 시간
    total_time = time.time() - total_start_time
    
    # Python 메모리 추적 정보
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # 결과 출력
    print("\n" + "="*50)
    print("🎯 성능 측정 결과")
    print("="*50)
    print(f"토크나이저 로딩:     {tokenizer_time:.2f}초")
    print(f"모델 로딩:          {model_time:.2f}초")
    print(f"입력 처리:          {input_time:.4f}초")
    print(f"평균 추론 시간:      {avg_inference_time:.4f}초")
    print(f"디코딩:            {decode_time:.4f}초")
    print(f"전체 실행 시간:      {total_time:.2f}초")
    
    print(f"\nPython 메모리 사용량:")
    print(f"현재: {format_bytes(current)}")
    print(f"피크: {format_bytes(peak)}")
    
    # 최종 메모리 상태
    print_memory_info("최종 상태")
    
    print(f"\n💬 AI 응답: {response}")
    
    # 모델 사이즈 정보
    param_count = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    print(f"\n📊 모델 정보:")
    print(f"파라미터 수: {param_count:,}")
    print(f"모델 크기: {format_bytes(param_size)}")
    
    # 처리량 계산
    input_tokens = chat_input['input_ids'].shape[1]
    output_tokens = chat_outputs.shape[1] - input_tokens
    tokens_per_second = output_tokens / avg_inference_time
    
    print(f"\n⚡ 처리량:")
    print(f"입력 토큰: {input_tokens}")
    print(f"출력 토큰: {output_tokens}")
    print(f"생성 속도: {tokens_per_second:.2f} tokens/sec")

if __name__ == "__main__":
    main()