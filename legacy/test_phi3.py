import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def run_chat_test():
    # --- 1. 모델 및 토크나이저 로드 ---

    model_id = "microsoft/Phi-4-mini-instruct"
    # model_id = "microsoft/Phi-3.5-mini-instruct"
    print(f"'{model_id}' 모델 로드를 시작합니다...")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용할 디바이스: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print("✅ 모델 로드 완료!")

    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        return

    # --- 2. 텍스트 생성 파이프라인 생성 ---
    # 파이프라인을 한 번만 생성하여 더 효율적으로 만듭니다.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    print("\n채팅을 시작합니다. 종료하려면 'exit' 또는 'quit'을 입력하세요.")
    
    # --- 3. 시스템 프롬프트 강화 ---
    # 모델의 역할을 더 구체적이고 명확하게 지시합니다.
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. You must answer all questions in Korean. Do not use any other languages."},
    ]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("채팅을 종료합니다.")
            break

        messages.append({"role": "user", "content": user_input})
        
        # apply_chat_template을 사용해 모델이 가장 잘 이해하는 형식으로 변환합니다.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False, # ✅ 이전 대화 내용을 제외하고 순수한 답변만 받도록 설정
            "temperature": 0.7,
            "do_sample": True,
        }

        print("Bot: ...생각 중...")
        # 파이프라인을 호출하여 텍스트 생성
        output = pipe(prompt, **generation_args)
        
        bot_response = output[0]['generated_text']
        print(f"Bot: {bot_response}")
        
        messages.append({"role": "assistant", "content": bot_response})


if __name__ == "__main__":
    run_chat_test()