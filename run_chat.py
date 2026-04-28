import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import chromadb

# --- 설정 ---
GENERATIVE_MODEL_ID = "microsoft/Phi-4-mini-instruct"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1"

def run_rag_chat():
    print("RAG 챗봇 시스템을 준비합니다...")

    if not os.path.exists(CHROMA_DB_DIR):
        print(f"❌ DB 폴더 '{CHROMA_DB_DIR}'을(를) 찾을 수 없습니다. 'build_db.py'를 먼저 실행하세요.")
        return
        
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    from chromadb.utils import embedding_functions
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_ID,
        trust_remote_code=True
    )

    # --- [핵심 수정] ---
    # build_db.py에서 생성한 최종 컬렉션 이름으로 변경합니다.
    collection = client.get_collection(
        name="palace_docs_final",
        embedding_function=embedding_function
    )
    # -------------------

    print("✅ DB 연결 완료!")

    print(f"🧠 LLM '{GENERATIVE_MODEL_ID}' 로드 중...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용할 디바이스: {device}")
        model = AutoModelForCausalLM.from_pretrained(
            GENERATIVE_MODEL_ID,
            torch_dtype="auto",
            trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_MODEL_ID)
        print("✅ LLM 로드 완료!")
    except Exception as e:
        print(f"❌ LLM 로드 중 오류: {e}")
        return

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("\n--- RAG 챗봇을 시작합니다. (종료: 'exit' 또는 'quit') ---")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        print("Bot: ...관련 정보를 찾는 중...")
        
        results = collection.query(
            query_texts=[f"search_query: {user_input}"],
            n_results=1 # 가장 관련 높은 1개의 통합 문서를 가져옴
        )
        context = "\n\n---\n\n".join(results["documents"][0]) if results.get("documents") else ""
        
        system_content = (
            "You are a simple information retrieval machine. Your ONLY job is to find the relevant explanation from the 'Context' below and present it to the user. "
            "Follow these rules strictly: "
            "1. Find the explanation that directly answers the user's 'Question'. "
            "2. Copy that explanation EXACTLY as it is. Do NOT add any of your own words, summaries, greetings, or introductory phrases. "
            "3. If the user's question is in Korean, provide the '한국어 설명' part from the context. "
            "4. If you cannot find a relevant explanation in the context, you must ONLY say '관련 정보를 찾을 수 없습니다.'"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"# Context:\n{context}\n\n# Question:\n{user_input}에 대해 한국어로 설명해줘."}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        print("Bot: ...답변을 생성하는 중...")
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        output = pipe(prompt, **generation_args)
        print(f"Bot: {output[0]['generated_text']}")

if __name__ == "__main__":
    run_rag_chat()