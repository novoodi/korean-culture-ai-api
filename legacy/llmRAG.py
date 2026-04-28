import json
import os
import chromadb
from chromadb.utils import embedding_functions

# --- 설정 ---
DATASET_FILE = "gung_text_dataset다국어.json"
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1"
CHROMA_DB_DIR = "./chroma_db"

def main():
    """
    JSON 데이터를 로드하고, 검색과 답변 생성에 모두 유리한 '통합 문서' 형태로 DB를 구축합니다.
    """
    if not os.path.exists(DATASET_FILE):
        print(f"❌ '{DATASET_FILE}'을(를) 찾을 수 없습니다. 'collect_data.py'를 먼저 실행하세요.")
        return
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        palace_data = json.load(f)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_ID, trust_remote_code=True
    )
    # 새로운 전략을 위해 새 컬렉션 이름을 사용합니다.
    collection = client.get_or_create_collection(
        name="palace_docs_final", 
        embedding_function=embedding_function
    )

    documents, metadatas, ids = [], [], []
    
    print("통합 문서(Chunk) 및 데이터 준비를 시작합니다...")
    for i, item in enumerate(palace_data):
        # --- [최종 핵심 수정] ---
        # 작은 청크로 나누는 대신, 하나의 풍부한 정보를 가진 문서로 만듭니다.
        # 이렇게 하면 검색도 잘 되고, LLM이 다른 문서를 참고하여 혼란을 겪는 일이 없습니다.
        doc_text = (
            f"제목: {item.get('궁궐', '')} {item.get('건물_한', '')}\n\n"
            f"궁궐: {item.get('궁궐', '')}\n"
            f"건물 (한국어): {item.get('건물_한', '')}\n"
            f"건물 (영어): {item.get('건물_영', '')}\n\n"
            f"--- 한국어 설명 ---\n{item.get('설명_한', '')}\n\n"
            f"--- English Description ---\n{item.get('설명_영', '')}\n\n"
            f"--- 日本語 説明 ---\n{item.get('설명_일', '')}\n\n"
            f"--- 中文 说明 ---\n{item.get('설명_중', '')}"
        )
        
        documents.append(f"search_document: {doc_text}")
        metadatas.append({"궁궐": item.get("궁궐", ""), "건물": item.get("건물_한", "")})
        ids.append(f"doc_{i}")
        
    if not documents:
        print("⚠️ 처리할 문서가 없습니다.")
        return

    print(f"총 {len(documents)}개의 통합 문서를 DB에 추가/업데이트합니다...")
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✅ DB 구축 완료! '{CHROMA_DB_DIR}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()