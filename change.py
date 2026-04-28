import json

# 파일 경로
input_file = "gung_text_dataset다국어.json"

# 결과 저장용 리스트
documents = []
metadatas = []
ids = []

# 언어 키 매핑 (설명 키 → lang 코드)
lang_keys = {
    "설명_한": "ko",
    "설명_영": "en",
    "설명_일": "ja",
    "설명_중": "zh"
}

# JSON 읽기
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 변환 로직
for item in data:
    building = item.get("건물_한")
    palace = item.get("궁궐")

    for desc_key, lang in lang_keys.items():
        content = item.get(desc_key)
        if content:
            documents.append(content)
            metadatas.append({
                "building": building,
                "palace": palace,
                "lang": lang
            })
            ids.append(f"{building}_{lang}")

# 결과 확인 (선택)
print("✅ 변환 완료")
print(f"총 문서 수: {len(documents)}")

# 필요하면 저장
with open("gung_text_chromadb.json", "w", encoding="utf-8") as f:
    json.dump({
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids
    }, f, ensure_ascii=False, indent=2)
