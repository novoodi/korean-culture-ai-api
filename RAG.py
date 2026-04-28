import json

# 원본 파일 로드
with open("gung_text_dataset다국어.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = data["documents"]

# 4개 언어 단위로 묶기
grouped_data = []
for i in range(0, len(documents), 4):
    ko = documents[i].strip()
    en = documents[i + 1].strip()
    ja = documents[i + 2].strip()
    zh = documents[i + 3].strip()
    
    item = {
        "building": f"건물_{i // 4 + 1}",  # 나중에 실제 건물명으로 매핑할 수 있음
        "texts": {
            "ko": ko,
            "en": en,
            "ja": ja,
            "zh": zh
        }
    }
    grouped_data.append(item)

# 결과 저장 (JSONL)
with open("gung_multilingual_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in grouped_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"{len(grouped_data)}개의 건물 정보가 변환되어 저장되었습니다.")
