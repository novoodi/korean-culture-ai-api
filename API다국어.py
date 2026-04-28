import requests
import xml.etree.ElementTree as ET
import json
import time

# API 엔드포인트
BASE_LIST_URL = "https://www.heritage.go.kr/heri/gungDetail/gogungListOpenApi.do"
BASE_DETAIL_URL = "https://www.heritage.go.kr/heri/gungDetail/gogungDetailOpenApi.do"

# 웹 브라우저처럼 보이게 할 헤더 정보
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

# 궁궐 번호 매핑
GUNG_MAP = {
    1: "경복궁",
    2: "창덕궁",
    3: "창경궁",
    4: "덕수궁",
    5: "종묘"
}

# 결과 리스트 초기화
result = []

print("📜 다국어 텍스트 데이터 수집을 시작합니다...")

for gung_number in GUNG_MAP.keys():
    print(f"\n--- {GUNG_MAP[gung_number]} 데이터 수집 시도 ---")
    list_params = {'gung_number': gung_number}
    
    try:
        response = requests.get(BASE_LIST_URL, params=list_params, headers=HEADERS, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            print(f"[오류] {GUNG_MAP[gung_number]} 목록 요청 실패 (상태 코드: {response.status_code})")
            continue

        root = ET.fromstring(response.content)
        items = root.findall('list')
        
        if not items:
            print(f"[{GUNG_MAP[gung_number]}] 수집할 데이터가 없습니다.")
            continue
        
        print(f"[{GUNG_MAP[gung_number]}] {len(items)}개의 항목 발견. 상세 정보를 요청합니다.")

        for item in items:
            try:
                serial_number = item.findtext('serial_number')
                detail_code = item.findtext('detail_code')

                if not serial_number or not detail_code:
                    print("[경고] serial_number 또는 detail_code가 없어 건너뜁니다.")
                    continue

                detail_params = {
                    'serial_number': int(serial_number),
                    'detail_code': int(detail_code),
                    'gung_number': gung_number
                }
                detail_response = requests.get(BASE_DETAIL_URL, params=detail_params, headers=HEADERS, timeout=10)
                detail_response.encoding = 'utf-8'

                if detail_response.status_code == 200:
                    detail_root = ET.fromstring(detail_response.content)

                    # --- [핵심 수정] ---
                    # 실제 XML 구조에 맞게 <item> 태그 없이 detail_root에서 직접 데이터를 찾습니다.
                    contents_kor = detail_root.findtext('contents_kor', '').strip()
                    contents_eng = detail_root.findtext('contents_eng', '').strip()
                    contents_jpa = detail_root.findtext('contents_jpa', '').strip()
                    contents_chi = detail_root.findtext('contents_chi', '').strip()
                    
                    explanation_kor = detail_root.findtext('explanation_kor', '').strip()
                    explanation_eng = detail_root.findtext('explanation_eng', '').strip()
                    explanation_jpa = detail_root.findtext('explanation_jpa', '').strip()
                    explanation_chi = detail_root.findtext('explanation_chi', '').strip()
                
                    # 데이터가 하나라도 있는 경우에만 결과에 추가
                    if contents_kor or explanation_kor:
                        data = {
                            '궁궐': GUNG_MAP.get(gung_number),
                            '건물_한': contents_kor,
                            '건물_영': contents_eng,
                            '건물_일': contents_jpa,
                            '건물_중': contents_chi,
                            '설명_한': explanation_kor,
                            '설명_영': explanation_eng,
                            '설명_일': explanation_jpa,
                            '설명_중': explanation_chi,
                        }
                        result.append(data)
                else:
                    print(f"[오류] 상세 조회 실패 (S/N: {serial_number}, 상태 코드: {detail_response.status_code})")

                time.sleep(0.1)
            except Exception as e:
                print(f"[내부 오류] 항목 처리 중 예외 발생: {e}")
                continue
                
    except requests.exceptions.RequestException as e:
        print(f"[네트워크 오류] {GUNG_MAP[gung_number]} 목록 요청 중 오류 발생: {e}")
        continue

# 결과 저장
try:
    with open('gung_text_dataset다국어.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    if len(result) > 0:
        print(f"\n✅ 총 {len(result)}건 수집 완료. 'gung_text_dataset다국어.json' 파일로 저장되었습니다.")
    else:
        print(f"\n⚠️ 수집된 데이터가 없습니다. API 서버 상태를 다시 확인해보세요.")

except IOError as e:
    print(f"[파일 오류] 결과를 저장하는 데 실패했습니다: {e}")