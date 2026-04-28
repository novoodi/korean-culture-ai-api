import requests
import xml.etree.ElementTree as ET
import json
import time

# API 엔드포인트
BASE_LIST_URL = "https://www.heritage.go.kr/heri/gungDetail/gogungListOpenApi.do"
BASE_DETAIL_URL = "https://www.heritage.go.kr/heri/gungDetail/gogungDetailOpenApi.do"

# --- 수정된 부분 1: 웹 브라우저처럼 보이게 할 헤더 정보 ---
# 서버가 파이썬 스크립트의 요청을 거부하는 것을 우회하기 위함입니다.
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

print("📜 궁궐 데이터 수집을 시작합니다...")

# 각 궁 번호 순회
for gung_number in GUNG_MAP.keys():
    print(f"\n--- {GUNG_MAP[gung_number]} 데이터 수집 시도 ---")
    list_params = {'gung_number': gung_number}
    
    try:
        # 헤더를 포함하여 요청 전송
        response = requests.get(BASE_LIST_URL, params=list_params, headers=HEADERS, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            print(f"[오류] {GUNG_MAP[gung_number]} 목록 요청 실패 (상태 코드: {response.status_code})")
            continue

        root = ET.fromstring(response.content)

        # --- 수정된 부분 2: 실제 XML 구조에 맞게 'item' 대신 'list'를 찾도록 변경 ---
        items = root.findall('list')
        
        if not items:
            print(f"[{GUNG_MAP[gung_number]}] 수집할 데이터가 없습니다.")
            continue
        
        print(f"[{GUNG_MAP[gung_number]}] {len(items)}개의 항목 발견. 상세 정보를 요청합니다.")

        for item in items:
            try:
                # 목록 API에서 바로 상세 정보를 추출
                contents_kor = item.findtext('.//contents_kor', '').strip()
                explanation_kor = item.findtext('.//explanation_kor', '').strip()
                img_url = item.findtext('.//imgUrl', '').strip()

                # 상세 조회를 위한 정보 추출
                serial_number = item.findtext('serial_number')
                detail_code = item.findtext('detail_code')

                if not serial_number or not detail_code:
                    print("[경고] serial_number 또는 detail_code가 없어 건너뜁니다.")
                    continue

                # 상세 데이터 요청 (추가 정보 획득을 위해)
                detail_params = {
                    'serial_number': int(serial_number),
                    'detail_code': int(detail_code),
                    'gung_number': gung_number
                }
                detail_response = requests.get(BASE_DETAIL_URL, params=detail_params, headers=HEADERS, timeout=10)
                detail_response.encoding = 'utf-8'

                detail_image = ''
                moving_url = ''
                if detail_response.status_code == 200:
                    detail_root = ET.fromstring(detail_response.content)
                    detail_item = detail_root.find('item')
                    if detail_item is not None:
                        detail_image = detail_item.findtext('image', '')
                        moving_url = detail_item.findtext('moving', '')

                data = {
                    '궁궐': GUNG_MAP.get(gung_number),
                    '건물': contents_kor,
                    '설명': explanation_kor,
                    '이미지': img_url,
                    '상세이미지': detail_image,
                    '동영상': moving_url
                }
                result.append(data)

                time.sleep(0.2) # 서버 부하를 줄이기 위한 약간의 지연
            except Exception as e:
                print(f"[내부 오류] 항목 처리 중 예외 발생: {e}")
                continue
                
    except requests.exceptions.RequestException as e:
        print(f"[네트워크 오류] {GUNG_MAP[gung_number]} 목록 요청 중 오류 발생: {e}")
        continue

# 결과 저장
try:
    with open('gung_rag_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    if len(result) > 0:
        print(f"\n✅ 총 {len(result)}건 수집 완료. 'gung_rag_dataset.json' 파일로 저장되었습니다.")
    else:
        print(f"\n⚠️ 수집된 데이터가 없습니다. API 서버 상태를 다시 확인해보세요.")

except IOError as e:
    print(f"[파일 오류] 결과를 저장하는 데 실패했습니다: {e}")
