import os
import shutil
from sklearn.model_selection import train_test_split # scikit-learn 설치 필요: pip install scikit-learn

# ====================================================================
#           ***** 반드시 수정해야 할 경로들 *****
# ====================================================================
# KIST 한식 데이터셋의 최상위 폴더 경로 (예: C:/Users/admin/kfood_original)
# KIST 데이터셋을 다운로드 받아 압축 해제한 폴더의 경로로 변경하세요.
ORIGINAL_KFOOD_DATASET_ROOT = 'C:/Users/admin4/microstone/한국 음식 이미지' # 예시: 실제 경로로 변경 필요

# 변환된 YOLO 형식 데이터셋이 저장될 루트 폴더 (예: C:/Users/admin/yolo_kfood_dataset)
# 이 폴더 안에 images/train, images/val, labels/train, labels/val이 생성됩니다.
YOLO_DATASET_OUTPUT_ROOT = 'C:/Users/admin4/microstone/yolo_kfood_dataset' # 예시: 실제 경로로 변경 필요
# ====================================================================

# 학습/검증 데이터 분할 비율
TRAIN_RATIO = 0.8

# 클래스 이름 목록 (my_korean_food.yaml의 names와 동일한 순서여야 합니다!)
# 아래 리스트는 제공해주신 텍스트를 기반으로 작성되었습니다.
# 실제 KIST 한식 데이터셋의 폴더명(소분류명)과 정확히 일치하는지 다시 확인해주세요.
CLASS_NAMES = [
    # 구이 (13가지) - 수정: '조기구이' 추가
    '갈비구이', '갈치구이', '고등어구이', '곱창구이', '닭갈비', '더덕구이', '떡갈비', '불고기',
    '삼겹살', '장어구이', '조개구이', '황태구이', '훈제오리', '조기구이', # 추가됨
    # 국 (9가지) - 수정: '떡국_만두국' 추가 (기존 떡국/만두국을 이름에 맞춰 수정)
    '계란국', '떡국_만두국', # 이름 수정
    '무국', '미역국', '북엇국', '소고기무국', '시래기국', '육개장', '콩나물국',
    # 김치 (11가지)
    '갓김치', '깍두기', '나박김치', '무생채', '배추김치', '백김치', '부추김치', '열무김치',
    '오이소박이', '총각김치', '파김치',
    # 나물 (6가지)
    '가지볶음', '고사리나물', '미역줄기볶음', '숙주나물', '시금치나물', '애호박볶음',
    # 떡 (1가지) - 수정: '꿀떡', '송편' 추가
    '경단', '꿀떡', '송편', # 추가됨
    # 만두 (1가지)
    '만두',
    # 면 (13가지) - 수정: '짜장면' 추가 (이전에 '자장면'으로 되어있었으니 '짜장면'으로 통일)
    '막국수', '물냉면', '비빔냉면', '수제비', '열무국수', '잔치국수', '쫄면', '칼국수',
    '콩국수', '라면', '짜장면', '짬뽕', # '짜장면'으로 수정 (이전에 '자장면'이었을 수 있음)
    # 무침 (7가지) - 수정: '회무침' 추가
    '고추된장무침', '꽈리고추무침', '도토리묵', '잡채', '도라지무침', '콩나물무침', '홍어무침', '회무침', # 추가됨
    # 밥 (8가지) - 수정: '누룽지' 추가
    '김밥', '김치볶음밥', '비빔밥', '새우볶음밥', '알밥', '잡곡밥', '주먹밥', '유부초밥', '누룽지', # 추가됨
    # 볶음 (12가지) - 수정: '주꾸미볶음' 추가
    '건새우볶음', '오징어채볶음', '감자채볶음', '고추장진미채볶음', '두부김치', '떡볶이', '라볶이',
    '멸치볶음', '소세지볶음', '어묵볶음', '제육볶음', '쭈꾸미볶음', '주꾸미볶음', # '쭈꾸미볶음'과 함께 또는 하나로 통일
    # 쌈 (1가지)
    '보쌈',
    # 음청류 (2가지)
    '수정과', '식혜',

    # --- 아래는 이전에 누락되었던 대분류와 소분류들입니다. ---
    # 장 (2가지) - 추가됨
    '간장게장', '양념게장',
    # 장아찌 (1가지) - 추가됨
    '깻잎장아찌',
    # 적 (1가지) - 추가됨
    '떡꼬치', # 문서 상 떡꼬치가 적에 해당하는지 확인 필요
    # 전 (8가지) - 추가됨
    '감자전', '계란말이', '계란후라이', '김치전', '동그랑땡', '생선전', '파전', '호박전',
    # 전골 (1가지) - 추가됨
    '곱창전골',
    # 조림 (11가지) - 추가됨
    '갈치조림', '감자조림', '고등어조림', '꽁치조림', '두부조림', '땅콩조림', '메추리알장조림',
    '연근조림', '우엉조림', '장조림', '코다리조림',
    # 죽 (2가지) - 추가됨
    '전복죽', '호박죽',
    # 찌개 (5가지) - 추가됨
    '김치찌개', '닭계장', '동태찌개', '된장찌개', '순두부찌개',
    # 찜 (10가지) - 추가됨
    '갈비찜', '계란찜', '김치찜', '꼬막찜', '닭볶음탕', '수육', '순대', '족발', '찜닭', '해물찜',
    # 탕 (6가지) - 추가됨
    '갈비탕', '감자탕', '곰탕_설렁탕', '매운탕', '삼계탕', '추어탕',
    # 튀김 (3가지) - 추가됨
    '고추튀김', '새우튀김', '오징어튀김',
    # 한과 (3가지) - 추가됨
    '약과', '약식', '한과',
    # 해물 (2가지) - 추가됨
    '멍게', '산낙지',
    # 회 (2가지) - 추가됨
    '물회', '육회',

    # 기타 영역에서 언급되었지만 대분류 분류가 불확실한 항목들 (데이터셋의 실제 폴더명 확인 필요)
    '과메기', '양념치킨', '젓갈', '콩자반', '편육', '피자', '후라이드치킨'
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

def prepare_dataset():
    if not os.path.exists(ORIGINAL_KFOOD_DATASET_ROOT):
        print(f"오류: 원본 KIST 데이터셋 경로를 찾을 수 없습니다: {ORIGINAL_KFOOD_DATASET_ROOT}")
        print("ORIGINAL_KFOOD_DATASET_ROOT 변수를 실제 경로로 수정해주세요.")
        return

    # 출력 폴더 구조 생성
    output_images_train = os.path.join(YOLO_DATASET_OUTPUT_ROOT, 'images', 'train')
    output_images_val = os.path.join(YOLO_DATASET_OUTPUT_ROOT, 'images', 'val')
    output_labels_train = os.path.join(YOLO_DATASET_OUTPUT_ROOT, 'labels', 'train')
    output_labels_val = os.path.join(YOLO_DATASET_OUTPUT_ROOT, 'labels', 'val')

    for folder in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
        os.makedirs(folder, exist_ok=True)

    all_image_paths = []
    all_class_ids = []

    # 원본 데이터셋을 탐색하며 이미지 경로와 클래스 ID 수집
    print(f"원본 데이터셋 탐색 중: {ORIGINAL_KFOOD_DATASET_ROOT}")
    processed_small_categories = set() # 처리된 소분류를 추적하여 경고 중복 방지

    for big_cat_folder in os.listdir(ORIGINAL_KFOOD_DATASET_ROOT):
        big_cat_path = os.path.join(ORIGINAL_KFOOD_DATASET_ROOT, big_cat_folder)
        if not os.path.isdir(big_cat_path):
            continue

        for small_cat_folder in os.listdir(big_cat_path):
            small_cat_path = os.path.join(big_cat_path, small_cat_folder)
            if not os.path.isdir(small_cat_path):
                continue

            class_id = CLASS_TO_ID.get(small_cat_folder)
            if class_id is None:
                if small_cat_folder not in processed_small_categories:
                    print(f"경고: 'CLASS_NAMES'에 정의되지 않은 소분류 폴더 '{small_cat_folder}'가 발견되었습니다. 이 소분류는 스킵됩니다.")
                    processed_small_categories.add(small_cat_folder)
                continue

            for image_name in os.listdir(small_cat_path):
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(small_cat_path, image_name)
                    all_image_paths.append(image_path)
                    all_class_ids.append(class_id)
        print(f"대분류 '{big_cat_folder}' 탐색 완료.")

    if not all_image_paths:
        print("\n오류: 원본 데이터셋에서 이미지를 찾을 수 없습니다.")
        print("ORIGINAL_KFOOD_DATASET_ROOT 경로가 정확한지, 그리고 그 안에 '대분류/소분류/이미지.jpg'와 같은 구조로 이미지가 있는지 확인해주세요.")
        return

    # 학습/검증 데이터 분할
    print("학습/검증 데이터 분할 중...")
    train_img_paths, val_img_paths, train_class_ids, val_class_ids = train_test_split(
        all_image_paths, all_class_ids, test_size=1 - TRAIN_RATIO, random_state=42, stratify=all_class_ids
    )

    print(f"총 이미지 수: {len(all_image_paths)}")
    print(f"학습 이미지 수: {len(train_img_paths)}")
    print(f"검증 이미지 수: {len(val_img_paths)}")

    # 학습 데이터 복사 및 라벨 생성
    print("학습 데이터 처리 중...")
    for i, img_path in enumerate(train_img_paths):
        class_id = train_class_ids[i]
        img_filename = os.path.basename(img_path)
        base_filename = os.path.splitext(img_filename)[0]

        # 이미지 복사
        shutil.copy(img_path, os.path.join(output_images_train, img_filename))

        # 라벨 파일 생성 (이미지 전체를 바운딩 박스로 가정)
        label_filepath = os.path.join(output_labels_train, f"{base_filename}.txt")
        with open(label_filepath, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n") # class_id center_x center_y width height
    print("학습 데이터 처리 완료.")

    # 검증 데이터 복사 및 라벨 생성
    print("검증 데이터 처리 중...")
    for i, img_path in enumerate(val_img_paths):
        class_id = val_class_ids[i]
        img_filename = os.path.basename(img_path)
        base_filename = os.path.splitext(img_filename)[0]

        # 이미지 복사
        shutil.copy(img_path, os.path.join(output_images_val, img_filename))

        # 라벨 파일 생성 (이미지 전체를 바운딩 박스로 가정)
        label_filepath = os.path.join(output_labels_val, f"{base_filename}.txt")
        with open(label_filepath, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n") # class_id center_x center_y width height
    print("검증 데이터 처리 완료.")

    print("\n✅ KIST 한식 데이터셋 변환 및 YOLO 포맷 준비 완료!")
    print(f"결과 폴더: {YOLO_DATASET_OUTPUT_ROOT}")
    print(f"my_korean_food.yaml의 'path'를 '{YOLO_DATASET_OUTPUT_ROOT}'로, 'nc'를 {len(CLASS_NAMES)}로, 'names'를 이 스크립트의 CLASS_NAMES와 동일하게 설정해주세요.")


if __name__ == '__main__':
    # 이 함수를 실행하기 전에 'ORIGINAL_KFOOD_DATASET_ROOT'와 'YOLO_DATASET_OUTPUT_ROOT' 경로를
    # 본인의 컴퓨터에 맞게 정확히 설정해야 합니다.
    # 또한 CLASS_NAMES 리스트가 KIST 한식 데이터셋의 소분류로 완전히 채워져 있고 정확한지 확인해야 합니다.
    print("데이터셋 변환 스크립트를 실행합니다. 시작하기 전에 경로 설정을 확인해주세요.")
    prepare_dataset()