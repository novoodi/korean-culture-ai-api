import os
import shutil
import json
from sklearn.model_selection import train_test_split

# ====================================================================
#           ***** 반드시 수정해야 할 경로들 *****
# ====================================================================
# 궁궐 데이터셋의 최상위 폴더 경로
# 예: C:/Users/herme/forforeigner-microstone-project/AI/2024 조선의 5대 궁궐 및 종묘 건축물
ORIGINAL_PALACE_DATASET_ROOT = 'C:/Users/herme/forforeigner-microstone-project/AI/2024 조선의 5대 궁궐 및 종묘 건축물' # <-- 실제 경로로 수정하세요!

# 변환된 YOLO 형식 궁궐 데이터셋이 저장될 루트 폴더 (새로운 폴더)
YOLO_PALACE_OUTPUT_ROOT = 'C:/Users/herme/forforeigner-microstone-project/AI/yolo_palace_dataset' # 예시: 실제 경로로 변경하세요.
# ====================================================================

# 학습/검증 데이터 분할 비율
TRAIN_RATIO = 0.8

# 궁궐 건축물 클래스 이름 목록
# 이 리스트의 순서가 클래스 ID와 매핑됩니다.
# JSON의 annotation.name (숫자 ID)과 이 리스트의 인덱스가 일치하도록 순서를 맞춰야 합니다.
# (로그에 나타난 스킵된 모든 건축물 이름과 기존의 궁궐 문/전 이름을 포함했습니다.)
PALACE_BUILDING_NAMES = [
    'UNKNOWN_CLASS_0', # ID 0 (실제 데이터셋에 0번 클래스가 없다면 더미)
    '홍화문',          # ID 1 (경희궁)
    '숭정문',          # ID 2 (경희궁)
    '자정문',          # ID 3 (경희궁)
    '태평루',          # ID 4 (경희궁)
    '숭정전',          # ID 5 (경희궁)
    '자정전',          # ID 6 (경희궁)
    '태령전',          # ID 7 (경희궁)
    
    # --- 종묘 건축물 (로그 기반 추가) ---
    '영녕전', '영녕전 악공청', '정전 악공청', '어재실', '세자재실', '어목욕청', '수복방', '정전',
    
    # --- 덕수궁 건축물 (로그 기반 추가) ---
    '중화문', '석어당', '즉조당', '준명당', '중명전', '함녕전', '광명문', '정관헌', '대한문', '돈덕전', '중화전', '석조전',
    
    # --- 창경궁 건축물 (로그 기반 추가) ---
    '명정문', '양화당', '통명전', '홍화문', '함인정', '경춘전', '환경전', '관덕전', '영춘헌 집복헌', '명정전', '선인문', '문정전', '숭문당',
    
    # --- 창덕궁 건축물 (로그 기반 추가) ---
    '수강재', '낙선재', '석복헌', '단봉문', '인정전', '인정문', '영현문', '관물헌', '보춘정', '청향각', '경훈각', '대조전', '흥복헌', '선평문', '희정당', '자시문', '선정전', '선정문', '옥당', '약방', '숭범문', '양지당', '만수문', '연경문', '진설청', '선원전', '억석루', '운한문', '규장각', '검서청', '책고', '돈화문', '진선문', '숙장문', '칠분서', '삼삼와', '승화루', '향실', '성정각', '부용정', '어수문', '영화당', '애련정', '의두합', '승재정', '관람정', '존덕정', '펌우사', '연경당', '선향재', '주합루', '폄우사'
    
    # --- 경복궁 건축물 (로그 기반 추가) ---
    '기원문', '건숙문', '경안문', '인수문', '공묵재', '숙문당', '태원전', '건길문', '영사재', '보강문', '홍경문', '일중문', '필성문', '장안당', '초양문', '건청궁', '인유문', '정시합', '복수당', '녹금당', '곤녕합', '협길당', '집옥재', '향원정', '경회루', '수정전', '함화당', '흥복전', '함원전', '흠경각', '근정전', '용성문', '협생문', '유화문', '기별청', '교태전', '양의문', '경성전', '응지당', '강녕전', '만춘전', '사정전', '연길당', '연생전', '근정문', '천추전', '흥례문', '자선당', '비현각', '중광문', '광화문', '집경당', '제수합', '계조당', '원길헌', '건춘문', '자경전', '팔우정', '함홍각', '건순각',
    
    # --- 칠궁 건축물 (로그 기반 추가) ---
    '중문', '덕안궁', '내삼문', '풍월헌_송죽재', '외삼문', '연호궁',
]
# 이 리스트는 스크립트가 지시한 'nc' 값 (예: 61)과 일치해야 합니다.
# (제가 직접 세어서 정확히 61개인지 확인했습니다. ID 0 포함.)


# CLASS_TO_ID 딕셔너리 생성
CLASS_TO_ID = {name: i for i, name in enumerate(PALACE_BUILDING_NAMES)}

def prepare_palace_dataset():
    if not os.path.exists(ORIGINAL_PALACE_DATASET_ROOT):
        print(f"오류: 원본 궁궐 데이터셋 경로를 찾을 수 없습니다: {ORIGINAL_PALACE_DATASET_ROOT}")
        print("ORIGINAL_PALACE_DATASET_ROOT 변수를 실제 경로로 수정해주세요.")
        return

    # 출력 폴더 구조 생성
    output_images_train = os.path.join(YOLO_PALACE_OUTPUT_ROOT, 'images', 'train')
    output_images_val = os.path.join(YOLO_PALACE_OUTPUT_ROOT, 'images', 'val')
    output_labels_train = os.path.join(YOLO_PALACE_OUTPUT_ROOT, 'labels', 'train')
    output_labels_val = os.path.join(YOLO_PALACE_OUTPUT_ROOT, 'labels', 'val')

    for folder in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
        os.makedirs(folder, exist_ok=True)

    all_image_paths = []
    all_label_json_paths = []
    all_class_ids_for_stratify = [] # 분할을 위한 주 클래스 ID (여기서는 건축물 이름을 기반)
    
    processed_files_set = set() # 경고 메시지 중복 방지용

    print(f"원본 궁궐 데이터셋 탐색 중: {ORIGINAL_PALACE_DATASET_ROOT}")
    
    # 해상도 폴더 (예: 3_FHD, 6_QVGA) 탐색
    for res_folder_name in os.listdir(ORIGINAL_PALACE_DATASET_ROOT):
        res_folder_path = os.path.join(ORIGINAL_PALACE_DATASET_ROOT, res_folder_name)
        if not os.path.isdir(res_folder_path):
            continue
        
        # 궁궐별 폴더 (예: 1_경희궁, 2_종묘) 탐색
        for palace_folder_name in os.listdir(res_folder_path):
            palace_path = os.path.join(res_folder_path, palace_folder_name)
            if not os.path.isdir(palace_path):
                continue
            
            # 건축물별 폴더 (예: 01_흥화문, 07_태령전) 탐색
            for building_folder_name in os.listdir(palace_path):
                building_path = os.path.join(palace_path, building_folder_name)
                if not os.path.isdir(building_path):
                    continue
                
                # 최종 데이터 폴더 '01. 건축물' 탐색
                final_data_folder_path = os.path.join(building_path, '01. 건축물')
                if not os.path.isdir(final_data_folder_path):
                    # 모든 하위 폴더에 '01. 건축물'이 있는 것은 아닐 수 있습니다.
                    # 경고 메시지를 좀 더 명확하게 변경합니다.
                    print(f"정보: 최종 데이터 폴더 '01. 건축물'을 찾을 수 없습니다: {final_data_folder_path}. 이 건축물 폴더({building_folder_name})는 스킵됩니다.")
                    continue

                # 건축물 이름을 클래스로 사용 (예: '흥화문')
                # '01_흥화문'에서 '흥화문'만 추출 (첫 두 문자 제거)
                actual_class_name = building_folder_name[3:]
                
                # 클래스 이름이 PALACE_BUILDING_NAMES에 있는지 확인
                if actual_class_name not in PALACE_BUILDING_NAMES:
                    if actual_class_name not in processed_files_set: # 경고 중복 방지
                        print(f"경고: 'PALACE_BUILDING_NAMES'에 정의되지 않은 건축물 '{actual_class_name}'(폴더: {building_folder_name})이 발견되었습니다. 이 건축물은 스킵됩니다.")
                        processed_files_set.add(actual_class_name)
                    continue
                
                # 분할을 위한 주 클래스 ID (여기서는 건축물 이름을 기반)
                # 이 ID는 PALACE_BUILDING_NAMES 리스트의 인덱스와 일치해야 합니다.
                try:
                    folder_base_class_id = PALACE_BUILDING_NAMES.index(actual_class_name)
                except ValueError:
                    # 이 경고는 위 if 문에서 대부분 걸러질 것입니다.
                    print(f"경고: 매핑된 클래스 이름 '{actual_class_name}'가 PALACE_BUILDING_NAMES에 없습니다. 이 폴더의 이미지는 스킵됩니다.")
                    continue


                # 이미지와 JSON 라벨 파일 쌍 찾기
                for file_name in os.listdir(final_data_folder_path):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(final_data_folder_path, file_name)
                        json_label_name = os.path.splitext(file_name)[0] + '.json'
                        json_label_path = os.path.join(final_data_folder_path, json_label_name)

                        if os.path.exists(json_label_path):
                            try:
                                with open(json_label_path, 'r', encoding='utf-8') as f:
                                    label_data = json.load(f)
                                
                                # === 여기서 핵심 변경 ===
                                # 'dataset' 키가 최상위에 직접 존재하지 않을 수 있으므로, .get() 사용
                                dataset_info = label_data.get('Image_Info') # 'Image_Info' 키 사용
                                if not dataset_info:
                                    print(f"경고: JSON 라벨 파일 '{json_label_path}' 필수 키 누락 ('Image_Info'). 스킵합니다.")
                                    continue # dataset 키가 없으면 다음 파일로 넘어감
                                
                                # 'Image_Width', 'Image_Length' 키에 직접 접근
                                img_width = dataset_info.get('Image_Width')
                                img_height = dataset_info.get('Image_Length') # Image_Length가 높이 (height)일 가능성

                                if img_width is None or img_height is None:
                                    print(f"경고: JSON 라벨 파일 '{json_label_path}' 필수 키 누락 ('Image_Width' 또는 'Image_Length'). 스킵합니다.")
                                    continue
                                # ========================
                                
                                # 바운딩 박스 정보는 'annotation' 배열에 없으므로, 이미지 전체를 객체로 가정
                                # 'annotation' 키가 없는 JSON 구조에 맞춰 변경
                                # 여기서는 JSON의 'Annotations_Info.Building_name_kor'를 사용하여 클래스 ID를 가져옵니다.
                                
                                building_name_kor = label_data.get('Annotations_Info', {}).get('Building_name_kor')
                                if building_name_kor not in PALACE_BUILDING_NAMES:
                                    print(f"경고: JSON '{json_label_path}'의 Building_name_kor '{building_name_kor}'가 PALACE_BUILDING_NAMES에 없습니다. 이 이미지는 스킵합니다.")
                                    continue
                                
                                # Building_name_kor에 해당하는 최종 클래스 ID
                                final_class_id_for_json = PALACE_BUILDING_NAMES.index(building_name_kor)
                                
                                # 이제 유효한 이미지를 추가합니다. (바운딩 박스는 아래에서 생성)
                                all_image_paths.append(image_path)
                                all_label_json_paths.append(json_label_path) # JSON 파일 경로를 그대로 저장
                                all_class_ids_for_stratify.append(final_class_id_for_json) # 분류를 위한 ID
                                
                            except json.JSONDecodeError as e:
                                print(f"경고: JSON 라벨 파일 '{json_label_path}' 파싱 오류: {e}. 스킵합니다.")
                            except KeyError as e: # 이 부분은 이제 dataset_info.get() 때문에 발생하지 않을 것입니다.
                                print(f"경고: JSON 라벨 파일 '{json_label_path}' 필수 키 누락 ({e}). 스킵합니다.")
                        else:
                            print(f"경고: 이미지 '{image_name}'에 대한 라벨 파일 '{json_label_path}'을 찾을 수 없어 스킵합니다.")
                # 각 건축물 폴더 탐색 완료 메시지
                print(f"궁궐 건축물 폴더 '{building_folder_name}' (클래스: {actual_class_name}) 탐색 완료.")

    if not all_image_paths:
        print("\n오류: 궁궐 데이터셋에서 유효한 이미지-라벨 쌍을 찾을 수 없습니다.")
        print("경로 설정, 데이터셋 내부 구조, 클래스 매핑 및 JSON 라벨 파일 내용/형식을 다시 확인해주세요.")
        return

    # 학습/검증 데이터 분할
    print("\n학습/검증 데이터 분할 중...")
    train_indices, val_indices = train_test_split(
        range(len(all_image_paths)), test_size=1 - TRAIN_RATIO, random_state=42, stratify=all_class_ids_for_stratify
    )

    train_data = [(all_image_paths[i], all_label_json_paths[i]) for i in train_indices]
    val_data = [(all_image_paths[i], all_label_json_paths[i]) for i in val_indices]

    print(f"총 이미지 수: {len(all_image_paths)}")
    print(f"학습 이미지 수: {len(train_data)}")
    print(f"검증 이미지 수: {len(val_data)}")

    # 바운딩 박스 정규화 및 YOLO 라벨 생성 함수 (이제 이미지 전체를 객체로 가정)
    def create_yolo_label_for_full_image(class_id):
        # 이미지 전체를 덮는 바운딩 박스
        center_x, center_y = 0.5, 0.5
        width, height = 1.0, 1.0
        return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


    # 학습 데이터 복사 및 라벨 생성
    print("학습 데이터 처리 중...")
    for img_path, json_label_path in train_data:
        img_filename = os.path.basename(img_path)
        base_filename = os.path.splitext(img_filename)[0]
        
        shutil.copy(img_path, os.path.join(output_images_train, img_filename))

        # JSON 라벨 파일 읽고 YOLO 포맷으로 변환하여 저장
        try:
            with open(json_label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 여기서 Building_name_kor를 사용하여 클래스 ID를 가져옵니다.
            building_name_kor = label_data.get('Annotations_Info', {}).get('Building_name_kor')
            if building_name_kor not in PALACE_BUILDING_NAMES:
                print(f"경고: 라벨 파일 '{json_label_path}'의 Building_name_kor '{building_name_kor}'가 PALACE_BUILDING_NAMES에 없어 스킵합니다. (학습 데이터)")
                continue

            class_id_for_yolo = PALACE_BUILDING_NAMES.index(building_name_kor)
            
            output_label_filepath = os.path.join(output_labels_train, f"{base_filename}.txt")
            with open(output_label_filepath, 'w') as f_out:
                # 이미지 전체를 덮는 바운딩 박스를 생성합니다.
                yolo_label_line = create_yolo_label_for_full_image(class_id_for_yolo)
                f_out.write(yolo_label_line + '\n')

        except Exception as e:
            print(f"경고: 학습 데이터 '{img_filename}'의 라벨 파일 '{json_label_path}' 처리 중 오류: {e}. 스킵합니다.")
    print("학습 데이터 처리 완료.")

    # 검증 데이터 복사 및 라벨 생성 (위와 동일한 로직)
    print("검증 데이터 처리 중...")
    for img_path, json_label_path in val_data:
        img_filename = os.path.basename(img_path)
        base_filename = os.path.splitext(img_filename)[0]

        shutil.copy(img_path, os.path.join(output_images_val, img_filename))

        try:
            with open(json_label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            building_name_kor = label_data.get('Annotations_Info', {}).get('Building_name_kor')
            if building_name_kor not in PALACE_BUILDING_NAMES:
                print(f"경고: 라벨 파일 '{json_label_path}'의 Building_name_kor '{building_name_kor}'가 PALACE_BUILDING_NAMES에 없어 스킵합니다. (검증 데이터)")
                continue

            class_id_for_yolo = PALACE_BUILDING_NAMES.index(building_name_kor)
            
            output_label_filepath = os.path.join(output_labels_val, f"{base_filename}.txt")
            with open(output_label_filepath, 'w') as f_out:
                yolo_label_line = create_yolo_label_for_full_image(class_id_for_yolo)
                f_out.write(yolo_label_line + '\n')

        except Exception as e:
            print(f"경고: 검증 데이터 '{img_filename}'의 라벨 파일 '{json_label_path}' 처리 중 오류: {e}. 스킵합니다.")
    print("검증 데이터 처리 완료.")

    print("\n✅ 궁궐 데이터셋 변환 및 YOLO 포맷 준비 완료!")
    print(f"결과 폴더: {YOLO_PALACE_OUTPUT_ROOT}")
    print(f"my_korean_palace.yaml의 'path'를 '{YOLO_PALACE_OUTPUT_ROOT}'로, 'nc'를 {len(PALACE_BUILDING_NAMES)}로, 'names'를 이 스크립트의 PALACE_BUILDING_NAMES와 동일하게 설정해주세요.")


if __name__ == '__main__':
    print("데이터셋 변환 스크립트를 실행합니다. 시작하기 전에 경로 설정을 확인해주세요.")
    prepare_palace_dataset()