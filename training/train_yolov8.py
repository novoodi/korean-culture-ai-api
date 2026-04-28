from ultralytics import YOLO

def train_yolov8_model():
    """
    YOLOv8 모델을 커스텀 데이터셋으로 학습합니다.
    """
    try:
        # 사전 학습된 YOLOv8s 모델 로드 (small 버전, 더 좋은 성능을 위해 추천)
        # 'n' (nano) 버전을 사용해도 됩니다.
        model = YOLO('yolov8s.pt') # 학습을 위한 기본 모델 로드

        # 학습 설정 (data.yaml 파일 경로 지정)
        # my_korean_food.yaml 파일이 현재 스크립트와 같은 경로에 있거나
        # 정확한 상대/절대 경로를 지정해야 합니다.
        results = model.train(
            data='my_korean_food.yaml',  # 학습에 사용할 데이터셋 설정 파일
            epochs=50,                   # 학습 에폭 수
            imgsz=640,                   # 이미지 크기
            batch=16,                    # 배치 크기
            name='korean_food_detection_v1', # 학습 결과가 저장될 폴더 이름
            pretrained=True              # 사전 학습된 가중치 사용
            # cache=True                 # (선택 사항) 데이터를 메모리에 캐시하여 학습 속도 향상
            # device='0'                 # (선택 사항) 특정 GPU 사용 (예: device='0', '0,1,2,3' 또는 'cpu')
        )
        print("\n✅ YOLOv8 모델 학습이 성공적으로 완료되었습니다!")
        # 학습된 모델은 `runs/detect/korean_food_detection_v1/weights/best.pt` 경로에 저장됩니다.
    except Exception as e:
        print(f"❌ YOLOv8 모델 학습 중 오류 발생: {e}")
        print("데이터셋 경로, my_korean_food.yaml 파일, 그리고 GPU 설정 등을 확인해주세요.")

if __name__ == "__main__":
    train_yolov8_model()