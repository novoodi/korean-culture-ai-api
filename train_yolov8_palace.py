from ultralytics import YOLO
import os

def train_yolov8_palace_model():
    """
    YOLOv8 모델을 궁궐 커스텀 데이터셋으로 학습합니다.
    """
    try:
        # 사전 학습된 YOLOv8s 모델 로드 (전이 학습 시작)
        # 새로운 모델을 학습하므로, 이전에 학습된 한식 모델을 로드하는 대신,
        # YOLOv8s 기본 모델부터 시작합니다. (yolov8s.pt)
        model = YOLO('yolov8s.pt')

        # 학습 설정 (data.yaml 파일 경로 지정)
        # my_korean_palace.yaml 파일이 현재 스크립트와 같은 경로에 있거나
        # 정확한 상대/절대 경로를 지정해야 합니다.
        results = model.train(
            data='my_korean_palace.yaml',  # 궁궐 데이터셋 설정 파일
            epochs=50,                   # 학습 에폭 수 (충분한 학습을 위해 50~100 이상 권장)
            imgsz=640,                   # 이미지 크기 (640x640 픽셀)
            batch=16,                    # 배치 크기 (GPU 메모리에 따라 조절)
            name='korean_palace_detection_v1', # 학습 결과가 저장될 폴더 이름
            pretrained=True              # 사전 학습된 가중치 사용 (필수)
            # cache=True                 # (선택 사항) 데이터를 메모리에 캐시하여 학습 속도 향상
            # device='0'                 # (선택 사항) 특정 GPU 사용 (예: device='0', '0,1,2,3' 또는 'cpu')
        )
        print("\n✅ YOLOv8 궁궐 모델 학습이 성공적으로 완료되었습니다!")
        # 학습된 모델은 `runs/detect/korean_palace_detection_v1/weights/best.pt` 경로에 저장됩니다.
    except Exception as e:
        print(f"❌ YOLOv8 모델 학습 중 오류 발생: {e}")
        print("데이터셋 경로, my_korean_palace.yaml 파일, 그리고 GPU 설정 등을 확인해주세요.")

if __name__ == "__main__":
    train_yolov8_palace_model()