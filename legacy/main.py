from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from contextlib import asynccontextmanager
import cv2
import numpy as np

# --- 모델 및 lifespan 관리자 ---
# YOLOv8 모델을 사용하므로 food_model 변수명을 유지합니다.
food_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global food_model
    print("모델 로드 중...")
    try:
        # 학습된 YOLOv8 모델 가중치 파일을 로드합니다. (예: best.pt)
        food_model = YOLO('models/best.pt')
        print("모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")

    yield
    print("서버 종료 중...")

# --- FastAPI 앱 및 라우터 ---
app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/api")

# --- Pydantic 모델 정의 ---
# 요청 모델은 이제 파일 업로드이므로 필요 없습니다.
class AnalysisResponse(BaseModel):
    detected_objects: list[str]

# --- API 엔드포인트 수정 ---
@router.post("/analyze", response_model=AnalysisResponse)
# ✅ JSON(ImageRequest) 대신 파일(UploadFile)을 직접 받도록 수정합니다.
async def analyze_image(file: UploadFile = File(...)):
    if not food_model:
        raise HTTPException(status_code=503, detail="Server is not ready")

    try:
        # ✅ 업로드된 파일의 내용을 바이트로 읽습니다.
        image_bytes = await file.read()
        
        # 바이트 데이터를 OpenCV 이미지로 변환합니다.
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process image file: {e}")

    # ✅ 이미지 객체 자체를 YOLO 모델로 분석합니다.
    results = food_model(img)
    
    detected_names = []
    for r in results:
        for box in r.boxes:
            class_name = food_model.names[int(box.cls)]
            detected_names.append(class_name)
    
    unique_names = list(set(detected_names))
    print(f"감지된 객체: {unique_names}")
    
    return AnalysisResponse(detected_objects=unique_names)

app.include_router(router)
