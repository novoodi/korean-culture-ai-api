# AI Cultural Guide Server — ForForeigner

> 딥러닝 이미지 분석과 LLM의 연계를 통한 외국인 문화 정보 제공 웹 시스템의 **AI 서버** 파트입니다.  
> 2025 융복합지식학회 추계학술발표대회 논문으로 발표되었습니다.

## Overview

외국인 대상 생활 정착 지원 웹 서비스 **ForForeigner**의 AI 백엔드 서버입니다.  
사용자가 촬영한 **궁궐 또는 한식 이미지**를 받아, 객체를 탐지하고 RAG 기반으로 정확한 설명을 생성합니다.

```
이미지 입력
    ↓
RT-DETR / YOLOv8 (객체 탐지)
    ↓
궁궐 모델 / 음식 모델 (도메인 분류)
    ↓
RAG 파이프라인 (FAISS + ko-sbert + Phi-3-Mini)
    ↓
한국어 설명 생성 → 사용자에게 반환
```

## System Architecture

ForForeigner 전체 시스템은 **프론트엔드(React), 백엔드 API(Spring Boot), 데이터베이스(AWS RDS/MySQL), 이미지 저장소(AWS S3), AI 서버(FastAPI)** 로 구성됩니다.  
본 레포지토리는 AI 서버 모듈에 해당합니다.

## Key Features

### Object Detection
- **궁궐 모델**: 5대 궁궐(경복궁·창덕궁·창경궁·덕수궁·종묘) 건축물 탐지
  - mAP@0.5: **0.992**, F1-Score: **0.99**
- **음식 모델**: 한식 153개 클래스 탐지 (AI-Hub 한국 음식 이미지 데이터셋)
  - mAP@0.5: **0.639**, F1-Score: **0.65**

### RAG Description Generation
- **LLM**: Phi-3-Mini-4k-instruct (GGUF quantized, llama-cpp-python)
- **Embedding**: `jhgan/ko-sbert-nli` (한국어 의미 유사도)
- **Vector Store**: FAISS (인메모리 고속 유사도 검색)
- **Framework**: LangChain
- 도메인별 **역할 기반 프롬프트 템플릿**으로 환각(Hallucination) 억제

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | 서버 상태 확인 |
| POST | `/api/analyze` | 이미지 URL로 객체 탐지 (palace/food) |
| POST | `/api/describe` | 탐지된 객체 이름으로 RAG 설명 생성 |

### Request Examples

**POST /api/analyze**
```json
{
  "image_url": "https://your-s3-bucket.../image.jpg",
  "type": "palace"
}
```

**POST /api/describe**
```json
{
  "object_name": "근정전"
}
```
Query param: `?type=palace`

## Project Structure

```
├── main3.py               # 메인 서버 (최종) — 궁궐+음식 YOLO + RAG
├── main2.py               # v2 — HuggingFace Phi-3 직접 로드 방식
├── main.py                # v1 — YOLO 단독 서버
├── train_yolov8.py        # 음식 YOLOv8 학습 스크립트
├── train_yolov8_palace.py # 궁궐 YOLOv8 학습 스크립트
├── prepare_kfood_dataset.py   # 음식 데이터셋 전처리
├── prepare_palace_dataset.py  # 궁궐 데이터셋 전처리
├── API.py                 # 궁궐 데이터 수집 (heritage.go.kr)
├── API다국어.py            # 다국어(한/영/일/중) 궁궐 데이터 수집
├── llmRAG.py              # ChromaDB 벡터 DB 구축
├── palace_data.json       # 궁궐 RAG 데이터
├── food_data.json         # 음식 RAG 데이터
├── my_korean_food.yaml    # 음식 YOLO 학습 설정
├── my_korean_palace.yaml  # 궁궐 YOLO 학습 설정
└── models/
    ├── best.pt            # 궁궐 탐지 모델 가중치 (gitignore)
    ├── food2.pt           # 음식 탐지 모델 가중치 (gitignore)
    └── Phi-3-mini-4k-instruct-q4.gguf  # LLM (gitignore)
```

## Getting Started

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

PyTorch는 CUDA 버전에 맞게 별도 설치:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 모델 파일 준비

`models/` 디렉토리에 아래 파일을 배치합니다 (용량이 커서 별도 배포):

| 파일 | 크기 | 설명 |
|------|------|------|
| `best.pt` | ~191MB | 궁궐 YOLOv8 가중치 |
| `food2.pt` | ~64MB | 음식 YOLOv8 가중치 |
| `Phi-3-mini-4k-instruct-q4.gguf` | ~2.3GB | Phi-3-Mini LLM |

### 3. 서버 실행

```bash
uvicorn main3:app --host 0.0.0.0 --port 8000
```

## Dataset

- **궁궐**: 한국문화정보원 '조선의 5대 궁궐 및 종료 건축물' 데이터 (5,000장)
- **음식**: 과학기술정보통신부 AI-Hub '한국 음식 이미지' 데이터 (153개 클래스)

## Tech Stack

| Category | Technology |
|----------|------------|
| Web Framework | FastAPI |
| Object Detection | YOLOv8 (Ultralytics) |
| LLM | Phi-3-Mini-4k-instruct (llama-cpp-python) |
| RAG | LangChain + FAISS |
| Embedding | jhgan/ko-sbert-nli |
| CV | OpenCV, PyTorch |
| Deployment | Cloudflare Tunnel |

## Paper

> 고현지, 김정현, 박종섭, 서원후, 양예찬, 이예은, 정윤정, 이동규.  
> "딥러닝 이미지 분석과 LLM의 연계를 통한 외국인 문화 정보 제공 웹 시스템 구현"  
> 2025년도 (사)융복합지식학회 추계학술발표대회 논문집.  
> 신한대학교.

## Acknowledgement

본 연구는 2023년도 과학기술정보통신부 및 정보통신기획평가원의 SW중심대학사업 지원을 받아 수행되었음 (2023-0-00089)
