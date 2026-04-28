import os
import json
import asyncio
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Query, Form, Request
from pydantic import BaseModel, Field
from ultralytics import YOLO
from contextlib import asynccontextmanager
import cv2
import numpy as np
from typing import Literal

import requests

import torch
from llama_cpp import Llama
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.llms import LlamaCpp

# ------------------- 글로벌 변수 -------------------
palace_model = None      # 궁궐 객체 탐지 YOLO 모델
food_model = None        # 음식 객체 탐지 YOLO 모델
palace_rag_chain = None  # 궁궐 설명 RAG 체인
food_rag_chain = None    # 음식 설명 RAG 체인

# ------------------- 범용 RAG 파이프라인 설정 함수 -------------------
def setup_rag_pipeline(
    data_path: str,
    db_path: str,
    content_key: str,
    source_key: str,
    template: str,
    llm: LlamaCpp,
    hf_embeddings: HuggingFaceEmbeddings
):
    """주어진 설정에 따라 RAG 파이프라인을 생성하는 범용 함수"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일 '{data_path}'를 찾을 수 없습니다.")

    loader = JSONLoader(
        file_path=data_path,
        jq_schema='.[]',
        content_key=content_key,
        metadata_func=lambda record, metadata: {"source": record.get(source_key)}
    )
    documents = loader.load()

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    if os.path.exists(db_path):
        db = FAISS.load_local(db_path, hf_embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(docs, hf_embeddings)
        db.save_local(db_path)
    
    retriever = db.as_retriever(search_kwargs={'k': 1})
    prompt = PromptTemplate.from_template(template)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ------------------- FastAPI 시작/종료 시 실행 (Lifespan) -------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global palace_model, food_model, palace_rag_chain, food_rag_chain

    # 1. 모든 YOLO 모델 로드
    print("--- YOLO 모델 로드 시작 ---")
    try:
        palace_model = YOLO('models/best.pt')
        print("✅ 궁궐 YOLO 모델 로드 완료")
        food_model = YOLO('models/food2.pt')
        print("✅ 음식 YOLO 모델 로드 완료")
    except Exception as e:
        print(f"🚨 YOLO 모델 로드 중 오류 발생: {e}")

    # 2. RAG 파이프라인 공통 모델 (LLM, 임베딩) 한 번만 로드
    print("\n--- RAG 공통 모델 로드 시작 ---")
    try:
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sbert-nli",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        llm = LlamaCpp(
            model_path="./models/phi-3-mini-4k-instruct-q4.gguf",
            n_ctx=2048, n_gpu_layers=-1, temperature=0.2,
            max_tokens=500, n_threads=8, verbose=False
        )
        print("✅ RAG 공통 모델 (LLM, Embedding) 로드 완료")
    except Exception as e:
        print(f"🚨 RAG 공통 모델 로드 중 오류 발생: {e}")
        llm, hf_embeddings = None, None

    # 3. 각 RAG 파이프라인 생성
    if llm and hf_embeddings:
        print("\n--- RAG 파이프라인 설정 시작 ---")
        # 3-1. 궁궐 RAG 파이프라인
        try:
            palace_template = """<|user|>
                주어진 [Description] 내용을 바탕으로 '{question}'에 대한 핵심 설명을 1~2 문장으로 요약해 주세요.

                [규칙]
                1. 첫 문장은 반드시 "이곳 {question}은/는..."으로 시작하세요.
                2. [Description]에 명시된 핵심 내용(주요 용도, 의미 등)만을 사용하여 간결하게 설명하세요.
                3. 절대로 [Description]에 없는 내용을 지어내지 마세요.
                4. 외국인이 알아들을 수 있을 정도로 쉽게 설명하세요

                [Description]
                {context}
                <|end|><|assistant|>"""
            
            palace_rag_chain = setup_rag_pipeline(
                data_path='./palace_data.json', db_path='db', content_key='설명_한',
                source_key='건물_한', template=palace_template, llm=llm, hf_embeddings=hf_embeddings
            )
            print("✅ 궁궐 RAG 파이프라인 설정 완료")
            _ = palace_rag_chain.invoke("근정전") # Warm-up
            print("   - 궁궐 RAG Warm-up 완료")
        except Exception as e:
            print(f"🚨 궁궐 RAG 파이프라인 설정 오류: {e}")

        # 3-2. 음식 RAG 파이프라인
        try:
            food_template = """<|user|>
                    당신은 음식 정보를 정확하게 요약하는 가이드입니다. 아래 [Description]에 있는 내용만을 사용하여, 음식을 처음 보는 사람도 쉽게 이해할 수 있도록 핵심 정보를 요약해주세요.

                    [규칙]
                    1. 반드시 "{question}은/는..."으로 문장을 시작하세요.
                    2. 1~2개의 간결한 문장으로 요약하세요.
                    3. **절대로 [Description]에 없는 내용을 상상해서 지어내지 마세요.**
                    4. 주요 재료, 맛, 음식의 종류(예: 인스턴트 식품)를 중심으로 설명하세요.

                    [Description]
                    {context}

                    [Food Name]
                    {question}<|end|><|assistant|>"""
            
            food_rag_chain = setup_rag_pipeline(
                data_path='./food_data.json', db_path='food_db', content_key='설명_한',
                source_key='음식_한', template=food_template, llm=llm, hf_embeddings=hf_embeddings
            )
            print("✅ 음식 RAG 파이프라인 설정 완료")
            _ = food_rag_chain.invoke("Bibimbap") # Warm-up
            print("   - 음식 RAG Warm-up 완료")
        except Exception as e:
            print(f"🚨 음식 RAG 파이프라인 설정 오류: {e}")

    yield
    print("\n--- 서버 종료 ---")

# ------------------- FastAPI 앱 및 라우터 설정 -------------------
app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/api")

# ------------------- API 데이터 모델 -------------------
class AnalysisResponse(BaseModel):
    detected_objects: list[str]

class DescriptionRequest(BaseModel):
    object_name: str = Field(..., description="설명을 요청할 객체의 이름")

class DescriptionResponse(BaseModel):
    description: str
class AnalyzeUrlRequest(BaseModel):
    image_url: str = Field(..., description="분석할 이미지가 있는 S3 Presigned URL")
    type: Literal["palace", "food"] = Field("palace", description="분석할 이미지 종류")
# ------------------- API 엔드포인트 -------------------
@app.get("/")
async def read_root():
    return {"message": "Palace & Food Guide API"}

# @router.post("/analyze", response_model=AnalysisResponse)
# async def analyze_image(
#     file: UploadFile = File(...),
#     # Query를 Form으로 변경하여 form-data에서 값을 받도록 수정
#     type: Literal["palace", "food"] = Form("palace", description="분석할 이미지 종류 ('palace' 또는 'food')")
# ):
#     model_to_use, model_name = (palace_model, "궁궐") if type == "palace" else (food_model, "음식")
    
#     if not model_to_use:
#         raise HTTPException(status_code=503, detail=f"{model_name} 분석 모델이 준비되지 않았습니다.")

#     try:
#         image_bytes = await file.read()
#         img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
#         if img is None:
#             raise HTTPException(status_code=400, detail="이미지 파일을 열 수 없습니다.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {e}")
    
#     results = model_to_use(img)
#     detected_names = list({model_to_use.names[int(box.cls)] for r in results for box in r.boxes})
#     return AnalysisResponse(detected_objects=detected_names)
# 기존 /api/analyze 엔드포인트를 이 코드로 완전히 교체하세요

# @router.post("/analyze")
# async def analyze_image_debug(request: Request):
#     print(f"\n=== 🔍 AI 서버 디버깅 시작 ===")
    
#     # 1. 요청 헤더 확인
#     print("[ 요청 헤더 ]")
#     for header_name, header_value in request.headers.items():
#         if 'content' in header_name.lower():
#             print(f"   {header_name}: {header_value}")
    
#     # 2. 요청 메서드와 URL 확인
#     print(f"   Method: {request.method}")
#     print(f"   URL: {request.url}")
    
#     # 3. Raw body 크기 확인
#     try:
#         body = await request.body()
#         print(f"   Raw Body 크기: {len(body)} bytes")
#         if len(body) < 1000:  # 작은 경우에만 내용 출력
#             print(f"   Raw Body (처음 500자): {body[:500]}")
#     except Exception as e:
#         print(f"   Raw Body 읽기 오류: {e}")
    
#     # 4. Form 데이터 파싱 시도
#     try:
#         # 새 Request 객체 생성 (body는 한 번만 읽을 수 있어서)
#         from fastapi import Request
#         from starlette.requests import Request as StarletteRequest
        
#         # body를 다시 읽기 위해 새로운 요청 생성
#         form_data = await request.form()
        
#         print("[ Form 데이터 분석 ]")
#         if not form_data:
#             print("   >> Form 데이터가 비어있음!")
#         else:
#             print(f"   Form 데이터 키 개수: {len(form_data)}")
#             for key, value in form_data.items():
#                 if key == "file":
#                     print(f"   ✅ 'file' 키 발견!")
#                     print(f"      타입: {type(value)}")
#                     if hasattr(value, 'filename'):
#                         print(f"      파일명: {value.filename}")
#                     if hasattr(value, 'content_type'):
#                         print(f"      Content-Type: {value.content_type}")
#                     if hasattr(value, 'size'):
#                         print(f"      크기: {value.size}")
                    
#                     # 파일 내용 읽기 시도
#                     try:
#                         file_content = await value.read()
#                         print(f"      파일 내용 크기: {len(file_content)} bytes")
#                     except Exception as file_error:
#                         print(f"      파일 내용 읽기 오류: {file_error}")
                        
#                 elif key == "type":
#                     print(f"   ✅ 'type' 키: {value}")
#                 else:
#                     print(f"   기타 키 '{key}': {value} (타입: {type(value)})")
                    
#     except Exception as e:
#         print(f"   🚨 Form 데이터 파싱 오류: {e}")
#         print(f"   오류 타입: {type(e)}")
        
#     print("=== 🔍 디버깅 종료 ===\n")
    
#     # 200 응답으로 디버깅 완료 알림
#     return {"status": "debug_complete", "message": "Check AI server logs for details"}
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image_from_url(request: AnalyzeUrlRequest):
    """S3 URL로부터 이미지를 다운로드하여 객체 탐지를 수행합니다."""
    model_to_use, model_name = (palace_model, "궁궐") if request.type == "palace" else (food_model, "음식")

    if not model_to_use:
        raise HTTPException(status_code=503, detail=f"{model_name} 분석 모델이 준비되지 않았습니다.")

    try:
        # 1. URL에서 이미지 데이터 다운로드
        print(f"이미지 다운로드 시도: {request.image_url}")
        response = requests.get(request.image_url)
        response.raise_for_status()  # HTTP 오류가 있으면 예외 발생
        
        # 2. 다운로드한 바이트 데이터를 이미지로 변환
        image_bytes = response.content
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="URL에서 이미지를 로드할 수 없습니다.")
        print(f"이미지 로드 성공. 크기: {len(image_bytes)} bytes")
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"이미지 URL 다운로드 실패: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {e}")
    
    # 3. YOLO 모델로 분석 수행
    print("YOLO 분석 시작...")
    results = model_to_use(img)
    detected_names = list({model_to_use.names[int(box.cls)] for r in results for box in r.boxes})
    print(f"YOLO 분석 완료: {detected_names}")
    
    return AnalysisResponse(detected_objects=detected_names)
@router.post("/describe", response_model=DescriptionResponse)
async def describe_object(
    request: DescriptionRequest,
    type: Literal["palace", "food"] = Query(..., description="설명할 객체 종류 ('palace' 또는 'food')")
):
    rag_chain_to_use, chain_name = (palace_rag_chain, "궁궐") if type == "palace" else (food_rag_chain, "음식")

    if not rag_chain_to_use:
        raise HTTPException(status_code=503, detail=f"{chain_name} 설명 챗봇이 준비되지 않았습니다.")
        
    try:
        answer = await asyncio.to_thread(rag_chain_to_use.invoke, request.object_name)
        cleaned_answer = answer.strip().split('\n')[0]
        return DescriptionResponse(description=cleaned_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"설명 생성 중 오류 발생: {e}")

app.include_router(router)