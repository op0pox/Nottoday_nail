import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.api import router as measure_router

# FastAPI 객체 생성
app = FastAPI(title="Nail Measurement API")

# cors문제 해결
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 프론트엔드 주소만 넣는 것이 좋지만 개발 중에는 "*"로 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# api라우터
app.include_router(measure_router)

if __name__ == "__main__":
    # main.py 안에 app이 있으므로 main:app 으로 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)