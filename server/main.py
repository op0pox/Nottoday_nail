import uvicorn
from fastapi import FastAPI
from api.api import router as measure_router

# 전체 애플리케이션 객체 생성
app = FastAPI(title="Nail Measurement API")

# 분리된 라우터를 메인 앱에 등록
app.include_router(measure_router)

if __name__ == "__main__":
    # main.py 안에 app이 있으므로 main:app 으로 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)