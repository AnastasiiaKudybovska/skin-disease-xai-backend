import uvicorn
from fastapi import FastAPI
from app.routes import classify, xai
from fastapi.middleware.cors import CORSMiddleware
from app.config import origins

app = FastAPI(
    title="Skin Disease Classifier with XAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify.router, prefix="/api/classify", tags=["Classification"])
app.include_router(xai.router, prefix="/api/xai", tags=["XAI"])

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)