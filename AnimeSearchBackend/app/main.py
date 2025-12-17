"""
FastAPI Main Application
Khởi tạo FastAPI app và cấu hình routes
"""

import warnings
# Suppress pkg_resources deprecation warning from pymilvus
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import search
from app.config import settings

# Khởi tạo FastAPI app
app = FastAPI(
    title="Anime Search Engine",
    description="API tìm kiếm anime bằng hình ảnh và metadata",
    version="1.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn origins cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router, prefix="/api", tags=["search"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Anime Search Engine API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái hệ thống"""
    return {
        "status": "healthy",
        "milvus": "connected",  # TODO: Thêm logic kiểm tra thực tế
        "elasticsearch": "connected"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
