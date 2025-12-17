# 🚀 Hướng dẫn Setup Từ Đầu - AnimeSearchEngine

## 📋 Mục lục
1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Cài đặt môi trường](#cài-đặt-môi-trường)
3. [Khởi động Database](#khởi-động-database)
4. [Cấu hình Project](#cấu-hình-project)
5. [Test & Verify](#test--verify)
6. [Nạp dữ liệu](#nạp-dữ-liệu)
7. [Chạy API Server](#chạy-api-server)
8. [Troubleshooting](#troubleshooting)

---

## 📌 Yêu cầu hệ thống

### Software cần thiết:
- ✅ **Python 3.9+** ([Download](https://www.python.org/downloads/))
- ✅ **Docker Desktop** ([Download](https://www.docker.com/products/docker-desktop/))
- ✅ **Git** ([Download](https://git-scm.com/downloads))
- ✅ **FFmpeg** (cho video processing)
  - Windows: `choco install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

### Hardware khuyến nghị:
- RAM: 8GB+ (16GB recommended)
- Disk: 50GB+ free space
- CPU: 4+ cores
- GPU: Optional (CUDA-compatible cho faster inference)

---

## 🛠️ Cài đặt môi trường

### Bước 1: Clone project

```bash
# Clone repository
git clone <your-repo-url>
cd AnimeSearchEngine

# Hoặc nếu đã có folder
cd d:\2025-2026\Term1\IR\AnimeSearchEngine
```

### Bước 2: Tạo Virtual Environment

```powershell
# Windows PowerShell
cd AnimeSearchEngine
python -m venv venv
.\venv\Scripts\activate

# Verify activation (should see (venv) prefix)
```

**Lưu ý:** Nếu gặp lỗi execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Bước 3: Cài đặt Python Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installations
pip list | findstr "fastapi torch transformers"
```

**Expected packages:**
- fastapi==0.109.0
- torch==2.1.2
- transformers==4.37.0
- google-generativeai>=0.3.0
- deep-translator==1.11.4
- pymilvus==2.3.5
- elasticsearch==8.11.1
- selenium, selenium-stealth (cho crawler)

---

## 🐳 Khởi động Database

### Bước 1: Check Docker

```powershell
# Verify Docker is running
docker --version
docker ps

# Nếu Docker chưa chạy, mở Docker Desktop
```

### Bước 2: Start Database Services

```powershell
# Start Milvus + Elasticsearch
docker-compose up -d

# Check logs
docker-compose logs -f

# Wait until you see:
# - milvus-standalone: "Server started successfully"
# - elasticsearch: "started"
```

### Bước 3: Verify Services

```powershell
# Test Milvus (should return version info)
curl http://localhost:9091/healthz

# Test Elasticsearch (should return cluster info)
curl http://localhost:9200

# Test Kibana (optional, UI for Elasticsearch)
# Browser: http://localhost:5601
```

**Lưu ý:** Lần đầu start có thể mất 2-5 phút để các services khởi động hoàn toàn.

---

## ⚙️ Cấu hình Project

### Bước 1: Tạo file .env

```powershell
# Copy từ template
copy .env.example .env

# Hoặc tạo mới
notepad .env
```

### Bước 2: Sửa .env với nội dung:

```env
# FastAPI Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=anime_frames
VECTOR_DIM=512

# Elasticsearch Configuration
ELASTIC_HOST=localhost
ELASTIC_PORT=9200
ELASTIC_INDEX=anime_metadata

# AI Model Configuration
MODEL_NAME=clip-vit-base-patch32
DEVICE=cpu

# Translation Configuration
TRANSLATION_MODE=GEMINI
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash

# Data Paths
DATA_DIR=./data
VIDEO_DIR=./data/videos
FRAME_DIR=./data/frames
```

### Bước 3: Lấy Gemini API Key (Miễn phí)

1. Truy cập: https://makersuite.google.com/app/apikey
2. Đăng nhập Google Account
3. Click "Create API Key"
4. Copy key và paste vào `.env` file

**Quota miễn phí:** 60 req/min, 1,500 req/day

### Bước 4: Tạo thư mục data

```powershell
# Tạo folders
mkdir data
mkdir data\videos
mkdir data\frames

# Verify structure
tree data /F
```

---

## ✅ Test & Verify

### Test 1: Check Python Imports

```powershell
python -c "import fastapi, torch, transformers; print('✅ Imports OK')"
```

### Test 2: Check Database Connections

```powershell
# Test script
python test_milvus.py
```

**Expected output:**
```
✅ Milvus: True
✅ Elasticsearch: True
```

### Test 3: Check Translation Service

```powershell
python -c "
from app.services.translation import translation_service
result = translation_service.translate('Xin chào')
print('✅ Translation:', result)
"
```

---

## 📦 Nạp dữ liệu

### Option 1: Integrated Pipeline (Crawl + Ingest)

**Bước 1:** Tạo config file

```powershell
python scripts/integrated_pipeline.py --create-sample config_crawl.json
```

**Bước 2:** Sửa config (thêm URLs)

```json
{
  "pipeline_settings": {
    "batch_size": 24,
    "auto_cleanup": true,
    "retry_count": 3
  },
  "anime": [{
    "anime_id": "jujutsu_kaisen_s2",
    "title": "Jujutsu Kaisen Season 2",
    "title_vietnamese": "Chú Thuật Hồi Chiến Phần 2",
    "genres": ["Action", "Supernatural"],
    "year": 2023,
    "season": "2023-Summer",
    "episodes": [
      {
        "episode": 1,
        "crawl_url": "https://vuighe.cam/chu-thuat-hoi-chien-phan-2/tap-1/",
        "fps": 1.0
      },
      {
        "episode": 2,
        "crawl_url": "https://vuighe.cam/chu-thuat-hoi-chien-phan-2/tap-2/",
        "fps": 1.0
      }
    ]
  }]
}
```

**Bước 3:** Chạy pipeline

```powershell
python scripts/integrated_pipeline.py --config config_crawl.json
```

**Output:**
```
🚀 Starting Integrated Crawl & Ingest Pipeline
📺 Processing Anime: Jujutsu Kaisen Season 2
🌐 Phase 1/3: Crawling episodes...
✅ Episode 1: Crawled successfully
💾 Phase 2/3: Ingesting episodes...
✅ Episode 1: Ingested successfully
🧹 Phase 3/3: Cleaning up videos...
✅ Batch cleanup complete! Freed: 0.5 GB
```

### Option 2: Manual Ingest (Video có sẵn)

```powershell
python scripts/ingest_anime.py `
  --video "./data/videos/anime_ep01.mp4" `
  --anime-id "anime_001" `
  --episode 1 `
  --fps 1.0 `
  --title "Anime Title" `
  --genres Action Adventure `
  --year 2023 `
  --season "2023-Spring"
```

---

## 🌐 Chạy API Server

### Bước 1: Start FastAPI Server

```powershell
# Development mode
cd AnimeSearchEngine
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Bước 2: Verify API

**Mở browser:**
- API Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

### Bước 3: Test Endpoints

**Test search:**
```powershell
curl -X POST "http://localhost:8000/search" `
  -H "Content-Type: application/json" `
  -d '{
    "text_query": "explosion scene",
    "top_k": 5
  }'
```

**Test temporal search:**
```powershell
curl -X POST "http://localhost:8000/search/temporal" `
  -H "Content-Type: application/json" `
  -d '{
    "current_action": "cảnh nổ",
    "previous_action": "nhân vật tấn công",
    "time_window": 10,
    "top_k": 5
  }'
```

**Test translation:**
```powershell
curl -X POST "http://localhost:8000/translate?text=Luffy+sử+dụng+haki"
```

**Check stats:**
```powershell
curl http://localhost:8000/stats
```

---

## 🐛 Troubleshooting

### Lỗi: "Cannot connect to Milvus"

**Nguyên nhân:** Docker services chưa sẵn sàng

**Giải pháp:**
```powershell
# Restart services
docker-compose down
docker-compose up -d

# Wait 2-3 minutes
Start-Sleep -Seconds 120

# Check logs
docker-compose logs milvus-standalone
```

### Lỗi: "ModuleNotFoundError"

**Nguyên nhân:** Dependencies chưa cài đúng

**Giải pháp:**
```powershell
# Activate venv
.\venv\Scripts\activate

# Reinstall
pip install -r requirements.txt --force-reinstall
```

### Lỗi: "FFmpeg not found"

**Giải pháp:**
```powershell
# Windows (Chocolatey)
choco install ffmpeg

# Verify
ffmpeg -version
```

### Lỗi: "GEMINI_API_KEY not found"

**Giải pháp:**
1. Check `.env` file có tồn tại
2. Verify API key từ https://makersuite.google.com/app/apikey
3. Restart server sau khi update `.env`

### Lỗi: "Port 8000 already in use"

**Giải pháp:**
```powershell
# Tìm process đang dùng port
netstat -ano | findstr :8000

# Kill process
taskkill /PID <PID> /F

# Hoặc dùng port khác
python -m uvicorn app.main:app --port 8001
```

### Lỗi: "Docker compose not found"

**Giải pháp:**
```powershell
# Check Docker version
docker --version
docker compose version

# Update Docker Desktop nếu cần
# Download: https://www.docker.com/products/docker-desktop/
```

---

## 📊 Verify Complete Setup

### Checklist:

```powershell
# 1. Virtual environment
python --version  # Should show Python 3.9+

# 2. Dependencies
pip list | findstr "fastapi"

# 3. Docker services
docker ps | findstr "milvus"
docker ps | findstr "elasticsearch"

# 4. .env file
Test-Path .env  # Should be True

# 5. Data folders
Test-Path data\videos  # Should be True
Test-Path data\frames  # Should be True

# 6. API server
curl http://localhost:8000/health

# 7. Database connections
curl http://localhost:19530
curl http://localhost:9200
```

**Tất cả phải PASS!** ✅

---

## 📚 Next Steps

Sau khi setup xong, bạn có thể:

1. **Crawl & Ingest Data:**
   ```powershell
   python scripts/integrated_pipeline.py --config config_crawl.json
   ```

2. **Test Search:**
   - Truy cập http://localhost:8000/docs
   - Thử các endpoints

3. **Monitor:**
   - Check Kibana: http://localhost:5601
   - Check stats: http://localhost:8000/stats

---

## 🎯 Quick Commands Reference

```powershell
# Activate environment
.\venv\Scripts\activate

# Start databases
docker-compose up -d

# Start API server
python -m uvicorn app.main:app --reload

# Create crawl config
python scripts/integrated_pipeline.py --create-sample config.json

# Run pipeline
python scripts/integrated_pipeline.py --config config.json

# Stop databases
docker-compose down

# Clean data
Remove-Item data\videos\* -Force
Remove-Item data\frames\* -Recurse -Force
```

---

## 📖 Documentation Links

- [API Documentation](http://localhost:8000/docs)
- [Integrated Pipeline Guide](./INTEGRATED_PIPELINE.md)
- [Gemini Translation Guide](./GEMINI_TRANSLATION.md)
- [Quick Start Guide](./PIPELINE_QUICKSTART.md)

---

**Chúc bạn setup thành công! 🎉**

Nếu gặp vấn đề, hãy check [Troubleshooting](#troubleshooting) section hoặc mở issue trên GitHub.
