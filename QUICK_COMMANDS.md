# ⚡ Quick Setup Commands - AnimeSearchEngine

## 🚀 Setup từ đầu (5 phút)

```powershell
# 1. Clone & Navigate
cd d:\2025-2026\Term1\IR\AnimeSearchEngine\AnimeSearchEngine

# 2. Create Virtual Environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Start Databases (Docker)
docker-compose up -d
Start-Sleep -Seconds 120  # Đợi DB ready

# 5. Create .env file
copy .env.example .env
notepad .env  # Thêm GEMINI_API_KEY

# 6. Create Data Folders
mkdir data, data\videos, data\frames

# 7. Start API Server
python -m uvicorn app.main:app --reload
```

**Done!** API running at http://localhost:8000/docs

---

## 🔧 Daily Commands

### Start Working:
```powershell
cd d:\2025-2026\Term1\IR\AnimeSearchEngine\AnimeSearchEngine
.\venv\Scripts\activate
docker-compose up -d
python -m uvicorn app.main:app --reload
```

### Stop Working:
```powershell
# Ctrl+C to stop API server
docker-compose down
deactivate
```

---

## 📦 Data Ingestion

### Create Config:
```powershell
python scripts/integrated_pipeline.py --create-sample config_crawl.json
```

### Edit config_crawl.json:
```json
{
  "anime": [{
    "anime_id": "anime_001",
    "title": "Anime Title",
    "episodes": [
      {
        "episode": 1,
        "crawl_url": "https://vuighe.cam/anime/tap-1/",
        "fps": 1.0
      }
    ]
  }]
}
```

### Run Pipeline:
```powershell
python scripts/integrated_pipeline.py --config config_crawl.json
```

---

## 🧪 Test Commands

```powershell
# Test health
curl http://localhost:8000/health

# Test translation
curl -X POST "http://localhost:8000/translate?text=xin+chào"

# Test search
curl -X POST http://localhost:8000/search/temporal `
  -H "Content-Type: application/json" `
  -d '{"current_action":"explosion","previous_action":"attack","time_window":10,"top_k":5}'

# Check stats
curl http://localhost:8000/stats
```

---

## 🐛 Fix Common Errors

### "Cannot activate venv":
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Docker not running":
```powershell
# Open Docker Desktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
Start-Sleep -Seconds 30
docker ps
```

### "Port 8000 in use":
```powershell
# Kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### "Milvus connection failed":
```powershell
docker-compose down
docker-compose up -d
Start-Sleep -Seconds 120
```

---

## 📊 Verify Setup

```powershell
# All must return success ✅

# 1. Python version
python --version

# 2. Virtual env active
Get-Command python | Select-Object Source  # Should point to venv

# 3. Docker services
docker ps | findstr "milvus"
docker ps | findstr "elasticsearch"

# 4. API health
curl http://localhost:8000/health

# 5. Database connections
curl http://localhost:19530
curl http://localhost:9200
```

---

## 🔑 Get Gemini API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Login with Google
3. Click "Create API Key"
4. Copy to `.env`:
   ```env
   GEMINI_API_KEY=your_key_here
   ```

---

## 📁 Project Structure

```
AnimeSearchEngine/
├── .env                     # Config (add GEMINI_API_KEY here)
├── docker-compose.yml       # Database services
├── requirements.txt         # Python packages
├── app/                     # Source code
│   ├── main.py             # FastAPI app
│   ├── config.py           # Settings
│   ├── core/               # DB connections
│   ├── services/           # Business logic
│   ├── routers/            # API endpoints
│   └── models/             # Data schemas
├── scripts/                 # Tools
│   ├── integrated_pipeline.py  # Crawl + Ingest
│   ├── ingest_anime.py         # Ingest only
│   └── crawler.py              # Crawl only
├── data/
│   ├── videos/             # Downloaded videos (temp)
│   └── frames/             # Extracted frames (permanent)
└── venv/                   # Virtual environment
```

---

## 🎯 Workflow

```
1. Activate venv → Start Docker → Start API
2. Create crawl config
3. Run pipeline (crawl → ingest → cleanup)
4. Test search via API docs
5. Monitor stats
```

---

## 📚 Full Docs

- **Setup:** [SETUP_GUIDE.md](./SETUP_GUIDE.md)
- **Pipeline:** [INTEGRATED_PIPELINE.md](./INTEGRATED_PIPELINE.md)
- **Translation:** [GEMINI_TRANSLATION.md](./GEMINI_TRANSLATION.md)

---

**Hotline:** Check [SETUP_GUIDE.md](./SETUP_GUIDE.md) for detailed troubleshooting! 🆘
