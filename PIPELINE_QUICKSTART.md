# ⚡ Quick Start: Integrated Crawl & Ingest Pipeline

## 🎯 Tính năng

Pipeline tự động **crawl → ingest → cleanup** anime với batch processing:

✅ Crawl 24 tập → Ingest → Xóa video → Lặp lại  
✅ Tiết kiệm dung lượng ổ cứng  
✅ Tự động retry khi thất bại  
✅ Theo dõi tiến độ real-time  

---

## 🚀 3 Bước Sử Dụng

### 1️⃣ Tạo Config

```bash
python scripts/integrated_pipeline.py --create-sample config_crawl.json
```

### 2️⃣ Thêm URLs

Sửa `config_crawl.json`:

```json
{
  "anime": [{
    "anime_id": "jujutsu_kaisen_s2",
    "title": "Jujutsu Kaisen Season 2",
    "title_vietnamese": "Chú Thuật Hồi Chiến Phần 2",
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

### 3️⃣ Chạy Pipeline

```bash
python scripts/integrated_pipeline.py --config config_crawl.json
```

---

## 📊 Output Example

```
🚀 Starting Integrated Crawl & Ingest Pipeline
================================================================================
📺 Processing Anime: Jujutsu Kaisen Season 2
📦 Processing Batch 1 (24 episodes)

🌐 Phase 1/3: Crawling 24 episodes...
✅ Episode 1: Crawled successfully
✅ Episode 2: Crawled successfully
...

💾 Phase 2/3: Ingesting 24 episodes...
✅ Episode 1: Ingested successfully
✅ Episode 2: Ingested successfully
...

🧹 Phase 3/3: Cleaning up 24 videos...
🗑️ Episode 1: Deleted video (450 MB)
🗑️ Episode 2: Deleted video (480 MB)
...
✅ Batch cleanup complete! Freed: 11.2 GB

📊 PIPELINE EXECUTION SUMMARY
Total Episodes: 48
Successfully Ingested: 46
Disk Space Freed: 22.3 GB
Success Rate: 95.8%
```

---

## ⚙️ Options

### Thay đổi batch size:
```bash
python scripts/integrated_pipeline.py --config config.json --batch-size 12
```

### Giữ lại videos (không cleanup):
```bash
python scripts/integrated_pipeline.py --config config.json --no-cleanup
```

### Tăng retry:
```bash
python scripts/integrated_pipeline.py --config config.json --retry 5
```

---

## 🔧 Requirements

```bash
# Cài dependencies
pip install selenium selenium-stealth webdriver-manager

# Cài FFmpeg
# Windows: choco install ffmpeg
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

---

## 💡 Workflow

```
Batch 1 (24 tập):
  Crawl 24 videos → Ingest 24 episodes → Delete 24 videos
  
Batch 2 (24 tập):
  Crawl 24 videos → Ingest 24 episodes → Delete 24 videos
  
Batch N (còn lại):
  Crawl N videos → Ingest N episodes → Delete N videos
```

**Lợi ích:**
- ✅ Chỉ cần dung lượng cho ~24 videos cùng lúc
- ✅ Frames được giữ lại, videos bị xóa
- ✅ Có thể crawl hàng trăm tập mà không lo đầy ổ cứng

---

## 📖 Full Documentation

Xem chi tiết: [INTEGRATED_PIPELINE.md](./INTEGRATED_PIPELINE.md)

---

## 🎯 Example Config

```json
{
  "pipeline_settings": {
    "batch_size": 24,        // Số tập/batch
    "auto_cleanup": true,    // Xóa video sau ingest
    "retry_count": 3         // Số lần retry
  },
  "anime": [
    {
      "anime_id": "anime_001",
      "title": "Anime Title",
      "title_vietnamese": "Tên Tiếng Việt",
      "genres": ["Action"],
      "year": 2023,
      "season": "2023-Summer",
      "episodes": [
        {
          "episode": 1,
          "crawl_url": "https://...",
          "fps": 1.0
        }
      ]
    }
  ]
}
```

---

**Happy Pipeline-ing! 🔄🎌**
