# 🔄 Integrated Crawl & Ingest Pipeline

## 📖 Tổng quan

Pipeline tích hợp **crawl** và **ingest** anime với tính năng **auto-cleanup** để tiết kiệm dung lượng ổ cứng.

### ✨ Tính năng chính:

- ✅ **Batch Processing**: Xử lý theo lô (mặc định 24 tập/batch)
- ✅ **Auto Crawl**: Tự động tải video từ nguồn
- ✅ **Auto Ingest**: Tự động extract frames và index vào database
- ✅ **Auto Cleanup**: Tự động xóa video sau khi ingest xong
- ✅ **Retry Mechanism**: Tự động retry khi crawl/ingest thất bại
- ✅ **Statistics Tracking**: Theo dõi tiến độ và thống kê

---

## 🚀 Quick Start

### 1. Tạo Config File

```bash
python scripts/integrated_pipeline.py --create-sample config_crawl.json
```

### 2. Sửa Config File

Mở `config_crawl.json` và thêm URL anime cần crawl:

```json
{
  "pipeline_settings": {
    "batch_size": 24,
    "auto_cleanup": true,
    "keep_frames": true,
    "retry_count": 3,
    "delay_between_episodes": 2
  },
  "anime": [
    {
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
    }
  ]
}
```

### 3. Chạy Pipeline

```bash
python scripts/integrated_pipeline.py --config config_crawl.json
```

---

## ⚙️ Pipeline Settings

### Cấu hình trong JSON:

| Setting                  | Mô tả                          | Mặc định |
| ------------------------ | ------------------------------ | -------- |
| `batch_size`             | Số tập xử lý trước khi cleanup | 24       |
| `auto_cleanup`           | Tự động xóa video sau ingest   | true     |
| `keep_frames`            | Giữ lại frames đã extract      | true     |
| `retry_count`            | Số lần retry khi thất bại      | 3        |
| `delay_between_episodes` | Delay giữa các tập (giây)      | 2        |

### Command Line Options:

```bash
# Thay đổi batch size
python scripts/integrated_pipeline.py --config config.json --batch-size 12

# Tắt auto cleanup (giữ lại video)
python scripts/integrated_pipeline.py --config config.json --no-cleanup

# Thay đổi retry count
python scripts/integrated_pipeline.py --config config.json --retry 5
```

---

## 🔄 Workflow Pipeline

### Phase 1: Crawl Batch (24 tập)
```
📥 Episode 1: Downloading...
📥 Episode 2: Downloading...
...
📥 Episode 24: Downloading...
```

### Phase 2: Ingest Batch
```
💾 Episode 1: Extracting frames → Generating embeddings → Indexing...
💾 Episode 2: Extracting frames → Generating embeddings → Indexing...
...
💾 Episode 24: Extracting frames → Generating embeddings → Indexing...
```

### Phase 3: Cleanup Batch
```
🗑️  Episode 1: Deleted video (450 MB)
🗑️  Episode 2: Deleted video (480 MB)
...
🗑️  Episode 24: Deleted video (470 MB)
✅ Batch cleanup complete! Freed: 11.2 GB
```

### Repeat cho batch tiếp theo...

---

## 📊 Statistics Tracking

Pipeline tự động theo dõi:

```
📊 PIPELINE EXECUTION SUMMARY
================================================================================
Total Anime Processed: 2
Total Episodes: 48
Successfully Crawled: 46
Successfully Ingested: 46
Failed: 2
Videos Cleaned Up: 46
Disk Space Freed: 22.3 GB
Success Rate: 95.8%
================================================================================
```

---

## 🎯 Use Cases

### Case 1: Crawl toàn bộ anime

```json
{
  "anime": [
    {
      "anime_id": "anime_001",
      "title": "Anime Title",
      "episodes": [
        {"episode": 1, "crawl_url": "https://..."},
        {"episode": 2, "crawl_url": "https://..."},
        ...
        {"episode": 100, "crawl_url": "https://..."}
      ]
    }
  ]
}
```

**Kết quả:**
- Xử lý 24 tập đầu → cleanup
- Xử lý 24 tập tiếp → cleanup
- ...
- Xử lý 4 tập cuối → cleanup

### Case 2: Mix crawl và local videos

```json
{
  "episodes": [
    {"episode": 1, "crawl_url": "https://..."},
    {"episode": 2, "video_path": "./data/videos/ep002.mp4"},
    {"episode": 3, "crawl_url": "https://..."}
  ]
}
```

Pipeline sẽ:
- Crawl episode 1
- Skip episode 2 (đã có video)
- Crawl episode 3
- Ingest cả 3
- Cleanup chỉ episode 1 và 3

### Case 3: Large scale crawling

```bash
# Tăng batch size để xử lý nhanh hơn
python scripts/integrated_pipeline.py \
  --config large_anime.json \
  --batch-size 50 \
  --retry 5
```

---

## 🛠️ Troubleshooting

### Lỗi: "FFmpeg not found"

**Giải pháp:**
```bash
# Windows (Chocolatey)
choco install ffmpeg

# Linux
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Lỗi: "Selenium driver error"

**Giải pháp:**
```bash
# Cài ChromeDriver
pip install selenium webdriver-manager

# Hoặc download manual: https://chromedriver.chromium.org/
```

### Lỗi: "Out of disk space"

**Giải pháp:**
- Giảm batch_size: `--batch-size 12`
- Hoặc cleanup manual:
```bash
rm -rf data/videos/*.mp4
```

### Lỗi: "Crawl failed after 3 retries"

**Nguyên nhân:**
- Website đổi cấu trúc
- Bị block IP
- Video bị xóa

**Giải pháp:**
1. Kiểm tra URL còn hoạt động không
2. Tăng retry: `--retry 5`
3. Thêm delay: Sửa `delay_between_episodes` trong config

---

## 🔐 Best Practices

### 1. Backup Frames

Frames đã extract rất quý giá. **Luôn backup** trước khi cleanup:

```bash
# Backup frames
cp -r data/frames data/frames_backup

# Hoặc compress
tar -czf frames_backup.tar.gz data/frames
```

### 2. Monitor Disk Space

```bash
# Check disk usage
du -sh data/videos data/frames

# Auto cleanup khi disk đầy
python scripts/integrated_pipeline.py --config config.json --batch-size 12
```

### 3. Schedule Pipeline

Chạy tự động bằng cron (Linux/macOS):

```bash
# Cron job: Chạy mỗi ngày lúc 2 AM
0 2 * * * cd /path/to/project && python scripts/integrated_pipeline.py --config auto_crawl.json
```

Windows Task Scheduler:
```powershell
# Tạo scheduled task
schtasks /create /tn "AnimeCrawl" /tr "python scripts/integrated_pipeline.py --config auto_crawl.json" /sc daily /st 02:00
```

---

## 📈 Performance Tips

### Tối ưu Crawl Speed

```json
{
  "pipeline_settings": {
    "batch_size": 24,
    "delay_between_episodes": 1  // Giảm delay
  }
}
```

### Tối ưu Ingest Speed

```python
# Trong config
{
  "episodes": [
    {"episode": 1, "fps": 0.5}  // Giảm FPS = ít frames hơn
  ]
}
```

### Parallel Processing (Advanced)

Chạy multiple pipelines song song:

```bash
# Terminal 1: Anime 1-5
python scripts/integrated_pipeline.py --config anime_batch1.json

# Terminal 2: Anime 6-10
python scripts/integrated_pipeline.py --config anime_batch2.json
```

---

## 🎓 Example Complete Workflow

```bash
# Step 1: Tạo config
python scripts/integrated_pipeline.py --create-sample my_anime.json

# Step 2: Sửa config (thêm URLs)
nano my_anime.json

# Step 3: Test với 1 anime trước
# (Giảm episodes xuống chỉ 2-3 tập để test)

# Step 4: Chạy pipeline
python scripts/integrated_pipeline.py --config my_anime.json

# Step 5: Monitor progress
tail -f logs/pipeline.log

# Step 6: Check results
curl http://localhost:8000/stats
```

---

## 💡 Tips & Tricks

### Tip 1: Resume Failed Episodes

Nếu pipeline bị gián đoạn, chỉ cần chạy lại với cùng config. Pipeline sẽ:
- Skip các tập đã ingest
- Retry các tập failed
- Tiếp tục từ tập tiếp theo

### Tip 2: Selective Cleanup

```json
{
  "pipeline_settings": {
    "auto_cleanup": false  // Tắt auto cleanup
  }
}
```

Sau đó cleanup manual:
```bash
# Xóa chỉ videos đã ingest thành công
python scripts/cleanup_videos.py --status ingested
```

### Tip 3: Quality Control

Kiểm tra frames trước khi cleanup videos:

```bash
# Check một số frames random
ls -lh data/frames/anime_001/ep01/ | head -10

# Nếu OK → Cleanup manual
rm data/videos/anime_001_*.mp4
```

---

## 📚 Related Documentation

- [Crawler Documentation](./CRAWLER.md)
- [Ingest Documentation](./INGEST.md)
- [Gemini Translation](./GEMINI_TRANSLATION.md)

---

**Happy Crawling! 🎌📥🗑️**
