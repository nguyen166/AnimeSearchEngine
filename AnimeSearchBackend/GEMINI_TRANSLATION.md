# Google Gemini Translation Integration

## 🎯 Tổng quan

Dự án AnimeSearchEngine đã được tích hợp **Google Gemini API** để cải thiện khả năng dịch thuật các thuật ngữ Anime từ tiếng Việt sang tiếng Anh.

### ✨ Tính năng chính:
- **3 chế độ dịch thuật**: GEMINI, ONLINE (Google Translate), LOCAL (HuggingFace)
- **Auto-detection**: Tự động phát hiện và dịch tiếng Việt trong Temporal Search
- **Smart caching**: Cache kết quả dịch để tối ưu performance
- **Fallback mechanism**: Tự động chuyển về phương thức khác nếu lỗi

---

## 🚀 Cài đặt

### 1. Cập nhật Dependencies

```bash
pip install -r requirements.txt
```

Dependencies mới:
- `google-generativeai>=0.3.0` - Gemini API
- `deep-translator==1.11.4` - Fallback translator

### 2. Lấy Gemini API Key (MIỄN PHÍ)

1. Truy cập: https://makersuite.google.com/app/apikey
2. Đăng nhập bằng Google Account
3. Nhấn "Create API Key"
4. Copy API key

**Quota miễn phí:**
- 60 requests/phút
- 1,500 requests/ngày
- Hoàn toàn đủ cho development và testing

### 3. Cấu hình Environment Variables

Tạo file `.env` từ `.env.example`:

```bash
cp .env.example .env
```

Sửa các biến sau trong `.env`:

```env
# Translation Configuration
TRANSLATION_MODE=GEMINI

# Google Gemini API
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-1.5-flash  # Nhanh và miễn phí
```

**Lựa chọn model:**
- `gemini-1.5-flash` - ⚡ Nhanh nhất, khuyến nghị cho production
- `gemini-pro` - 🎯 Chính xác hơn, chậm hơn

---

## 📖 Cách sử dụng

### 1. Temporal Search với Auto-Translation

Khi gửi query tiếng Việt, hệ thống tự động dịch sang tiếng Anh:

```python
# Request
POST /search/temporal
{
    "current_action": "cảnh nổ lớn",
    "previous_action": "nhân vật rút kiếm",
    "time_window": 10,
    "top_k": 10
}

# System tự động:
# 1. Phát hiện tiếng Việt
# 2. Dịch: "cảnh nổ lớn" → "big explosion"
# 3. Dịch: "nhân vật rút kiếm" → "character draws sword"
# 4. Thực hiện search với text đã dịch
```

### 2. Test Translation Endpoint

Để test dịch thuật riêng lẻ:

```bash
# curl
curl -X POST "http://localhost:8000/translate?text=Nhân vật rút kiếm"

# Response
{
    "success": true,
    "original": "Nhân vật rút kiếm",
    "translated": "Character draws sword",
    "mode": "GEMINI"
}
```

### 3. Kiểm tra Translation Stats

```bash
curl http://localhost:8000/stats

# Response
{
    "success": true,
    "milvus": {...},
    "elasticsearch": {...},
    "translation": {
        "mode": "GEMINI",
        "cache_size": 15,
        "model_info": {
            "gemini_model": "gemini-1.5-flash",
            "device": null
        }
    }
}
```

---

## 🔧 Chế độ Translation

### GEMINI Mode (Khuyến nghị)

**Ưu điểm:**
- ✅ Hiểu ngữ cảnh tốt nhất (LLM)
- ✅ Giữ nguyên thuật ngữ Anime ("Haki", "Bankai", "Chakra")
- ✅ Miễn phí trong quota
- ✅ Prompt engineering tối ưu

**Cấu hình:**
```env
TRANSLATION_MODE=GEMINI
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-1.5-flash
```

### ONLINE Mode (Fallback)

**Ưu điểm:**
- ✅ Không cần API key
- ✅ Nhanh
- ⚠️ Có thể bị block khi spam requests

**Cấu hình:**
```env
TRANSLATION_MODE=ONLINE
```

### LOCAL Mode (Offline)

**Ưu điểm:**
- ✅ Hoàn toàn offline
- ⚠️ Cần download model (~300MB)
- ⚠️ Chậm hơn

**Cấu hình:**
```env
TRANSLATION_MODE=LOCAL
DEVICE=cpu  # or cuda
```

---

## 🎨 Use Cases

### 1. Tìm kiếm chuỗi hành động bằng tiếng Việt

```json
POST /search/temporal
{
    "current_action": "vụ nổ",
    "previous_action": "tấn công",
    "time_window": 5
}
```

### 2. Temporal search với filters

```json
POST /search/temporal
{
    "current_action": "chiến đấu quyết liệt",
    "previous_action": "nhân vật biến hình",
    "time_window": 15,
    "top_k": 20,
    "filters": {
        "anime_id": "one_piece",
        "genres": ["Action"]
    }
}
```

### 3. Dịch thuật standalone

```python
from app.services.translation import translation_service

# Dịch text
result = translation_service.translate("Luffy sử dụng Gear 5")
# Output: "Luffy uses Gear 5"

# Xem stats
stats = translation_service.get_stats()

# Clear cache
translation_service.clear_cache()
```

---

## 🐛 Troubleshooting

### Lỗi: "GEMINI_API_KEY is required"

**Nguyên nhân:** Chưa set API key

**Giải pháp:**
```bash
export GEMINI_API_KEY=your_key_here
# hoặc thêm vào .env
```

### Lỗi: "Quota exceeded"

**Nguyên nhân:** Vượt quota miễn phí (60 req/min)

**Giải pháp:**
1. System tự động fallback về ONLINE mode
2. Hoặc chuyển về ONLINE mode thủ công:
```env
TRANSLATION_MODE=ONLINE
```

### Lỗi: "Failed to initialize Gemini"

**Nguyên nhân:** API key không hợp lệ hoặc mạng lỗi

**Giải pháp:**
1. Kiểm tra API key: https://makersuite.google.com/app/apikey
2. Kiểm tra kết nối internet
3. Xem logs để debug

---

## 📊 Performance

### Benchmark (gemini-1.5-flash):

| Operation          | Latency | Throughput |
| ------------------ | ------- | ---------- |
| Single translation | ~0.5s   | 2 req/s    |
| With cache         | ~0.001s | 1000 req/s |
| Temporal search    | ~1.2s   | 0.8 req/s  |

### Caching Strategy:

- ✅ Cache translations trong memory
- ✅ TTL: 3600s (configurable)
- ✅ LRU-based eviction

---

## 🔐 Security Notes

⚠️ **Quan trọng:**

1. **Không commit API key** vào Git:
   ```bash
   # .env đã có trong .gitignore
   echo ".env" >> .gitignore
   ```

2. **Rotate API key định kỳ** trên Google AI Studio

3. **Monitor usage** để tránh vượt quota

---

## 📚 Tài liệu tham khảo

- [Google Gemini API Docs](https://ai.google.dev/docs)
- [Get API Key](https://makersuite.google.com/app/apikey)
- [Pricing & Quotas](https://ai.google.dev/pricing)

---

## 🎉 Example Complete Workflow

```python
import asyncio
from app.services.search import SearchService
from app.models.schemas import TemporalSearchRequest

# 1. Tạo request với tiếng Việt
request = TemporalSearchRequest(
    current_action="cảnh chiến đấu",
    previous_action="nhân vật biến hình",
    time_window=10,
    top_k=5
)

# 2. Thực hiện search (tự động dịch)
response = await SearchService.search_temporal(request)

# 3. Xem kết quả
for pair in response.pairs:
    print(f"Score: {pair.combined_score:.2f}")
    print(f"Sequence: {pair.sequence_context}")
    print(f"Anime: {pair.current_frame.anime_title}")
    print(f"Episode: {pair.current_frame.episode}")
    print("---")
```

---

**Happy Searching! 🎌🔍**
