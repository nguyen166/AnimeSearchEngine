# 🚀 Quick Start: Gemini Translation Setup

## ⚡ 3 bước setup nhanh

### 1️⃣ Install Dependencies
```bash
pip install google-generativeai deep-translator
```

### 2️⃣ Get FREE API Key
👉 https://makersuite.google.com/app/apikey

### 3️⃣ Configure .env
```env
TRANSLATION_MODE=GEMINI
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

## ✅ Test It

### Test translation endpoint:
```bash
curl -X POST "http://localhost:8000/translate?text=Luffy sử dụng haki"
```

### Test temporal search với tiếng Việt:
```bash
curl -X POST http://localhost:8000/search/temporal \
  -H "Content-Type: application/json" \
  -d '{
    "current_action": "cảnh nổ lớn",
    "previous_action": "nhân vật tấn công",
    "time_window": 10,
    "top_k": 5
  }'
```

## 📖 Full Documentation
Xem chi tiết: [GEMINI_TRANSLATION.md](./GEMINI_TRANSLATION.md)

---

## 🎯 Features

✅ Auto-detect Vietnamese text  
✅ Smart translation with Gemini  
✅ Preserve anime terminology  
✅ Cache for performance  
✅ Fallback to Google Translate  
✅ Free tier: 60 req/min, 1.5K req/day

## 🔥 Example

**Before:**
```json
{
  "current_action": "explosion scene",
  "previous_action": "character attacks"
}
```

**Now supports:**
```json
{
  "current_action": "cảnh nổ",
  "previous_action": "nhân vật tấn công"
}
```

System tự động dịch và search! 🎌
