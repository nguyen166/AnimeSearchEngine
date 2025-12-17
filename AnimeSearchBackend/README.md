# Anime Search Engine

Hệ thống tìm kiếm anime bằng hình ảnh sử dụng FastAPI, Milvus, và Elasticsearch.

## 📁 Cấu trúc dự án

```
anime-search-engine/
├── app/
│   ├── __init__.py
│   ├── main.py              # Khởi tạo FastAPI
│   ├── config.py            # Quản lý Environment Variables
│   ├── models/              # Pydantic Models
│   │   ├── __init__.py
│   │   └── schemas.py       # Định nghĩa Input/Output
│   ├── core/                # Database Connections
│   │   ├── __init__.py
│   │   ├── milvus.py        # Milvus Vector DB
│   │   └── elastic.py       # Elasticsearch
│   ├── services/            # Business Logic
│   │   ├── __init__.py
│   │   ├── embedding.py     # AI Model (CLIP)
│   │   └── search.py        # Search Logic
│   └── routers/             # API Endpoints
│       ├── __init__.py
│       └── search.py
├── scripts/                 # Offline Scripts
│   ├── ingest_anime.py      # Video Processing & Data Ingestion
│   └── pipeline_runner.py   # Batch Processing
├── data/                    # Data Storage
│   ├── videos/              # Video files
│   └── frames/              # Extracted frames
├── docker-compose.yml       # Database Services
├── requirements.txt         # Python Dependencies
├── .env                     # Environment Variables
└── README.md
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd anime-search-engine
```

### 2. Tạo virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Khởi động databases

```bash
docker-compose up -d
```

Kiểm tra services:
- Milvus: http://localhost:19530
- Elasticsearch: http://localhost:9200
- Kibana (optional): http://localhost:5601

### 5. Cấu hình environment

Copy file `.env` và điều chỉnh các giá trị nếu cần:

```bash
# Không cần copy, file .env đã có sẵn
# Chỉ cần điều chỉnh các giá trị nếu muốn
```

## 📊 Nạp dữ liệu

### Nạp một video đơn lẻ

```bash
python scripts/ingest_anime.py \
  --video ./data/videos/one_piece_ep001.mp4 \
  --anime-id one_piece_001 \
  --episode 1 \
  --fps 1.0 \
  --title "One Piece" \
  --genres Action Adventure Fantasy \
  --year 1999
```

### Nạp nhiều video với config file

1. Tạo config file mẫu:

```bash
python scripts/pipeline_runner.py --create-sample config.json
```

2. Chỉnh sửa `config.json` với thông tin anime của bạn

3. Chạy pipeline:

```bash
python scripts/pipeline_runner.py --config config.json
```

## 🔥 Chạy API Server

```bash
# Development mode
python app/main.py

# hoặc với uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API sẽ chạy tại: http://localhost:8000

Swagger UI: http://localhost:8000/docs

## 📡 API Endpoints

### 1. Tìm kiếm bằng hình ảnh (POST)

```bash
POST /api/search
Content-Type: application/json

{
  "image_base64": "base64_encoded_image",
  "top_k": 10,
  "filters": {
    "genres": ["Action"],
    "year": 1999
  }
}
```

### 2. Tìm kiếm bằng upload file

```bash
POST /api/search/upload
Content-Type: multipart/form-data

file: <image_file>
top_k: 10
```

### 3. Tìm kiếm bằng text

```bash
POST /api/search
Content-Type: application/json

{
  "text_query": "pirate adventure",
  "top_k": 10
}
```

### 4. Tìm kiếm hybrid (image + text)

```bash
POST /api/search
Content-Type: application/json

{
  "image_base64": "base64_encoded_image",
  "text_query": "pirate adventure",
  "top_k": 10
}
```

### 5. Lấy thông tin anime

```bash
GET /api/anime/{anime_id}
```

### 6. Liệt kê anime

```bash
GET /api/anime?limit=20&offset=0&genre=Action&year=1999
```

### 7. Thống kê hệ thống

```bash
GET /api/stats
```

## 🛠️ Cấu trúc Database

### Milvus (Vector Database)

Collection: `anime_frames`

Fields:
- `id` (VARCHAR): Frame ID
- `anime_id` (VARCHAR): Anime ID
- `episode` (INT32): Số tập
- `timestamp` (FLOAT): Thời điểm trong video
- `embedding` (FLOAT_VECTOR): Vector embedding (512 dims)

### Elasticsearch (Metadata Database)

Index: `anime_metadata`

Fields:
- `anime_id`: ID của anime
- `title`: Tên anime
- `title_english`, `title_japanese`: Tên khác
- `genres`: Thể loại
- `year`: Năm phát hành
- `episodes`: Số tập
- `rating`: Điểm đánh giá
- `description`: Mô tả
- `studio`: Studio sản xuất
- `frames`: Nested array of frame information

## 🧪 Testing

Kiểm tra API với curl:

```bash
# Health check
curl http://localhost:8000/health

# Stats
curl http://localhost:8000/api/stats

# Search (cần có data)
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"text_query": "one piece", "top_k": 5}'
```

## 📝 Notes

- Model mặc định: CLIP ViT-B/32 (512 dims)
- FPS mặc định: 1 frame/giây
- Similarity threshold: 0.5
- Device: CPU (có thể đổi sang CUDA trong .env)

## 🔧 Troubleshooting

### Lỗi kết nối database

Kiểm tra docker containers:
```bash
docker-compose ps
docker-compose logs milvus
docker-compose logs elasticsearch
```

### Lỗi memory

Giảm batch size trong `ingest_anime.py` hoặc tăng memory cho Docker.

### Lỗi model

Nếu không có GPU, đảm bảo `DEVICE=cpu` trong `.env`.

## 📚 Tài liệu tham khảo

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Milvus Documentation](https://milvus.io/docs)
- [Elasticsearch Documentation](https://www.elastic.co/guide/)
- [CLIP Model](https://github.com/openai/CLIP)

## 📄 License

MIT License
