# 🚀 Hướng Dẫn Vận Hành Anime Search Engine (Full Workflow)

Tài liệu này hướng dẫn chi tiết quy trình khởi chạy hệ thống, từ việc dựng hạ tầng (Docker), thu thập dữ liệu (Crawl), nạp dữ liệu (Ingest) cho đến khi tìm kiếm được trên API.

---

## 📋 Mục lục

1. [Giai đoạn 1: Khởi động Hạ tầng (Infrastructure)](#giai-đoạn-1-khởi-động-hạ-tầng-infrastructure)
2. [Giai đoạn 2: Thu thập Dữ liệu (Data Pipeline)](#giai-đoạn-2-thu-thập-dữ-liệu-data-pipeline)
3. [Giai đoạn 3: Nạp & Xử lý Dữ liệu (Ingestion)](#giai-đoạn-3-nạp--xử-lý-dữ-liệu-ingestion)
4. [Giai đoạn 4: Sử dụng & Tìm kiếm (Serving)](#giai-đoạn-4-sử-dụng--tìm-kiếm-serving)
5. [🛠️ Khắc phục sự cố thường gặp](#️-khắc-phục-sự-cố-thường-gặp)

---

## 🛠️ Giai đoạn 1: Khởi động Hạ tầng (Infrastructure)

Bước này sẽ bật các Container:
- **Milvus** (Vector DB)
- **Elasticsearch** (Text DB)
- **AI Service** (Model)
- **API Gateway** (Backend)

### 1. Chuẩn bị

- Đảm bảo **Docker Desktop** đang chạy.
- Đứng tại thư mục gốc của dự án: `AnimeSearchEngine/`

### 2. Chạy lệnh Docker

Mở Terminal (PowerShell/CMD) và chạy:

```bash
docker-compose up -d --build
```

> ⏱️ Lần đầu chạy sẽ mất **10-15 phút** để tải Image và thư viện.

### 3. Kiểm tra sức khỏe hệ thống (Health Check)

⚠️ **Đây là bước QUAN TRỌNG NHẤT**. Bạn không được sang giai đoạn sau nếu bước này chưa xong.

#### Kiểm tra AI Service:

```bash
docker logs -f anime-embeddings-service
```

- **Chờ đợi:** Bạn sẽ thấy các dòng `Loading model...`
- **Thành công:** Khi thấy dòng `Application startup complete` hoặc `Uvicorn running on http://0.0.0.0:8000`
- Nhấn `Ctrl+C` để thoát xem log

#### Kiểm tra API Gateway:

```bash
docker logs -f anime-api-server
```

- **Thành công:** Khi thấy `✅ Connected to Milvus` và `✅ Connected to Elasticsearch`

---

## 🕷️ Giai đoạn 2: Thu thập Dữ liệu (Data Pipeline)

Các bước này chạy trên máy thật (Localhost), **không chạy trong Docker**.

### 1. Cài đặt môi trường Python (Làm 1 lần)

Mở một Terminal mới, di chuyển vào thư mục pipeline:

```bash
cd data-pipeline

# Tạo môi trường ảo (nếu chưa có)
python -m venv venv

# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate

# Cài đặt thư viện cần thiết
pip install -r requirements.txt
pip install scenedetect[opencv] opencv-python numpy requests python-dotenv pydantic-settings
```

### 2. Tạo file cấu hình Crawl

Sử dụng script helper để tạo config nhanh cho bộ phim bạn muốn (Ví dụ: One Piece).

```bash
# Đứng tại thư mục data-pipeline
python helper/create_config.py
```

- **Nhập URL:** `https://vuighe.cam/one-piece`
- **Nhập tập:** `1` đến `5` (Test ít trước)
- **Kết quả:** Sinh ra file `config_one-piece.json`

### 3. Chạy Crawler

```bash
python run_crawler.py --config config_one-piece.json
```

**Kết quả:** Video `.mp4` và file metadata `.json` sẽ được tải về thư mục `data/raw_videos`

---

## 📥 Giai đoạn 3: Nạp & Xử lý Dữ liệu (Ingestion)

Bước này sẽ cắt video thành ảnh, gửi sang AI Service để lấy Vector và lưu vào Database.

### 1. Chạy lệnh Ingest

⚠️ **Lưu ý:** Phải đứng ở **Thư mục gốc** (`AnimeSearchEngine`) để chạy lệnh này (để Python nhận diện được module `core`).

```bash
# Nếu đang ở data-pipeline thì lùi ra 1 cấp
cd ..

# Chạy lệnh
python data-pipeline/ingest_video.py --dir ./data-pipeline/data/raw_videos
```

### 2. Quan sát quá trình

Script sẽ thực hiện:

1. Đọc video và file JSON đi kèm
2. Phát hiện cảnh (Scene Detection)
3. Lưu ảnh vào `data-pipeline/data/frames/...`
4. Gọi AI Service (qua port 8001) để lấy Vector
5. Lưu Vector vào **Milvus** và Metadata vào **Elasticsearch**

✅ Nếu thấy log chạy liên tục các dòng `Inserted batch...` là thành công.

---

## 🔍 Giai đoạn 4: Sử dụng & Tìm kiếm (Serving)

Lúc này dữ liệu đã sẵn sàng. Bạn có thể gọi API.

### 1. Truy cập Swagger UI

Mở trình duyệt web và vào địa chỉ:

👉 **http://localhost:8000/docs**

### 2. Thử nghiệm API Search

1. Tìm endpoint **POST /search/text** (hoặc `/search` tùy code hiện tại)
2. Nhấn **Try it out**
3. Nhập JSON test:

```json
{
  "text": "Luffy ăn thịt",
  "top_k": 5,
  "mode": "moment"
}
```

4. Nhấn **Execute**
5. Kiểm tra **Response Body**: Bạn sẽ thấy danh sách kết quả kèm `score`, `frame_path` và `url`

---

## 🛠️ Khắc phục sự cố thường gặp

### 1. Lỗi `UnicodeEncodeError: 'charmap' codec...`

**Nguyên nhân:** Windows Terminal không hiển thị được icon cảm xúc.

**Khắc phục:** Chạy lệnh này trước khi chạy python:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### 2. Lỗi Timeout khi chạy Ingest

**Nguyên nhân:** Máy tính xử lý model AI chậm hơn thời gian chờ mặc định (10s).

**Khắc phục:** Mở file `ingest_video.py`, tìm dòng `timeout=10` và sửa thành `timeout=120`.

### 3. Lỗi Connection Refused

**Nguyên nhân:** Docker Container chưa bật hoặc bị tắt.

**Khắc phục:**

1. Gõ `docker ps -a` để xem trạng thái
2. Nếu thấy `Exited`, gõ `docker logs <tên-container>` để xem lỗi
3. Thường do cấu hình GPU sai → Sửa `docker-compose.yml` (bỏ phần deploy GPU) → Chạy lại `docker-compose up -d`

### 4. Lỗi `ModuleNotFoundError: No module named 'core'`

**Nguyên nhân:** Chạy script từ sai thư mục.

**Khắc phục:** Luôn đứng ở thư mục gốc `AnimeSearchEngine/` khi chạy lệnh ingest.

---

## 📝 Ghi chú

- Sau khi setup xong, mỗi lần khởi động lại máy chỉ cần chạy: `docker-compose up -d`
- Để tắt hệ thống: `docker-compose down`
- Để xóa toàn bộ dữ liệu và bắt đầu lại: `docker-compose down -v`