# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# --- THAY ĐỔI QUAN TRỌNG Ở ĐÂY ---

# 1. Copy requirements.txt VÀO TRƯỚC
COPY requirements.txt .

# 2. Cài đặt thư viện ngay lập tức (Tận dụng Cache)
# (Thêm --no-cache-dir để giảm dung lượng image)
RUN pip install --no-cache-dir -r requirements.txt

# 3. (Tùy chọn) Copy và chạy install.sh nếu nó làm việc khác ngoài pip install
COPY install.sh ./
RUN chmod +x install.sh && ./install.sh

# 4. Cuối cùng mới copy source code còn lại
COPY . .

# ---------------------------------

# Expose the port the app runs on
EXPOSE 8000

# Set the default command
CMD ["python", "apis.py"]