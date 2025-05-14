# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Clone repo TTS với nhánh add-vietnamese-xtts
RUN rm -rf TTS/ && git clone --branch add-vietnamese-xtts https://github.com/thinhlpg/TTS.git

# Cài đặt các thư viện Python
RUN pip install --upgrade pip
RUN pip install --use-deprecated=legacy-resolver -e TTS
RUN pip install numpy==1.26.4 deepspeed cutlet unidic==1.1.0 underthesea deepfilternet==0.5.6 websockets huggingface_hub aiofiles

# Tải model viXTTS từ HuggingFace
RUN python -m unidic download
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='thinhlpg/viXTTS', repo_type='model', local_dir='model')"

# Chỉ copy requirements.txt để cache install requirements
COPY requirements.txt ./

# Copy code vào sau cùng (để khi sửa code không phải build lại các bước trên)
COPY server.py ./
COPY vixtts_demo.py ./
COPY assets ./assets

# Expose port cho websocket
EXPOSE 8765

# Lệnh mặc định: chạy server websocket
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8765"]