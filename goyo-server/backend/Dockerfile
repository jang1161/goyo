FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치 (pyaudio용)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libportaudio2 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# requirements 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]