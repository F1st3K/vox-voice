FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libasound2-dev \
    portaudio19-dev \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./main.py .

CMD ["python", "main.py"]

