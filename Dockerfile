FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    alsa-utils \
    libasound2 \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    build-essential \
    wget \
    git \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/. .

CMD ["python3", "-u", "-m", "main"]

