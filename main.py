import asyncio
import os
import json
import queue
import threading
import time

from Dialog.RabbitDialog import RabbitDialog
from VoiceIO.PicoVoskCoquiVoiceIO import PicoVoskCoquiVoiceIO
from VoxFlow import VoxFlow
import numpy as np
import sounddevice as sd
import pvporcupine
from vosk import Model, KaldiRecognizer
import pika
import scipy.signal


# ========================
# ENV
# ========================
RABBIT_URL = os.getenv("RABBIT_URL", "amqp://guest:guest@localhost:5672/")
RABBIT_QUEUE = os.getenv("RABBIT_QUEUE", "raw_text")
WAKE_WORD = os.getenv("WAKE_WORD", "picovoice")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "/models/vosk")

# ========================
# RabbitMQ
# ========================
# conn = pika.BlockingConnection(pika.URLParameters(RABBIT_URL))
# channel = conn.channel()
# channel.queue_declare(queue=RABBIT_QUEUE)

# ========================
# Porcupine (wake-word)
# ========================
porcupine = pvporcupine.create(keywords=[WAKE_WORD])
SAMPLE_RATE = porcupine.sample_rate
FRAME_LENGTH = porcupine.frame_length

# ========================
# Vosk
# ========================
print("Loading Vosk model:", VOSK_MODEL_PATH)
vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
recognizer.SetWords(False)

# ========================
# Audio queue
# ========================
audio_q = queue.Queue()

print(sd.query_devices())
print("Default input device:", sd.default.device)

DEVICE_INFO = sd.query_devices(kind="input")
DEVICE_SR = int(DEVICE_INFO["default_samplerate"])
TARGET_SR = porcupine.sample_rate  # 16000

def audio_callback(indata, frames, time_info, status):
    print(f"\rProgress: {abs(indata).mean()}", end="")
    audio = indata[:, 0]

    if DEVICE_SR != TARGET_SR:
        audio = scipy.signal.resample_poly(
            audio,
            TARGET_SR,
            DEVICE_SR
        )

    pcm16 = (audio * 32768).astype(np.int16)
    audio_q.put(pcm16.tobytes())



# ========================
# Worker
# ========================
def process_audio():
    wake_buffer = np.array([], dtype=np.int16)
    listening = True
    last_voice_ts = time.time()

    while True:
        data = audio_q.get()
        if data is None:
            break

        pcm16 = np.frombuffer(data, dtype=np.int16)

        # ---- WAKE WORD ----
        if not listening:
            wake_buffer = np.concatenate([wake_buffer, pcm16])

            for i in range(0, len(wake_buffer) - FRAME_LENGTH, FRAME_LENGTH):
                frame = wake_buffer[i:i + FRAME_LENGTH]
                keyword_index = porcupine.process(frame)
                if keyword_index >= 0:
                    print("🔥 Wake word detected")
                    listening = True
                    recognizer.Reset()
                    wake_buffer = np.array([], dtype=np.int16)
                    break

        # ---- SPEECH ----
        else:
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()

                if text:
                    print("🗣", text)

                    # channel.basic_publish(
                    #     exchange="",
                    #     routing_key=RABBIT_QUEUE,
                    #     body=text.encode()

                listening = True
            else:
                last_voice_ts = time.time()

            # таймаут тишины (3 сек)
            if time.time() - last_voice_ts > 3:
                partial = json.loads(recognizer.FinalResult()).get("text", "")
                if partial:
                    print("🗣", partial)
                listening = True

# ========================
# Start
# ========================
threading.Thread(target=process_audio, daemon=True).start()

with sd.InputStream(
    samplerate=DEVICE_SR,
    channels=1,
    dtype="float32",
    callback=audio_callback,
):
    print("🎧 Listening...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        audio_q.put(None)


async def main():
    dialog = RabbitDialog()
    io = PicoVoskCoquiVoiceIO()

    flow = VoxFlow(dialog, io)
    flow.bind()

    try:
        await dialog.start()
        await io.start()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received, shutting down...")
    except Exception as e:
        print(f"[ERROR] main loop: {e}")
    finally:
        await io.close()
        await dialog.close()

asyncio.run(main())