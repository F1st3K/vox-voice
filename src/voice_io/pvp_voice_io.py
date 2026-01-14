from voice_io.voice_io_contract import VoiceIOContract
import asyncio
import json
import numpy as np
from typing import Optional, Callable
import threading

import sounddevice as sd
print(sd.query_devices())

from scipy.signal import resample_poly
# Получаем список всех устройств
# devices = sd.query_devices()
# if not devices:
#     print("No audio devices found!")

# # Выбираем default device: если есть стерео аналог
# device_index = None
# for i, d in enumerate(devices):
#     if "Analog" in d["name"] and d["max_output_channels"] > 0:
#         device_index = i
#         break
# if device_index is None:
#     device_index = sd.default.device[1]  # fallback на системный default

# print(f"Using device {device_index}: {devices[device_index]['name']}")
# sd.default.device = device_index
# print("defult divice;")
# sd.default.device = [0, 1]
print(sd.default.device)
from vosk import Model, KaldiRecognizer
import pvporcupine
from pvrecorder import PvRecorder
from piper import PiperVoice

class PicoVoskPiperVoiceIO(VoiceIOContract):
    def __init__(
        self,
        vosk_model_path: str,
        piper_model_path: str,
        wake_word: str,
        sample_rate: int = 16000,
    ):
        # ===== CALLBACK =====
        self.on_wake: Optional[Callable[[str], None]] = None

        # ===== STATE =====
        self.running = False
        self.listening = False

        # ===== ASYNC =====
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.text_queue: asyncio.Queue[str] = asyncio.Queue()

        # ===== LOCK для безопасного использования recognizer =====
        self._listen_lock = threading.Lock()

        # ===== TTS =====
        self.tts = PiperVoice.load(piper_model_path+"/ru_RU-irina-medium.onnx")

        self._tts_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._tts_task: Optional[asyncio.Task] = None

        # ===== PICOVOICE =====
        self.porcupine = pvporcupine.create(keywords=[wake_word])
        self.recorder = PvRecorder(device_index=1, frame_length=self.porcupine.frame_length)

        # ===== VOSK =====
        self.sample_rate = sample_rate
        self.vosk_model = Model(vosk_model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, sample_rate)

    # ================= PUBLIC API =================

    async def start(self):
        """Запуск аудио-цикла и TTS воркера"""
        self.running = True
        self.loop = asyncio.get_running_loop()

        if not self._tts_task:
            self._tts_task = asyncio.create_task(self._tts_worker())
            print("[*] Start tts worker")

        # основной аудио-цикл в отдельном потоке
        print("[*] Start audio loop")
        await asyncio.to_thread(self._audio_loop)

    async def close(self):
        """Закрытие всех ресурсов"""
        self.running = False

        # остановка TTS
        if self._tts_task:
            await self._tts_queue.put(None)
            await self._tts_task
            self._tts_task = None

        self.porcupine.delete()
        self.recorder.stop()

    # ---- SAY ----
    def say(self, text: str) -> None:
        """Fire-and-forget последовательный TTS"""
        print("say:" + text)
        if not self._tts_task and self.loop:
            self._tts_task = self.loop.create_task(self._tts_worker())
        self._tts_queue.put_nowait(text)

    # ---- LISTEN ----
    async def listen(self) -> str:
        """Принудительно включить STT и дождаться текста"""
        print("listenning...")
        self._start_listening()
        text = await self.text_queue.get()
        print("listened: " + text)
        return text

    # ================= INTERNAL =================

    def _audio_loop(self):
        """Wake-word + STT цикл"""
        self.recorder.start()
        try:
            while self.running:
                pcm = self.recorder.read()

                # ---- Wake-word ----
                if self.porcupine.process(pcm) >= 0:
                    # включаем STT
                    self._start_listening()

                # ---- STT ----
                if self.listening:
                    audio_bytes = np.array(pcm, dtype=np.int16).tobytes()
                    with self._listen_lock:  # безопасный доступ к recognizer
                        if self.recognizer.AcceptWaveform(audio_bytes):
                            result = json.loads(self.recognizer.Result())
                            text = result.get("text", "")
                            if text:
                                # on_wake callback с распознанным текстом
                                if self.on_wake and self.loop:
                                    self.loop.call_soon_threadsafe(self.on_wake, text)
                                # текст для listen()
                                if self.loop:
                                    self.loop.call_soon_threadsafe(self.text_queue.put_nowait, text)
                                self._stop_listening()
        finally:
            self.recorder.stop()

    def _start_listening(self):
        with self._listen_lock:
            self.listening = True
            self.recognizer.Reset()

    def _stop_listening(self):
        with self._listen_lock:
            self.listening = False

    # ---- TTS воркер ----
    async def _tts_worker(self):
        while True:
            text = await self._tts_queue.get()
            if text is None:
                break
            await asyncio.to_thread(self._say_blocking, text)
            self._tts_queue.task_done()

    def _say_blocking(self, text: str):
        print("start say:", flush=True)

        # Открываем поток один раз
        with sd.OutputStream(
            device=3,
            samplerate=44100,
            channels=1,
            dtype="float32",
        ) as stream:

            print("decoding & streaming...", flush=True)

            # Генерация по чанкам
            for i, chunk in enumerate(self.tts.synthesize(text)):
                audio_chunk = chunk.audio_float_array
                if audio_chunk is None or len(audio_chunk) == 0:
                    continue

                # resample сразу
                audio_44100 = resample_poly(audio_chunk, 44100, 22050).astype(np.float32)

                # пишем в поток сразу
                stream.write(audio_44100)

                # можно вывести прогресс
                print(f"chunk {i} played, {len(audio_chunk)} samples", flush=True)

        print("done", flush=True)

