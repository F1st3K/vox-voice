from VoiceIO.VoiceIOContract import VoiceIOContract
import asyncio
import json
import numpy as np
from typing import Optional, Callable
import threading

import sounddevice as sd
from vosk import Model, KaldiRecognizer
from TTS.api import TTS
import pvporcupine
from pvrecorder import PvRecorder


class PicoVoskCoquiVoiceIO(VoiceIOContract):
    def __init__(
        self,
        vosk_model_path: str,
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
        self.tts = TTS(
            model_name="tts_models/en/ljspeech/tacotron2-DDC",
            progress_bar=False,
            gpu=False,
        )
        self._tts_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._tts_task: Optional[asyncio.Task] = None

        # ===== PICOVOICE =====
        self.porcupine = pvporcupine.create(keywords=[wake_word])
        self.recorder = PvRecorder(device_index=-1, frame_length=self.porcupine.frame_length)

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

        # основной аудио-цикл в отдельном потоке
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
        if not self._tts_task and self.loop:
            self._tts_task = self.loop.create_task(self._tts_worker())
        self._tts_queue.put_nowait(text)

    # ---- LISTEN ----
    async def listen(self) -> str:
        """Принудительно включить STT и дождаться текста"""
        self._start_listening()
        text = await self.text_queue.get()
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
        """Блокирующая генерация и воспроизведение речи через Coqui"""
        wav = self.tts.tts(text)
        sd.play(wav, samplerate=22050)
        sd.wait()
