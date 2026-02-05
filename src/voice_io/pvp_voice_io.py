from pathlib import Path
import queue
import time
from sound_io.sound_io_contract import SoundIOContract
from voice_io.voice_io_contract import VoiceIOContract
import asyncio
import json
import numpy as np
from typing import Optional, Callable
import threading

import sounddevice as sd
from scipy.signal import resample_poly
from vosk import Model, KaldiRecognizer
import pvporcupine
from piper import PiperVoice

class PicoVoskPiperVoiceIO(VoiceIOContract):
    def __init__(
        self,
        sound_io: SoundIOContract,
        vosk_model_path: str,
        piper_model_path: str,
        wake_word: str = "picovoice",
        sensitivities: float = 1,
        silence_timeout: float = 0,
        first_silence_timeout: float = 0,
    ):
        # ===== CALLBACK =====
        self.on_wake: Optional[Callable[[str], None]] = None
        self.sound_io = sound_io

        # ===== TTS =====
        self.tts = PiperVoice.load(str(next(Path(piper_model_path).glob("*.onnx"))))

        # ===== PICOVOICE =====
        self.porcupine = pvporcupine.create(keywords=[wake_word], sensitivities=[sensitivities])
        self._pp_buffer = np.zeros(0, dtype=np.int16)

        # ===== VOSK =====
        self.vosk_model = Model(vosk_model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, self.porcupine.sample_rate)

        # ===== STATE =====
        self.state = "IDLE"
        self.force_listen = False

        # ===== THREAD SAFE =====
        self._listen_queue: queue.Queue[str] = queue.Queue()
        self._listen_lock = threading.Lock()

        # ===== SILENCE =====
        self.silence_timeout = silence_timeout  # секунды
        self.first_silence_timeout = first_silence_timeout  # секунды
        self._last_voice_ts = 0.0

        self.res_text = ""


    # ================= PUBLIC API =================

    async def start(self):
        """Запуск аудио-цикла и TTS воркера"""
        self.sound_io.run_input(self._audio_callback)


    async def close(self):
        """Закрытие всех ресурсов"""
        self.sound_io.stop_input()

    # ---- SAY ----
    def say(self, text: str) -> None:
        """Fire-and-forget последовательный TTS"""
        print("say:" + text)

        def chunk_gen():
            print(f"Generating audio chunks for: {text}", flush=True)
            for i, chunk in enumerate(self.tts.synthesize(text)):
                audio_chunk = chunk.audio_float_array
                if audio_chunk is None or len(audio_chunk) == 0:
                    continue

                if chunk.sample_rate != self.sound_io.output_sr:
                    audio_chunk = resample_poly(
                        audio_chunk,
                        self.sound_io.output_sr,
                        chunk.sample_rate
                    )

                print(f"chunk {i} generated, {len(audio_chunk)} samples", flush=True)
                yield audio_chunk

        self.sound_io.play_chunks(chunk_gen())

    # ---- LISTEN ----
    async def listen(self) -> str:
        """
        Принудительно включить STT и дождаться текста
        """
        self.force_listen = True
        try:
            r = await asyncio.to_thread(self._listen_queue.get)
            print(f"listened: {r}")
            return r
        finally:
            self.force_listen = False



    # ================= INTERNAL =================

    def _audio_callback(self, audio_chunk: np.ndarray):
        """
        audio_chunk: int16 mono
        """

        if self.porcupine.sample_rate != self.sound_io.input_sr:
            audio_chunk = resample_poly(
                audio_chunk.astype(np.float32),
                self.porcupine.sample_rate,
                self.sound_io.input_sr
            )
            np.clip(audio_chunk, -32768, 32767, out=audio_chunk)
            audio_chunk = audio_chunk.astype(np.int16, copy=False) 

        # ================= IDLE =================
        if self.state == "IDLE":
            if self.force_listen:
                self._start_listening(mode="FORCE")
                return

            self._pp_buffer = np.concatenate([self._pp_buffer, audio_chunk])

            while len(self._pp_buffer) >= self.porcupine.frame_length:
                frame = self._pp_buffer[:self.porcupine.frame_length]
                self._pp_buffer = self._pp_buffer[self.porcupine.frame_length:]

                if self.porcupine.process(frame) >= 0:
                    print("🔥 Wake word detected")
                    self._play_wake_signal()
                    self._start_listening("WAKE")
                    self._pp_buffer = np.zeros(0, dtype=np.int16)
                    return
            return


        # ================= LISTEN =================
        if self.state in ("WAKE_LISTEN", "FORCE_LISTEN"):
            with self._listen_lock:
                if self.recognizer.AcceptWaveform(audio_chunk.tobytes()):
                    rs = json.loads(self.recognizer.Result())["text"]
                    self._last_voice_ts = time.monotonic()

                    if self.res_text:
                        self.res_text = self.res_text + ", " + rs
                    else:
                        self.res_text = rs
                elif json.loads(self.recognizer.PartialResult())["partial"]:
                    self._last_voice_ts = time.monotonic()

            b = time.monotonic() - self._last_voice_ts

            if b > self.silence_timeout:
                print("🔇 Silence detected")
                print(b)
                print(self.silence_timeout)

                if self.res_text:
                    self.res_text = self.res_text[0].upper() + self.res_text[1:] + "."
                print(self.res_text)

                if self.state == "WAKE_LISTEN" and self.on_wake:
                    self.on_wake(self.res_text)
                elif self.state == "FORCE_LISTEN":
                    self._listen_queue.put(self.res_text)

                self._stop_listening()


    def _start_listening(self, mode: str):
        self.recognizer.Reset()
        self._last_voice_ts = time.monotonic() + self.first_silence_timeout 

        if mode == "WAKE":
            self.state = "WAKE_LISTEN"
        else:
            self.state = "FORCE_LISTEN"

        print(f"🎙 Start listening ({self.state})")

    def _stop_listening(self):
        print("⏹ Stop listening")
        self.res_text = ""
        self.state = "IDLE"
        self.force_listen = False


    def _play_wake_signal(self):
        sample_rate = self.sound_io.input_sr   # твой sample_rate
        duration = 0.6

        # Временная ось
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)

        # Две ноты (двухчастотный beep)
        freq1 = 1200.0
        freq2 = 1500.0
        signal = 0.15 * np.sin(2*np.pi*freq1*t) + 0.15 * np.sin(2*np.pi*freq2*t)

        # Плавное экспоненциальное затухание
        decay = np.exp(-5 * t)  # чем выше коэффициент, тем быстрее спад
        signal *= decay

        # Минимальный fade-in/fade-out, чтобы не было щелчков
        fade_len = int(sample_rate * 0.01)  # 10 ms
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
        signal[:fade_len] *= fade_in
        signal[-fade_len:] *= fade_out

        # Если функция принимает Iterable[ndarray]
        chunks = [signal]

        # Вызов функции воспроизведения (замени play_chunks на свою)
        self.sound_io.play_chunks(chunks)
