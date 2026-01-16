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
        wake_word: str
    ):
        # ===== CALLBACK =====
        self.on_wake: Optional[Callable[[str], None]] = None
        self.sound_io = sound_io

        # ===== TTS =====
        self.tts = PiperVoice.load(piper_model_path+"/ru_RU-irina-medium.onnx")

        # ===== PICOVOICE =====
        self.porcupine = pvporcupine.create(keywords=[wake_word])
        self._pp_buffer = np.zeros(0, dtype=np.int16)

        # ===== VOSK =====
        self.vosk_model = Model(vosk_model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, 16000)

        # ===== STATE =====
        self.state = "IDLE"
        self.force_listen = False

        # ===== THREAD SAFE =====
        self._listen_queue: queue.Queue[str] = queue.Queue()
        self._listen_lock = threading.Lock()

        # ===== SILENCE =====
        self.silence_timeout = 10.0  # секунды
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

                audio_convert = resample_poly(audio_chunk, self.sound_io.samplerate, 22050).astype(np.float32)

                print(f"chunk {i} generated, {len(audio_chunk)} samples", flush=True)
                yield audio_convert

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
        audio_chunk: float32 mono
        """

        pcm_i16 = np.clip(
            resample_poly(
                audio_chunk,
                16000,
                self.sound_io.samplerate
            ) * 32768,
            -32768,
            32767
        ).astype(np.int16)

        # ================= IDLE =================
        if self.state == "IDLE":
            if self.force_listen:
                self._start_listening(mode="FORCE")
                return

            self._pp_buffer = np.concatenate([self._pp_buffer, pcm_i16])

            while len(self._pp_buffer) >= self.porcupine.frame_length:
                frame = self._pp_buffer[:self.porcupine.frame_length]
                self._pp_buffer = self._pp_buffer[self.porcupine.frame_length:]

                if self.porcupine.process(frame) >= 0:
                    print("🔥 Wake word detected")
                    self._start_listening("WAKE")
                    self._pp_buffer = np.zeros(0, dtype=np.int16)
                    return
            return


        # ================= LISTEN =================
        if self.state in ("WAKE_LISTEN", "FORCE_LISTEN"):
            with self._listen_lock:
                rs = ""
                if self.recognizer.AcceptWaveform(pcm_i16.tobytes()):
                    rs = json.loads(self.recognizer.Result())["text"]
                if rs != "":
                    self.res_text = self.res_text + ". " + rs
                    self._last_voice_ts = time.monotonic()

            b = time.monotonic() - self._last_voice_ts

            if b > self.silence_timeout:
                print("🔇 Silence detected")
                print(b)
                print(self.silence_timeout)
                print(self.res_text)

                if self.state == "WAKE_LISTEN" and self.on_wake:
                    self.on_wake(self.res_text)
                elif self.state == "FORCE_LISTEN":
                    self._listen_queue.put(self.res_text)

                self._stop_listening()


    def _start_listening(self, mode: str):
        self.recognizer.Reset()
        self._last_voice_ts = time.monotonic()

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


