from sound_io.sound_io_contract import SoundIOContract
from typing import Callable, Iterable, Optional

import numpy as np
import sounddevice as sd

class DeviceSoundIO(SoundIOContract):
    def __init__(
        self,
        device_index: int = 3,
        samplerate: int = 44100,
    ):
        self._device = device_index
        self._input_stream = None

        self.samplerate = samplerate

    def run_input(self, callback: Callable[[np.ndarray], None]):

        def _callback(indata, frames, time, status):
            if status:
                print("Audio status:", status)
            callback(indata[:, 0].copy())

        self._input_stream = sd.InputStream(
            device=self._device,
            samplerate=self.samplerate,
            channels=1,
            dtype='float32',
            callback=_callback,
        )

        self._input_stream.start()

    def stop_input(self):
        self._input_stream.stop()

    def play_chunks(self, chunks: Iterable[np.ndarray]):
        with sd.OutputStream(
            device=self._device,
            samplerate=self.samplerate,
            channels=1,
            dtype='float32'
        ) as stream:
            for i, audio_chunk in enumerate(chunks):
                stream.write(audio_chunk.astype(np.float32))
                print(f"chunk {i} played, {len(audio_chunk)} samples", flush=True)