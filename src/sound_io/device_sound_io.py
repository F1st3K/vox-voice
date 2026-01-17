from sound_io.sound_io_contract import SoundIOContract
from typing import Callable, Iterable, Optional

import numpy as np
import sounddevice as sd

class DeviceSoundIO(SoundIOContract):
    def __init__(
        self,
        input_device_index: int = 3,
        output_device_index: int = 3,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 22050,
    ):
        self._input_device = input_device_index
        self._output_device = output_device_index
        self._input_stream = None

        self.input_sr = input_sample_rate
        self.output_sr = output_sample_rate

    def run_input(self, callback: Callable[[np.ndarray], None]):

        def _callback(indata, frames, time, status):
            if status:
                print("Audio status:", status)
            callback(indata[:, 0].copy())

        self._input_stream = sd.InputStream(
            device=self._input_device,
            samplerate=self.input_sr,
            channels=1,
            dtype='int16',
            callback=_callback,
        )

        self._input_stream.start()

    def stop_input(self):
        if (self._input_stream != None):
            self._input_stream.close()

    def play_chunks(self, chunks: Iterable[np.ndarray]):
        with sd.OutputStream(
            device=self._output_device,
            samplerate=self.output_sr,
            channels=1,
            dtype='float32'
        ) as stream:
            for i, audio_chunk in enumerate(chunks):
                stream.write(audio_chunk)
                print(f"chunk {i} played, {len(audio_chunk)} samples", flush=True)