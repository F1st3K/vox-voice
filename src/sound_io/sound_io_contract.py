from typing import Iterable, Protocol, Callable, Optional
import numpy as np


class SoundIOContract(Protocol):
    samplerate: int

    def run_input(self, callback: Callable[[np.ndarray], None]) -> None: ...
    def stop_input(self) -> None: ...
    def play_chunks(self, chunks: Iterable[np.ndarray]) -> None: ...