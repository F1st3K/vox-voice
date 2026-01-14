from typing import Protocol, Callable, Optional


class VoiceIOContract(Protocol):
    on_wake: Optional[Callable[[str], None]]

    def say(self, text: str) -> None: ...
    async def listen(self) -> str: ...