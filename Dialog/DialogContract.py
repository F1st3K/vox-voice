from typing import Awaitable, Protocol, Callable, Optional


class DialogContract(Protocol):
    on_say: Optional[Callable[[str], None]]
    on_ask: Optional[Callable[[str], Awaitable[str]]]

    def pub_input(self, text: str) -> None: ...
