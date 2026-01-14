from dialog.dialog_contract import DialogContract
from voice_io.voice_io_contract import VoiceIOContract


class Flow:

    def __init__(self, dialog: DialogContract, io: VoiceIOContract):
        self.dialog = dialog
        self.io = io

    def bind(self):
        self.io.on_wake = lambda text: self.dialog.pub_input(text)

        async def ask(text):
            self.io.say(text)
            return await self.io.listen()
        self.dialog.on_ask = ask
        
        self.dialog.on_say = lambda text: self.io.say(text)
