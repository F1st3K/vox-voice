import asyncio
import os

from Dialog.RabbitDialog import RabbitDialog
from VoiceIO.PicoVoskCoquiVoiceIO import PicoVoskCoquiVoiceIO
from VoxFlow import VoxFlow


# ========================
# ENV
# ========================
RABBIT_URL = os.getenv("RABBIT_URL", "amqp://guest:guest@localhost:5672/")
SOURCE_NAME = os.getenv("SOURCE_NAME", "assistant")
WAKE_WORD = os.getenv("WAKE_WORD", "picovoice")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "/models/vosk")

# ========================
# MAIN LOOP
# ========================
async def main():
    dialog = RabbitDialog(RABBIT_URL, SOURCE_NAME)
    io = PicoVoskCoquiVoiceIO(VOSK_MODEL_PATH, WAKE_WORD)

    flow = VoxFlow(dialog, io)
    flow.bind()

    try:
        await dialog.start()
        await io.start()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received, shutting down...")
    except Exception as e:
        print(f"[ERROR] main loop: {e}")
    finally:
        await io.close()
        await dialog.close()

asyncio.run(main())