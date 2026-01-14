import asyncio
import os

from dialog.rabbit_dialog import RabbitDialog
from voice_io.pvp_voice_io import PicoVoskPiperVoiceIO
from flow import Flow


# ========================
# ENV
# ========================
RABBIT_URL = os.getenv("RABBIT_URL", "amqp://guest:guest@localhost:5672/")
SOURCE_NAME = os.getenv("SOURCE_NAME", "assistant")
WAKE_WORD = os.getenv("WAKE_WORD", "picovoice")
STT_MODEL_PATH = os.getenv("STT_MODEL_PATH", "/models/stt")
TTS_MODEL_PATH = os.getenv("TTS_MODEL_PATH", "/models/tts")

# =======================k
# MAIN LOOP
# ========================
async def main():
    dialog = RabbitDialog(RABBIT_URL, SOURCE_NAME)
    io = PicoVoskPiperVoiceIO(STT_MODEL_PATH, TTS_MODEL_PATH, WAKE_WORD)

    flow = Flow(dialog, io)
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