import asyncio
import os

from dialog.rabbit_dialog import RabbitDialog
from sound_io.device_sound_io import DeviceSoundIO
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

SENSITIVITIES = float(os.getenv("SENSITIVITIES", 0.7))
SILENCE_TMOUT = float(os.getenv("SILENCE_TMOUT", 0.8))
FIRST_SILENCE_TMOUT = float(os.getenv("FIRST_SENSITIVITIES_TMOUT", 5))

SOUND_DEVICE = int(os.getenv("SOUND_DEVICE", 3))
SOUND_INPUT_DEVICE = int(os.getenv("SOUND_INPUT_DEVICE", SOUND_DEVICE))
SOUND_OUTPUT_DEVICE = int(os.getenv("SOUND_OUTPUT_DEVICE", SOUND_DEVICE))
INPUT_SAMPLE_RATE = int(os.getenv("INPUT_SAMPLE_RATE", 48000))
OUTPUT_SAMPLE_RATE = int(os.getenv("OUTPUT_SAMPLE_RATE", 44100))


# ========================
# MAIN LOOP
# ========================
async def main():
    dialog = RabbitDialog(
        RABBIT_URL,
        SOURCE_NAME
    )
    sound = DeviceSoundIO(
        input_device_index=SOUND_INPUT_DEVICE,
        output_device_index=SOUND_OUTPUT_DEVICE,
        input_sample_rate=INPUT_SAMPLE_RATE,
        output_sample_rate=OUTPUT_SAMPLE_RATE
    )
    io = PicoVoskPiperVoiceIO(
        sound,
        STT_MODEL_PATH,
        TTS_MODEL_PATH,
        WAKE_WORD,
        SENSITIVITIES,
        SILENCE_TMOUT,
        FIRST_SILENCE_TMOUT
    )

    flow = Flow(dialog, io)
    flow.bind()

    stop_event = asyncio.Event()
    try:
        await dialog.start()
        await io.start()

        print("✅ System started")
        await stop_event.wait()        

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received, shutting down...")
    except Exception as e:
        print(f"[ERROR] main loop: {e}")
    finally:
        await io.close()
        await dialog.close()

asyncio.run(main())