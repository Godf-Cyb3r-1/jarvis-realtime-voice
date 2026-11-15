"""Validate the actual Jarvis Windows duplex sound output and microphone without Kaggle.

A short, gentle 440Hz test tone plays through the duplex audio callback.
Use headphones. This test makes no network calls.
"""
import asyncio
import numpy as np
import sounddevice as sd
from duplex_audio import FullDuplexAudio,RATE

async def main():
    print("Default input/output:",sd.default.device)
    print("If you cannot hear this tone, inspect Windows output device/volume.")
    t=np.arange(RATE,dtype=np.float32)/RATE
    tone=(np.sin(2*np.pi*440*t)*0.12*32767).astype("<i2")
    loop=asyncio.get_running_loop()
    audio=FullDuplexAudio(loop)
    with audio:
        audio.enqueue(tone)
        received=0
        end=loop.time()+1.7
        while loop.time()<end:
            try:
                block=await asyncio.wait_for(audio.mic.get(),timeout=0.5)
                received+=len(block)
            except asyncio.TimeoutError:
                pass
    print("Microphone bytes captured:",received)
    print("PASS: microphone and playback ran together" if received>0 else
          "FAIL: no microphone data arrived")

if __name__=="__main__":
    asyncio.run(main())
