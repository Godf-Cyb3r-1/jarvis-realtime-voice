"""Jarvis audio path isolation checks (Windows, no microphone required).

Run: .\\.venv\\Scripts\\python.exe realtime_voice_smoke.py --stage all
Prints no API keys. Tests local speaker, hosted Groq text and the independent
Kaggle B /tts WebSocket. Does not require Kaggle A or microphone permissions.
"""
import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import sys
import time
import wave

import numpy as np
from dotenv import load_dotenv

from hosted_llm import SYSTEM_PROMPT, stream_reply

RATE=16000

def speaker_tone():
    import sounddevice as sd
    print("[audio] Testing Windows output device with a 0.5-second tone ...",flush=True)
    t=np.arange(RATE//2,dtype=np.float32)/RATE
    signal=(0.13*np.sin(2*np.pi*440*t)).astype(np.float32)
    print("[audio] Current defaults:",sd.default.device,flush=True)
    sd.play(signal,RATE)
    sd.wait()
    print("[audio] Test completed. Did you hear the tone?",flush=True)

async def duplex_audio_test():
    """Use the SAME RawStream output path that realtime_jarvis.py uses."""
    from duplex_audio import FullDuplexAudio
    import sounddevice as sd
    loop=asyncio.get_running_loop()
    duration=1.0
    t=np.arange(int(RATE*duration),dtype=np.float32)/RATE
    tone=(np.sin(2*np.pi*500*t)*0.18*32767).astype("<i2")
    print("[duplex] using same RawStream path as live Jarvis",flush=True)
    print("[duplex] PortAudio defaults:",sd.default.device,flush=True)
    print("[duplex] MIC_DEVICE:",os.getenv("MIC_DEVICE","(default)"),
          "SPEAKER_DEVICE:",os.getenv("SPEAKER_DEVICE","(default)"),flush=True)
    with FullDuplexAudio(loop) as audio:
        audio.enqueue(tone)
        await asyncio.sleep(duration+0.4)
        output_time=audio.first_output_time
        print("[duplex] output callback consumed tone:",
              output_time is not None,flush=True)
        if output_time is None:
            raise RuntimeError("Audio callback never consumed speaker PCM")
    print("[duplex] Did you hear a 1-second 500 Hz tone?",flush=True)

async def groq_test():
    from hosted_llm import configuration
    selected,base,model,key=configuration("groq")
    print(f"[groq] provider={selected} model={model} key_configured={bool(key)}",flush=True)
    out=[]
    started=time.monotonic()
    async for part in stream_reply([
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":"Hello Jarvis. Please answer in one friendly sentence."}
    ]):
        if not out:
            print(f"[groq] first text after {1000*(time.monotonic()-started):.0f}ms",flush=True)
        out.append(part)
        if sum(map(len,out))>350:
            break
    print("[groq] reply:",("".join(out).strip() or "<EMPTY>")[:350],flush=True)

async def tts_test(play:bool=True):
    import websockets
    url=os.getenv("TTS_WS_URL","").strip()
    token=os.getenv("TTS_WS_TOKEN","").strip()
    if not url.startswith(("wss://","ws://127.0.0.1:")) or not token:
        raise RuntimeError("Fill TTS_WS_URL and TTS_WS_TOKEN in local Windows .env")
    print("[tts] Connecting to",url.split("://")[0]+"://"+url.split("://")[1].split("/")[0]+"/tts",flush=True)
    turn=99700
    text="Hello. I am Jarvis. Can you hear me speaking now?"
    blocks=[]
    start=time.monotonic()
    first_audio=None
    max_wait=int(os.getenv("TTS_DIAGNOSTIC_TIMEOUT","180"))
    async with websockets.connect(url,additional_headers={
            "Authorization":"Bearer "+token},max_size=16*1024*1024,
            ping_interval=20,ping_timeout=30) as ws:
        hello=json.loads(await asyncio.wait_for(ws.recv(),timeout=20))
        if hello.get("type")!="ready":
            raise RuntimeError("No TTS ready event: "+str(hello)[:300])
        print("[tts] connected; requesting a standalone English phrase",flush=True)
        await ws.send(json.dumps({"type":"tts_speak","turn":turn,
            "segment":0,"language":"en","text":text}))
        while True:
            remaining=max_wait-(time.monotonic()-start)
            if remaining<=0:
                raise TimeoutError("TTS timed out without completing; inspect Kaggle B server logs")
            msg=json.loads(await asyncio.wait_for(ws.recv(),timeout=remaining))
            kind=msg.get("type","")
            if kind=="tts_audio" and int(msg.get("turn",-1))==turn:
                if first_audio is None:
                    first_audio=time.monotonic()
                    print(f"[tts] first PCM after {(first_audio-start)*1000:.0f}ms",flush=True)
                blocks.append(base64.b64decode(msg["pcm16"],validate=True))
            elif kind=="tts_info":
                print("[tts] synthesis_ms=",msg.get("synthesis_ms"),
                      "audio_duration_ms=",msg.get("duration_ms"),flush=True)
            elif kind=="tts_done" and int(msg.get("turn",-1))==turn:
                break
            elif kind=="error":
                raise RuntimeError(f"Kaggle B TTS {msg.get('stage')}: {msg.get('message')}")
    if not blocks:
        raise RuntimeError("TTS job finished with NO PCM AUDIO. Inspect Kaggle B logs")
    raw=b"".join(blocks)
    data=np.frombuffer(raw,dtype="<i2")
    peak=int(np.max(np.abs(data.astype(np.int32)))) if len(data) else 0
    rms=float(np.sqrt(np.mean((data.astype(np.float32)/32768)**2))) if len(data) else 0
    duration=len(data)/RATE
    print(f"[tts] received={duration:.2f}s audio peak={peak} rms={rms:.4f}",flush=True)
    path=Path("jarvis_tts_smoke.wav")
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(raw)
    print("[tts] saved sample:",path.resolve(),flush=True)
    if peak<150 or rms<0.001:
        print("[tts] WARNING: PCM appears silent or extremely quiet",flush=True)
    if play:
        import sounddevice as sd
        print("[audio] Playing received speech on Windows default output device",flush=True)
        sd.play(data.astype(np.float32)/32768.0,RATE)
        sd.wait()
        print("[audio] Playback completed",flush=True)

async def main():
    parser=argparse.ArgumentParser(description="Isolate speaker, Groq and Kaggle B TTS")
    parser.add_argument("--stage",choices=("all","audio","duplex","groq","tts"),default="all")
    a=parser.parse_args()
    load_dotenv(override=True)
    if a.stage in ("all","audio"):
        speaker_tone()
    if a.stage in ("all","duplex"):
        await duplex_audio_test()
    if a.stage in ("all","groq"):
        await groq_test()
    if a.stage in ("all","tts"):
        await tts_test()
    print("[diagnostic] completed",flush=True)

if __name__=="__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[diagnostic FAILED] {type(exc).__name__}: {str(exc)[:450]}",file=sys.stderr,flush=True)
        sys.exit(1)
