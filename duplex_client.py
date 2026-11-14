"""Windows listening+speaking: capture, playback, overlap/backchannel and barge-in.

Start with headphones. Optional AEC_ENABLED=1 uses pywebrtc-audio.
"""
import asyncio
import base64
import json
import os
import time
import numpy as np
import webrtcvad
import websockets
from dotenv import load_dotenv
from duplex_audio import FullDuplexAudio, RATE

load_dotenv()
FRAME_MS=20
FRAME=RATE*FRAME_MS//1000

async def main():
    uri=os.getenv("VOICE_WS_URL","").strip()
    token=os.getenv("VOICE_API_TOKEN","").strip()
    if not uri or not token:
        raise RuntimeError("Set VOICE_WS_URL and VOICE_API_TOKEN in .env")
    if not (uri.startswith("wss://") or uri.startswith("ws://127.0.0.1:")):
        raise RuntimeError("Use WSS outside localhost")
    vad=webrtcvad.Vad(int(os.getenv("VAD_MODE","2")))
    async with websockets.connect(uri,additional_headers={"Authorization":"Bearer "+token},
            max_size=16*1024*1024,ping_interval=20,ping_timeout=20) as ws:
        loop=asyncio.get_running_loop()
        audio=FullDuplexAudio(loop)
        turn=0
        local_mode="IDLE"
        remote_state="IDLE"
        last_partial=""
        async def receive():
            nonlocal turn,local_mode,remote_state,last_partial
            async for raw in ws:
                event=json.loads(raw)
                kind=event.get("type")
                if kind=="ready":
                    print("[ready] Duplex transport connected. Speak naturally.")
                elif kind=="cancel_audio":
                    turn=int(event["turn"])
                    audio.flush()
                elif kind=="state":
                    remote_state=event["state"]
                    if remote_state in ("THINK","SPEAK","IDLE") and local_mode=="USER":
                        local_mode="IDLE"
                elif kind=="audio":
                    if int(event.get("turn",-1))==turn:
                        pcm=np.frombuffer(base64.b64decode(event["pcm16"]),dtype="<i2").copy()
                        audio.enqueue(pcm)
                elif kind in ("partial","transcript"):
                    text=event.get("text","")
                    if text and (kind!="partial" or text!=last_partial):
                        print(f"[{kind}] {text}")
                    if kind=="partial":last_partial=text
                elif kind=="reply_delta":
                    print(event.get("text",""),end="",flush=True)
                elif kind=="response_done":
                    print()
                elif kind=="overlap_result":
                    print(f"\n[overlap: {event['decision']}] {event.get('text','')}")
                    audio.restore()
                elif kind in ("warning","error"):
                    print(f"\n[{kind}:{event.get('component','')}] {event.get('message','')}")
        reader=asyncio.create_task(receive())
        last_partial_at=time.monotonic()
        last_pause_at=0.0
        voice_count=0
        trailing_silence=0
        preroll=[]
        overlap_frames=[]
        overlap_started=0.0
        try:
            with audio:
                while True:
                    frame=await audio.mic.get()
                    active=vad.is_speech(frame,RATE)
                    now=time.monotonic()
                    if local_mode=="IDLE":
                        preroll=(preroll+[frame])[-15:]
                        voice_count=voice_count+1 if active else max(0,voice_count-1)
                        if voice_count<3:continue
                        voice_count=0
                        trailing_silence=0
                        last_partial_at=now
                        last_pause_at=now
                        if audio.playing:
                            local_mode="OVERLAP"
                            overlap_started=now
                            overlap_frames=list(preroll)
                            audio.duck(float(os.getenv("OVERLAP_DUCK_GAIN","0.35")))
                            await ws.send(json.dumps({"type":"overlap_start"}))
                            for part in overlap_frames:
                                await ws.send(json.dumps({"type":"overlap_audio","pcm16":base64.b64encode(part).decode()}))
                        else:
                            local_mode="USER"
                            await ws.send(json.dumps({"type":"start"}))
                            for part in preroll:
                                await ws.send(json.dumps({"type":"audio","pcm16":base64.b64encode(part).decode()}))
                        preroll=[]
                    elif local_mode=="OVERLAP":
                        overlap_frames.append(frame)
                        trailing_silence=0 if active else trailing_silence+FRAME_MS
                        await ws.send(json.dumps({"type":"overlap_audio","pcm16":base64.b64encode(frame).decode()}))
                        elapsed=int((now-overlap_started)*1000)
                        if elapsed>=int(os.getenv("FORCE_BARGE_MS","1100")) and active:
                            audio.flush()
                            await ws.send(json.dumps({"type":"interrupt"}))
                            await ws.send(json.dumps({"type":"start"}))
                            for part in overlap_frames:
                                await ws.send(json.dumps({"type":"audio","pcm16":base64.b64encode(part).decode()}))
                            overlap_frames=[]
                            local_mode="USER"
                            trailing_silence=0
                        elif trailing_silence>=260:
                            await ws.send(json.dumps({"type":"overlap_end","duration_ms":elapsed}))
                            audio.restore()
                            overlap_frames=[]
                            local_mode="IDLE"
                            preroll=[]
                    elif local_mode=="USER":
                        await ws.send(json.dumps({"type":"audio","pcm16":base64.b64encode(frame).decode()}))
                        trailing_silence=0 if active else trailing_silence+FRAME_MS
                        if now-last_partial_at>=float(os.getenv("PARTIAL_INTERVAL","0.8")):
                            await ws.send(json.dumps({"type":"partial"}))
                            last_partial_at=now
                        if trailing_silence>=300 and now-last_pause_at>=0.32:
                            await ws.send(json.dumps({"type":"pause","silence_ms":trailing_silence}))
                            last_pause_at=now
        finally:
            reader.cancel()
            await asyncio.gather(reader,return_exceptions=True)
            audio.flush()

if __name__=="__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nJarvis stopped.")
