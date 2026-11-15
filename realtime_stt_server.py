"""Kaggle A: dedicated continuous audio ingestion + rolling Whisper recognition.

Designed for independent GPU session (no OmniVoice or LLM loaded).
WebSocket /stt accepts 16kHz mono signed-int16 frames even while Jarvis speaks.
Partial hypotheses use LocalAgreement-2; never present them as final.
"""
import asyncio
import base64
import hmac
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

from asr_quality import transcribe_audio
from realtime_protocol import Hypothesis, MAX_AUDIO_BYTES, RATE

TOKEN=os.getenv("STT_WS_TOKEN", "")
if len(TOKEN)<24:
    raise RuntimeError("Set STT_WS_TOKEN (24+ random characters) via Kaggle Secret")
MODEL_NAME=os.getenv("STT_MODEL","medium")
DEVICE=os.getenv("STT_DEVICE","cuda")
model=WhisperModel(MODEL_NAME,device=DEVICE,
                   device_index=int(os.getenv("STT_GPU_INDEX","0")),
                   compute_type=os.getenv("STT_COMPUTE","int8_float16"),
                   num_workers=1)
model_gate=threading.Lock()
app=FastAPI(title="Jarvis STT Ear")

@dataclass
class Utterance:
    turn: int
    samples: bytearray = field(default_factory=bytearray)
    hypothesis: Hypothesis = field(default_factory=Hypothesis)
    last_partial: float = 0.0
    partial_task: asyncio.Task | None = None
    started: float = field(default_factory=time.monotonic)

def run_recognition(pcm:bytes):
    x=np.frombuffer(pcm,dtype="<i2").copy()
    with model_gate:
        return transcribe_audio(model,x)

@app.get("/health")
async def health():
    return {"ready":True,"service":"stt","model":MODEL_NAME,"device":DEVICE,
            "sample_rate":RATE,"partial_agreement":2}

@app.websocket("/stt")
async def socket(ws:WebSocket):
    supplied=ws.headers.get("authorization","")
    if not hmac.compare_digest(supplied,"Bearer "+TOKEN):
        await ws.close(code=1008)
        return
    await ws.accept()
    send_lock=asyncio.Lock()
    tasks=set()
    current=None
    last_turn=-1

    async def send(kind,**fields):
        async with send_lock:
            await ws.send_json({"type":kind,**fields})

    def launch(coro):
        task=asyncio.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return task

    async def partial(u:Utterance,snapshot:bytes):
        try:
            text,language=await asyncio.to_thread(run_recognition,snapshot)
            if u is not current or not text:
                return
            stable,changed=u.hypothesis.observe(text)
            await send("partial",turn=u.turn,revision=u.hypothesis.revision,
                       stable=stable,raw=text,language=language,
                       stable_changed=changed)
        except Exception as exc:
            await send("error",stage="partial_asr",message=str(exc)[:250],turn=u.turn)

    async def finalize(u:Utterance,snapshot:bytes):
        try:
            t0=time.monotonic()
            text,language=await asyncio.to_thread(run_recognition,snapshot)
            await send("final",turn=u.turn,text=text,language=language,
                       asr_ms=round((time.monotonic()-t0)*1000),
                       audio_ms=round(len(snapshot)/(RATE*2)*1000))
        except Exception as exc:
            await send("error",stage="final_asr",message=str(exc)[:250],turn=u.turn)
    await send("ready",service="stt",rate=RATE,frame_ms=20)

    try:
        async for msg in ws.iter_json():
            kind=msg.get("type")
            if kind=="start":
                turn=int(msg.get("turn",-1))
                if turn<=last_turn:
                    await send("error",stage="protocol",message="Non-monotonic turn id",turn=turn)
                    continue
                last_turn=turn
                current=Utterance(turn)
                await send("listening",turn=turn)
            elif kind=="audio":
                if current is None or int(msg.get("turn",-1))!=current.turn:
                    continue
                try:
                    data=base64.b64decode(msg["pcm16"],validate=True)
                except (ValueError,KeyError):
                    continue
                if not data or len(data)>MAX_AUDIO_BYTES or len(data)%2:
                    continue
                current.samples.extend(data)
                if len(current.samples)>RATE*2*35:
                    current.samples=current.samples[-RATE*2*35:]
                now=time.monotonic()
                if (len(current.samples)>=RATE*2*0.9
                        and now-current.last_partial >= float(os.getenv("STT_PARTIAL_INTERVAL","0.55"))
                        and (current.partial_task is None or current.partial_task.done())):
                    current.last_partial=now
                    window=int(float(os.getenv("STT_PARTIAL_WINDOW_SECS","12"))*RATE*2)
                    snap=bytes(current.samples[-max(window,RATE*2):])
                    current.partial_task=launch(partial(current,snap))
            elif kind=="pause_probe":
                if current is None or int(msg.get("turn",-1))!=current.turn:
                    continue
                # Optional audio-native Smart Turn; no LLM or TTS on this GPU.
                async def decide(u=current,snapshot=bytes(current.samples)):
                    prob=None
                    if os.getenv("SMART_TURN_ENABLED","0")=="1":
                        try:
                            from smart_turn_adapter import completion_probability
                            prob=await asyncio.to_thread(
                                completion_probability,
                                np.frombuffer(snapshot[-RATE*2*8:],dtype="<i2").astype(np.float32)/32768)
                        except Exception as exc:
                            await send("warning",stage="smart_turn",message=str(exc)[:160])
                    if u is current:
                        await send("turn_probability",turn=u.turn,probability=prob)
                launch(decide())
            elif kind=="end":
                if current is None or int(msg.get("turn",-1))!=current.turn:
                    continue
                u=current
                current=None
                launch(finalize(u,bytes(u.samples)))
            elif kind=="ping":
                await send("pong",time_ms=int(time.monotonic()*1000))
    except WebSocketDisconnect:
        pass
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks,return_exceptions=True)
