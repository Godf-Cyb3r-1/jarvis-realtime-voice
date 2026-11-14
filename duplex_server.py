"""Duplex WebSocket server: GPU ASR, Smart Turn, streaming LLM and OmniVoice.

The socket receives user audio even while response audio is being sent.
Windows remains responsible for local VAD and optional AEC.
"""
import asyncio
import base64
import hmac
import json
import os
import time
from collections import deque
from typing import AsyncIterator
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from duplex_policy import overlap_action, should_finish, split_ready
from smart_turn_adapter import completion_probability
from tts_engine import VoiceEngine

load_dotenv()
RATE=16000
TOKEN=os.getenv("VOICE_API_TOKEN","")
if not TOKEN:raise RuntimeError("VOICE_API_TOKEN must be set")
ASR=WhisperModel(os.getenv("ASR_MODEL","small"),
    device=os.getenv("ASR_DEVICE","cuda"),
    compute_type=os.getenv("ASR_COMPUTE","float16"))
ASR_GATE=asyncio.Lock()
app=FastAPI()
voices=VoiceEngine()

async def transcribe(samples: np.ndarray) -> str:
    if not len(samples):return ""
    audio=np.asarray(samples,dtype=np.float32)/32768.0
    if np.sqrt(np.mean(audio**2))<0.002:
        return ""
    def run():
        segments,_=ASR.transcribe(audio,beam_size=3,condition_on_previous_text=False,
            vad_filter=False,language=None if os.getenv("ASR_LANGUAGE","auto")=="auto" else os.getenv("ASR_LANGUAGE"))
        return " ".join(s.text.strip() for s in segments).strip()
    async with ASR_GATE:
        return await asyncio.to_thread(run)

async def llm_tokens(messages: list[dict]) -> AsyncIterator[str]:
    base=os.getenv("LLM_BASE_URL","").rstrip("/")
    model=os.getenv("LLM_MODEL","")
    if not base or not model:
        yield "The speech pipeline is connected, but my language model is not configured yet."
        return
    headers={}
    if os.getenv("LLM_API_KEY"):
        headers["Authorization"]="Bearer "+os.environ["LLM_API_KEY"]
    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0,read=80.0)) as client:
        async with client.stream("POST",base+"/chat/completions",headers=headers,
            json={"model":model,"messages":messages[-15:],"stream":True,
                  "temperature":0.65,"max_tokens":400}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):continue
                payload=line[5:].strip()
                if payload=="[DONE]":break
                try:
                    item=json.loads(payload)
                    delta=item["choices"][0].get("delta",{}).get("content","")
                    if isinstance(delta,str) and delta:yield delta
                except (KeyError,IndexError,ValueError):
                    continue

class Session:
    def __init__(self,ws):
        self.ws=ws
        self.lock=asyncio.Lock()
        self.turn=0
        self.mode="IDLE"
        self.samples=bytearray()
        self.overlap=bytearray()
        self.partial_task=None
        self.endpoint_task=None
        self.response_task=None
        self.last_backchannel=0
        self.history=[{"role":"system","content":
            "You are Jarvis, a warm, lively voice companion. Talk naturally in concise "
            "English, Tamil or Tanglish as spoken by the user. Avoid robotic repetitions, "
            "habitual greetings or filler labels. You can respond to a correction directly."}]
        self.t0=time.monotonic()

    async def emit(self,kind,**fields):
        async with self.lock:
            await self.ws.send_json({"type":kind,**fields})

    async def state(self,value):
        self.mode=value
        await self.emit("state",state=value,turn=self.turn)

    async def cancel_response(self):
        self.turn+=1
        if self.response_task:self.response_task.cancel()
        if self.endpoint_task:self.endpoint_task.cancel()
        if self.partial_task:self.partial_task.cancel()
        self.response_task=None
        await self.emit("cancel_audio",turn=self.turn)

    async def begin(self):
        await self.cancel_response()
        self.samples.clear()
        await self.state("LISTEN")

    async def partial(self):
        if self.mode not in ("LISTEN","WAIT") or not self.samples:return
        if self.partial_task and not self.partial_task.done():return
        snap=np.frombuffer(bytes(self.samples[-RATE*2*10:]),dtype="<i2").copy()
        version=self.turn
        async def run():
            try:
                text=await transcribe(snap)
                if version==self.turn and text:
                    await self.emit("partial",text=text,turn=version)
            except Exception as error:
                await self.emit("error",component="asr_partial",message=str(error)[:200])
        self.partial_task=asyncio.create_task(run())

    async def pause(self,silence_ms):
        if self.mode not in ("LISTEN","WAIT") or len(self.samples)<RATE//10*2:return
        if self.endpoint_task and not self.endpoint_task.done():return
        version=self.turn
        snap=np.frombuffer(bytes(self.samples),dtype="<i2").copy()
        async def decide():
            try:
                prob=await asyncio.to_thread(completion_probability,snap.astype(np.float32)/32768.0)
            except Exception as exc:
                prob=None
                await self.emit("warning",component="smart_turn",message=str(exc)[:180])
            if version!=self.turn:return
            if should_finish(silence_ms,prob):
                await self.commit()
            else:
                await self.state("WAIT")
                await self.emit("keep_listening",probability=prob,turn=version)
                if (os.getenv("ALLOW_BACKCHANNEL","0")=="1" and len(snap)>RATE*5
                    and time.monotonic()-self.last_backchannel>9):
                    self.last_backchannel=time.monotonic()
                    async def ack():
                        try:
                            text="ம்." if os.getenv("ASR_LANGUAGE")=="ta" else "Mm-hmm."
                            audio=await voices.synthesize(text,"ta" if text=="ம்." else "en")
                            if version==self.turn and self.mode=="WAIT":
                                await self.send_pcm(audio,version,role="backchannel")
                        except Exception:pass
                    asyncio.create_task(ack())
        self.endpoint_task=asyncio.create_task(decide())

    async def commit(self):
        if self.mode not in ("LISTEN","WAIT"):return
        data=bytes(self.samples)
        self.samples.clear()
        await self.state("THINK")
        version=self.turn
        async def run():
            try:
                transcript=await transcribe(np.frombuffer(data,dtype="<i2"))
                if version!=self.turn:return
                if not transcript:
                    await self.state("IDLE")
                    return
                await self.emit("transcript",text=transcript,turn=version)
                self.history.append({"role":"user","content":transcript})
                await self.respond(version)
            except asyncio.CancelledError:raise
            except Exception as error:
                if version==self.turn:await self.emit("error",component="response",message=str(error)[:300])
                await self.state("IDLE")
        self.response_task=asyncio.create_task(run())

    async def send_pcm(self,audio,version,role="speech"):
        pcm=np.clip(np.asarray(audio,dtype=np.float32).reshape(-1),-1,1)
        pcm=(pcm*32767).astype("<i2")
        for offset in range(0,len(pcm),RATE//8):
            if version!=self.turn:return
            chunk=pcm[offset:offset+RATE//8]
            await self.emit("audio",pcm16=base64.b64encode(chunk.tobytes()).decode(),
                rate=RATE,turn=version,role=role)
            # Backpressure; avoid flooding the client queue minutes ahead.
            await asyncio.sleep(0.02)

    async def respond(self,version):
        await self.state("SPEAK")
        q=asyncio.Queue(maxsize=6)
        fragments=[]
        async def produce():
            buffer=""
            try:
                async for delta in llm_tokens(self.history):
                    if version!=self.turn:return
                    fragments.append(delta)
                    await self.emit("reply_delta",text=delta,turn=version)
                    ready,buffer=split_ready(buffer+delta)
                    for phrase in ready:
                        await q.put(phrase)
                ready,_=split_ready(buffer,force=True)
                for phrase in ready:await q.put(phrase)
            finally:
                await q.put(None)
        async def consume():
            language="en"
            while True:
                phrase=await q.get()
                if phrase is None:break
                if version!=self.turn:return
                if any("\u0b80"<=ch<="\u0bff" for ch in phrase):
                    language="ta"
                pcm=await voices.synthesize(phrase,language)
                if version!=self.turn:return
                await self.send_pcm(pcm,version)
        p=asyncio.create_task(produce())
        c=asyncio.create_task(consume())
        try:
            await asyncio.gather(p,c)
            if version==self.turn:
                reply="".join(fragments).strip()
                if reply:self.history.append({"role":"assistant","content":reply})
                await self.emit("response_done",turn=version)
                await self.state("IDLE")
        finally:
            for task in (p,c):
                if not task.done():task.cancel()
            await asyncio.gather(p,c,return_exceptions=True)

    async def overlap_end(self,duration_ms):
        data=bytes(self.overlap);self.overlap.clear()
        if not data:return
        version=self.turn
        text=await transcribe(np.frombuffer(data,dtype="<i2"))
        if version!=self.turn:return
        action=overlap_action(text,duration_ms)
        await self.emit("overlap_result",text=text,decision=action,turn=version)
        if action=="YIELD":
            await self.begin()
            self.samples.extend(data)
            await self.commit()

@app.get("/health")
async def health():
    return {"ready":True,"asr_device":os.getenv("ASR_DEVICE","cuda"),
            "tts_provider":voices.provider,"smart_turn":os.getenv("SMART_TURN_ENABLED","0")=="1"}

@app.websocket("/ws")
async def duplex_websocket(ws:WebSocket):
    expected="Bearer "+TOKEN
    supplied=ws.headers.get("authorization","")
    if not hmac.compare_digest(supplied,expected):
        await ws.close(code=1008)
        return
    await ws.accept()
    s=Session(ws)
    await s.emit("ready",rate=RATE,frame_ms=20,version=2)
    try:
        async for msg in ws.iter_text():
            try:
                event=json.loads(msg)
                kind=event.get("type")
                if kind=="start":
                    await s.begin()
                elif kind=="audio" and s.mode in ("LISTEN","WAIT"):
                    block=base64.b64decode(event["pcm16"],validate=True)
                    if len(block)<=6400 and len(block)%2==0:
                        s.samples.extend(block)
                        if len(s.samples)>RATE*2*45:s.samples=s.samples[-RATE*2*45:]
                elif kind=="partial":
                    await s.partial()
                elif kind=="pause":
                    await s.pause(max(0,min(int(event.get("silence_ms",0)),15000)))
                elif kind=="interrupt":
                    await s.cancel_response()
                    await s.state("IDLE")
                elif kind=="overlap_start":
                    s.overlap.clear()
                elif kind=="overlap_audio":
                    block=base64.b64decode(event["pcm16"],validate=True)
                    if len(block)<=6400:s.overlap.extend(block)
                    if len(s.overlap)>RATE*2*8:s.overlap=s.overlap[-RATE*2*8:]
                elif kind=="overlap_end":
                    await s.overlap_end(int(event.get("duration_ms",0)))
            except (ValueError,KeyError) as exc:
                await s.emit("warning",component="protocol",message=str(exc)[:150])
    except WebSocketDisconnect:
        pass
    finally:
        for task in (s.partial_task,s.endpoint_task,s.response_task):
            if task:task.cancel()
