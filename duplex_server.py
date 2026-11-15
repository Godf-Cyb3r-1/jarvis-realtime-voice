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
import threading
import traceback
from collections import deque
from typing import AsyncIterator
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException
from fastapi.responses import Response
from faster_whisper import WhisperModel
from duplex_policy import overlap_action, should_finish, split_ready
from smart_turn_adapter import completion_probability
from tts_engine import VoiceEngine
from tts_longform import chunk_text, combine_pcm
from hosted_llm import SYSTEM_PROMPT, stream_reply
from asr_quality import transcribe_audio
from gpu_guard import inventory, index_for

load_dotenv()
RATE=16000
TOKEN=os.getenv("VOICE_API_TOKEN","")
if not TOKEN:raise RuntimeError("VOICE_API_TOKEN must be set")
ASR_DEVICE=os.getenv("ASR_DEVICE","cuda")
ASR_INDEX=index_for("ASR_GPU_INDEX",0,require_ctranslate=True) if ASR_DEVICE=="cuda" else 0
print(f"[startup] ASR={os.getenv('ASR_MODEL','small')} device={ASR_DEVICE}:{ASR_INDEX} "
      f"compute={os.getenv('ASR_COMPUTE','int8_float16')}",flush=True)
ASR=WhisperModel(os.getenv("ASR_MODEL","small"),
    device=ASR_DEVICE,device_index=ASR_INDEX,
    compute_type=os.getenv("ASR_COMPUTE","int8_float16"),num_workers=1)
# The lock must live IN the worker thread. Cancelling asyncio.to_thread
# does not terminate the underlying GPU inference.
ASR_THREAD_GATE=threading.Lock()
app=FastAPI()
voices=VoiceEngine()

async def transcribe(samples: np.ndarray) -> str:
    if not len(samples):
        return ""
    # A thread lock survives asyncio cancellation; old and new GPU ASR cannot
    # run simultaneously after a quick user interruption.
    def run():
        with ASR_THREAD_GATE:
            text, language = transcribe_audio(ASR, samples)
            if text:
                print(f"[asr] language={language} text={text[:100]!r}", flush=True)
            return text
    return await asyncio.to_thread(run)


async def llm_tokens(messages: list[dict]) -> AsyncIterator[str]:
    if os.getenv("LLM_BACKEND", "groq") == "local":
        # Explicit legacy choice only; not used for the low-latency baseline.
        from local_llm import stream_local
        async for part in stream_local(messages):
            yield part
    else:
        async for part in stream_reply(messages):
            yield part


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
        self.history=[{"role":"system","content":SYSTEM_PROMPT}]
        self.t0=time.monotonic()
        self.committed_at=self.t0
        self.first_audio=False

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
        self.first_audio=False
        self.t0=time.monotonic()
        self.samples.clear()
        await self.state("LISTEN")

    async def partial(self):
        if self.mode not in ("LISTEN","WAIT") or not self.samples:return
        if os.getenv("PARTIAL_ASR_ENABLED","1")!="1":return
        if self.partial_task and not self.partial_task.done():return
        snap=np.frombuffer(bytes(self.samples[-RATE*2*4:]),dtype="<i2").copy()
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

    async def resume(self):
        if self.mode not in ("LISTEN","WAIT"):return
        if self.endpoint_task and not self.endpoint_task.done():
            self.endpoint_task.cancel()
        if self.mode=="WAIT":await self.state("LISTEN")

    async def commit(self):
        if self.mode not in ("LISTEN","WAIT"):return
        data=bytes(self.samples)
        self.samples.clear()
        await self.state("THINK")
        await self.emit("stage",component="asr",message="Final recognition running")
        self.committed_at=time.monotonic()
        version=self.turn
        async def run():
            try:
                transcript=await transcribe(np.frombuffer(data,dtype="<i2"))
                if version!=self.turn:return
                if not transcript:
                    await self.state("IDLE")
                    return
                await self.emit("metric",name="final_asr_ms",value=round((time.monotonic()-self.committed_at)*1000),turn=version)
                await self.emit("transcript",text=transcript,turn=version)
                self.history.append({"role":"user","content":transcript})
                await self.emit("stage",component="llm",message=f"Generating via {os.getenv('LLM_BACKEND','groq')}")
                await self.respond(version)
            except asyncio.CancelledError:raise
            except Exception as error:
                if version==self.turn:
                    detail=str(error)[:350]
                    if "out of memory" in detail.lower():
                        detail += " | Restart Kaggle; use ASR_MODEL=small, LLM_BACKEND=api or Qwen3-0.6B."
                    elif "invalid device" in detail.lower():
                        detail += " | Check GPU inventory and ASR_GPU_INDEX/TTS_GPU values."
                    await self.emit("error",component="response",message=detail)
                if version==self.turn:await self.state("IDLE")
        self.response_task=asyncio.create_task(run())

    async def send_pcm(self,audio,version,role="speech"):
        if role=="speech" and not self.first_audio and version==self.turn:
            await self.state("SPEAK")
            self.first_audio=True
            await self.emit("metric",name="first_pcm_after_commit_ms",value=round((time.monotonic()-self.committed_at)*1000),turn=version)
        pcm=np.clip(np.asarray(audio,dtype=np.float32).reshape(-1),-1,1)
        pcm=(pcm*32767).astype("<i2")
        for offset in range(0,len(pcm),RATE//8):
            if version!=self.turn:return
            chunk=pcm[offset:offset+RATE//8]
            await self.emit("audio",pcm16=base64.b64encode(chunk.tobytes()).decode(),
                rate=RATE,turn=version,role=role)
            # Backpressure; avoid flooding the client queue minutes ahead.
            await asyncio.sleep(len(chunk) / RATE * 0.85)

    async def respond(self,version):
        await self.state("THINK")
        q=asyncio.Queue(maxsize=6)
        fragments=[]
        first_llm_token_at=None
        async def produce():
            nonlocal first_llm_token_at
            buffer=""
            try:
                async for delta in llm_tokens(self.history):
                    if version!=self.turn:return
                    if first_llm_token_at is None:
                        first_llm_token_at=time.monotonic()
                        await self.emit("metric",name="llm_first_token_after_commit_ms",
                            value=round((first_llm_token_at-self.committed_at)*1000),turn=version)
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
                await self.emit("stage",component="tts",message=f"Synthesizing {language} voice")
                tts_started=time.monotonic()
                pcm=await voices.synthesize(phrase,language)
                await self.emit("metric",name="tts_generate_ms",value=round((time.monotonic()-tts_started)*1000),turn=version)
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
    return {"ready":True,"asr_device":ASR_DEVICE,"asr_gpu_index":ASR_INDEX,
            "asr_model":os.getenv("ASR_MODEL","small"),
            "tts_provider":voices.provider,
            "llm_backend":os.getenv("LLM_BACKEND","groq"),
            "smart_turn":os.getenv("SMART_TURN_ENABLED","0")=="1",
            "gpu":inventory()}

def _require_token(authorization):
    if not hmac.compare_digest(authorization,"Bearer "+TOKEN):
        raise HTTPException(status_code=401,detail="Invalid token")

@app.post("/debug/llm")
async def debug_llm(payload:dict, authorization:str=Header(default="")):
    """Runs the configured LLM inside the server process, not a duplicate GPU instance."""
    _require_token(authorization)
    question=str(payload.get("text","Hello Jarvis. Please reply in one sentence."))[:500]
    try:
        answer=[]
        async for delta in llm_tokens([
            {"role":"system","content":"You are Jarvis. Reply in one concise sentence."},
            {"role":"user","content":question}
        ]):
            answer.append(delta)
            if sum(map(len,answer))>1000:break
        return {"ok":True,"backend":os.getenv("LLM_BACKEND","groq"),"reply":"".join(answer)}
    except Exception as exc:
        raise HTTPException(503,detail=f"LLM failed: {str(exc)[:350]}")

@app.post("/debug/tts")
async def debug_tts(payload:dict,authorization:str=Header(default="")):
    """Synthesize ALL input text in server process, combining every speech chunk."""
    _require_token(authorization)
    text=payload.get("text","Hello. I am Jarvis.")
    if not isinstance(text,str) or not text.strip():
        raise HTTPException(400,detail="Provide nonempty text")
    max_chars=int(os.getenv("TTS_MAX_TEXT_CHARS","5000"))
    if len(text)>max_chars:
        raise HTTPException(413,detail=f"Too much text: {len(text)} > {max_chars} chars. Nothing truncated.")
    lang="ta" if str(payload.get("language","en")).lower().startswith("ta") else "en"
    try:
        import io
        import soundfile as sf
        chunks=chunk_text(text,max_chars=int(os.getenv("TTS_CHUNK_MAX_CHARS","100")))
        if not chunks:
            raise HTTPException(400,detail="No speech content")
        pieces=[]
        for i,chunk in enumerate(chunks,1):
            print(f"[debug/tts] {lang} chunk {i}/{len(chunks)}: {chunk!r}",flush=True)
            audio=await voices.synthesize(chunk,lang)
            pieces.append(audio)
        combined=combine_pcm(pieces,sample_rate=RATE,
                             gap_ms=int(os.getenv("TTS_CHUNK_GAP_MS","100")))
        output=io.BytesIO()
        sf.write(output,combined,RATE,format="WAV")
        return Response(content=output.getvalue(),media_type="audio/wav",
                        headers={"X-TTS-Chunks":str(len(chunks)),
                                 "X-TTS-Input-Characters":str(len(text)),
                                 "X-TTS-Duration-Seconds":f"{len(combined)/RATE:.2f}"})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503,detail=f"TTS failed: {str(exc)[:350]}")

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
                elif kind=="resume":
                    await s.resume()
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
