"""Kaggle B: dedicated OmniVoice WebSocket synthesis with speculative warmups.

GPU model is loaded once. Prefill output is cached but NEVER automatically
played. Windows must explicitly confirm a final utterance and request speech.
Cancelled/obsolete turns cannot enqueue audible PCM.
"""
import asyncio
import base64
import hmac
import os
import time
from collections import OrderedDict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException
from realtime_protocol import RATE
from fastapi.responses import Response
from tts_engine import VoiceEngine
from tts_longform import chunk_text

TOKEN=os.getenv("TTS_WS_TOKEN","")
if len(TOKEN)<24:
    raise RuntimeError("Set TTS_WS_TOKEN (24+ random characters) via Kaggle Secret")
app=FastAPI(title="Jarvis TTS Voice")
voices=VoiceEngine()
gpu_lock=asyncio.Lock()
cache=OrderedDict()
CACHE_LIMIT=int(os.getenv("TTS_PREFILL_CACHE_LIMIT","8"))

async def synth(text,lang):
    # One GPU generation at a time; different WS clients cannot race on OmniVoice.
    async with gpu_lock:
        return await voices.synthesize(text,lang)

def cache_put(key,lang,text,audio):
    if not key:
        return
    cache[key]=(lang,text,audio)
    cache.move_to_end(key)
    while len(cache)>CACHE_LIMIT:
        cache.popitem(last=False)

@app.get("/health")
async def health():
    return {"ready":True,"service":"tts","provider":voices.provider,
            "sample_rate":RATE,"prefill_cache":len(cache)}

@app.post("/debug/tts")
async def debug_tts(payload:dict,authorization:str=Header(default="")):
    """Authenticated Kaggle-only voice test; generates and concatenates full text."""
    if not hmac.compare_digest(authorization,"Bearer "+TOKEN):
        raise HTTPException(401,detail="Invalid bearer token")
    text=payload.get("text","Hello Jarvis.")
    if not isinstance(text,str) or not text.strip() or len(text)>5000:
        raise HTTPException(400,detail="Text must be 1-5000 characters")
    lang="ta" if str(payload.get("language","en")).startswith("ta") else "en"
    try:
        import io
        import soundfile as sf
        from tts_longform import combine_pcm
        parts=chunk_text(text,max_chars=110)
        pcm=combine_pcm([await synth(p,lang) for p in parts])
        output=io.BytesIO()
        sf.write(output,pcm,RATE,format="WAV")
        return Response(content=output.getvalue(),media_type="audio/wav",
            headers={"X-TTS-Chunks":str(len(parts)),
                     "X-TTS-Input-Characters":str(len(text))})
    except Exception as exc:
        raise HTTPException(503,detail=f"OmniVoice failed: {str(exc)[:280]}")

@app.websocket("/tts")
async def socket(ws:WebSocket):
    if not hmac.compare_digest(ws.headers.get("authorization",""),"Bearer "+TOKEN):
        await ws.close(code=1008)
        return
    await ws.accept()
    send_lock=asyncio.Lock()
    queue=asyncio.Queue(maxsize=12)
    background=set()
    turn_id=-1
    speaking=False

    async def send(kind,**kw):
        async with send_lock:
            await ws.send_json({"type":kind,**kw})

    async def prefill(req):
        key=str(req.get("cache_key",""))[:128]
        text=str(req.get("text",""))[:220].strip()
        lang="ta" if str(req.get("language","en")).startswith("ta") else "en"
        if not key or not text:
            return
        if key in cache and cache[key][:2]==(lang,text):
            await send("prefill_ready",cache_key=key,hit=True)
            return
        try:
            # Don't allow prefill to compete with committed audio.
            if speaking or not queue.empty() or gpu_lock.locked():
                await send("prefill_skipped",cache_key=key,reason="busy")
                return
            pcm=await synth(text,lang)
            cache_put(key,lang,text,pcm)
            await send("prefill_ready",cache_key=key,hit=False)
        except Exception as exc:
            await send("error",stage="tts_prefill",message=str(exc)[:240])

    async def worker():
        nonlocal speaking
        while True:
            job=await queue.get()
            if job is None:
                break
            turn=int(job["turn"])
            segment=int(job.get("segment",0))
            text=str(job.get("text","")).strip()
            lang="ta" if str(job.get("language","en")).startswith("ta") else "en"
            key=str(job.get("cache_key",""))[:128]
            if turn!=turn_id or not text:
                queue.task_done()
                continue
            speaking=True
            start=time.monotonic()
            try:
                parts=chunk_text(text,max_chars=int(os.getenv("TTS_CHUNK_MAX_CHARS","110")))
                for part_no,part in enumerate(parts):
                    if turn!=turn_id:
                        break
                    hit=cache.get(key) if part_no==0 else None
                    if hit is not None and hit[0]==lang and hit[1]==part:
                        pcm=hit[2]
                        cache.move_to_end(key)
                    else:
                        pcm=await synth(part,lang)
                    if turn!=turn_id:
                        break
                    pcm=(np.clip(np.asarray(pcm,dtype=np.float32),-1,1)*32767).astype("<i2")
                    await send("tts_info",turn=turn,segment=segment,part=part_no,
                               chars=len(part),synthesis_ms=round((time.monotonic()-start)*1000),
                               duration_ms=round(len(pcm)/RATE*1000))
                    # Output in 80ms chunks: playback can start while later phrases generate.
                    block_size=int(RATE*0.08)
                    for offset in range(0,len(pcm),block_size):
                        if turn!=turn_id:
                            break
                        block=pcm[offset:offset+block_size]
                        await send("tts_audio",turn=turn,segment=segment,part=part_no,
                                   pcm16=base64.b64encode(block.tobytes()).decode(),
                                   rate=RATE)
                        await asyncio.sleep(len(block)/RATE*0.80)
                if turn==turn_id:
                    await send("tts_done",turn=turn,segment=segment)
            except Exception as exc:
                if turn==turn_id:
                    await send("error",stage="tts",turn=turn,segment=segment,
                               message=str(exc)[:260])
            finally:
                speaking=False
                queue.task_done()

    worker_task=asyncio.create_task(worker())
    await send("ready",service="tts",rate=RATE,frame_ms=80)
    try:
        async for request in ws.iter_json():
            kind=request.get("type")
            if kind=="tts_prefill":
                task=asyncio.create_task(prefill(request))
                background.add(task)
                task.add_done_callback(background.discard)
            elif kind=="tts_speak":
                turn=int(request.get("turn",-1))
                if turn<turn_id:
                    continue
                if turn>turn_id:
                    turn_id=turn
                    # Drop queued segments from obsolete turns.
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                            queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                try:
                    queue.put_nowait(request)
                except asyncio.QueueFull:
                    await send("error",stage="tts_queue",turn=turn,message="TTS queue full")
            elif kind=="tts_cancel":
                turn=int(request.get("turn",-1))
                if turn>=turn_id:
                    turn_id=turn+1
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                            queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    await send("tts_canceled",turn=turn)
            elif kind=="ping":
                await send("pong",monotonic_ms=round(time.monotonic()*1000))
    except WebSocketDisconnect:
        pass
    finally:
        for task in background:
            task.cancel()
        # Stop worker but don't hang forever if model is still synthesizing.
        worker_task.cancel()
        await asyncio.gather(worker_task,*background,return_exceptions=True)
