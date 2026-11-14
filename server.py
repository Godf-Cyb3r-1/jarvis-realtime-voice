"""GPU-hosted streaming ASR -> LLM -> TTS conversational prototype."""
import asyncio,base64,io,json,os,re,tempfile
from pathlib import Path
import numpy as np
import httpx,soundfile as sf
from faster_whisper import WhisperModel
from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from dotenv import load_dotenv
load_dotenv()
TOKEN=os.getenv("VOICE_API_TOKEN")
if not TOKEN: raise RuntimeError("VOICE_API_TOKEN is required")
model=WhisperModel(os.getenv("ASR_MODEL","small"),device=os.getenv("ASR_DEVICE","cuda"),compute_type=os.getenv("ASR_COMPUTE","float16"))
app=FastAPI()
def recognize(samples):
    segments,_=model.transcribe(np.asarray(samples,dtype=np.float32)/32768,beam_size=3,vad_filter=False,condition_on_previous_text=False,language=None if os.getenv("ASR_LANGUAGE","auto")=="auto" else os.getenv("ASR_LANGUAGE"))
    return " ".join(s.text.strip() for s in segments).strip()
async def respond(history):
    base=os.getenv("LLM_BASE_URL","").rstrip("/")
    if not base or not os.getenv("LLM_MODEL"): return "I heard you. Configure my LLM connection to enable conversation."
    async with httpx.AsyncClient(timeout=90) as c:
        r=await c.post(base+"/chat/completions",headers={"Authorization":"Bearer "+os.getenv("LLM_API_KEY","")},json={"model":os.getenv("LLM_MODEL"),"messages":history[-12:],"stream":False})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
async def speak(text,lang):
    import edge_tts
    voice="ta-IN-PallaviNeural" if lang=="ta" else "en-IN-NeerjaNeural"
    with tempfile.TemporaryDirectory() as d:
        mp3=Path(d)/"out.mp3";wav=Path(d)/"out.wav"
        await edge_tts.Communicate(text,voice).save(str(mp3))
        p=await asyncio.create_subprocess_exec("ffmpeg","-y","-v","error","-i",str(mp3),"-ac","1","-ar","16000",str(wav))
        if await p.wait(): raise RuntimeError("ffmpeg failed")
        return sf.read(wav,dtype="float32")
@app.get("/health")
async def health():return {"ready":True}
@app.websocket("/ws")
async def ws_session(ws:WebSocket):
    if ws.query_params.get("token")!=TOKEN:
        await ws.close(code=1008);return
    await ws.accept()
    history=[{"role":"system","content":"You are Jarvis, a natural, warm, concise conversational companion. Speak English, Tamil, or Tanglish to match the user. Avoid repetitive fillers."}]
    samples=[];version=0;pending=None;partial_busy=False
    send_lock=asyncio.Lock()
    async def emit(data):
        async with send_lock: await ws.send_json(data)
    async def finalize(data,v):
        try:
            text=await asyncio.to_thread(recognize,data)
            if v!=version:return
            if not text:
                await emit({"type":"empty"});return
            await emit({"type":"transcript","text":text})
            history.append({"role":"user","content":text})
            reply=await respond(history)
            if v!=version:return
            history.append({"role":"assistant","content":reply})
            await emit({"type":"reply","text":reply})
            lang="ta" if any("\u0b80"<=ch<="\u0bff" for ch in text+reply) else "en"
            for chunk in re.split(r"(?<=[.!?।])\\s+",reply):
                if v!=version:return
                if not chunk.strip():continue
                audio,rate=await speak(chunk,lang)
                if v!=version:return
                await emit({"type":"audio","rate":rate,"pcm_f32":base64.b64encode(np.asarray(audio,dtype=np.float32).tobytes()).decode()})
            await emit({"type":"response_done"})
        except asyncio.CancelledError:raise
        except Exception as e:await emit({"type":"error","message":str(e)[:500]})
    await emit({"type":"ready"})
    try:
        async for raw in ws.iter_text():
            msg=json.loads(raw);kind=msg.get("type")
            if kind in ("start","interrupt"):
                version+=1
                if pending:pending.cancel()
                samples=[]
            elif kind=="audio":
                samples.extend(np.frombuffer(base64.b64decode(msg["pcm16"]),dtype="<i2").tolist())
                if len(samples)>16000*45:samples=samples[-16000*45:]
            elif kind=="partial" and not partial_busy and len(samples)>=16000:
                partial_busy=True
                snap=list(samples);v=version
                async def partial():
                    nonlocal partial_busy
                    try:
                        text=await asyncio.to_thread(recognize,snap)
                        if text and v==version:await emit({"type":"partial","text":text})
                    finally:partial_busy=False
                asyncio.create_task(partial())
            elif kind=="end" and len(samples)>=1600:
                snap=list(samples);samples=[];v=version
                pending=asyncio.create_task(finalize(snap,v))
    except WebSocketDisconnect:pass
    finally:
        if pending:pending.cancel()
