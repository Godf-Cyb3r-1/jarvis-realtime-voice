"""GPU speech service. Requires a working compatible OmniVoice installation."""
import io, os, threading
from pathlib import Path
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from omnivoice import OmniVoice
import torch

TOKEN=os.environ["VOICE_API_TOKEN"]
ROOT=Path(__file__).resolve().parent
if not torch.cuda.is_available(): raise RuntimeError("A CUDA GPU is required")
model=OmniVoice.from_pretrained("k2-fsa/OmniVoice",device_map="cuda:0",dtype=torch.float16)
lock=threading.Lock()
app=FastAPI()

class Synthesis(BaseModel):
    text:str
    language:str="en"

@app.get("/health")
def health(authorization:str=Header(default="")):
    if authorization!="Bearer "+TOKEN: raise HTTPException(401)
    return {"ready":True}

@app.post("/synthesize")
def synthesize(req:Synthesis,authorization:str=Header(default="")):
    if authorization!="Bearer "+TOKEN: raise HTTPException(401)
    if not req.text.strip() or len(req.text)>1000: raise HTTPException(400)
    language="ta" if req.language.lower().startswith("ta") else "en"
    reference=ROOT/"voices"/f"{language}.wav"
    reference_text=(ROOT/"voices"/f"{language}.txt").read_text(encoding="utf-8").strip()
    with lock:
        output=model.generate(text=req.text,ref_audio=str(reference),ref_text=reference_text)
    waveform=np.asarray(output[0]).squeeze().astype("float32")
    buffer=io.BytesIO()
    sf.write(buffer,waveform,24000,format="WAV")
    return Response(buffer.getvalue(),media_type="audio/wav")
