"""GPU voice cloning via official OmniVoice API; free Edge TTS fallback."""
import asyncio
import os
import subprocess
import tempfile
import threading
from pathlib import Path
import numpy as np

SAMPLE_RATE=16000

def _to16k(audio, sr):
    from scipy.signal import resample_poly
    from math import gcd
    if hasattr(audio,"detach"):audio=audio.detach().float().cpu().numpy()
    x=np.asarray(audio,dtype=np.float32).squeeze()
    if x.ndim!=1:raise ValueError("TTS audio must be mono")
    if int(sr)!=SAMPLE_RATE:
        d=gcd(int(sr),SAMPLE_RATE)
        x=resample_poly(x,SAMPLE_RATE//d,int(sr)//d).astype(np.float32)
    return np.clip(x,-1.0,1.0)

class OmniVoiceEngine:
    def __init__(self):
        import torch
        from omnivoice import OmniVoice
        if not torch.cuda.is_available():raise RuntimeError("OmniVoice needs a CUDA GPU")
        gpu=int(os.getenv("TTS_GPU",str(1 if torch.cuda.device_count()>1 else 0)))
        self.model=OmniVoice.from_pretrained(os.getenv("OMNIVOICE_MODEL","k2-fsa/OmniVoice"),
            device_map=f"cuda:{gpu}",dtype=torch.float16,load_asr=False)
        root=Path(os.getenv("VOICE_DIR","voices"))
        self.prompts={}
        for lang in ("en","ta"):
            audio=root/f"{lang}.wav";transcript=root/f"{lang}.txt"
            if not audio.is_file() or not transcript.is_file():
                raise FileNotFoundError(f"Missing private {audio} or {transcript}")
            self.prompts[lang]=self.model.create_voice_clone_prompt(
                ref_audio=str(audio),ref_text=transcript.read_text(encoding="utf-8").strip())
        self.guard=threading.Lock()
    def generate(self,text,language):
        with self.guard:
            output=self.model.generate(text=text,
                voice_clone_prompt=self.prompts["ta" if language=="ta" else "en"],
                num_step=int(os.getenv("OMNIVOICE_STEPS","16")))
        return _to16k(output[0],getattr(self.model,"sampling_rate",24000))

async def edge_generate(text,language):
    import edge_tts
    voice=os.getenv("EDGE_VOICE_TA" if language=="ta" else "EDGE_VOICE_EN",
                    "ta-IN-PallaviNeural" if language=="ta" else "en-IN-NeerjaNeural")
    with tempfile.TemporaryDirectory() as d:
        mp3=Path(d)/"audio.mp3";wav=Path(d)/"audio.wav"
        await edge_tts.Communicate(text,voice).save(str(mp3))
        proc=await asyncio.create_subprocess_exec("ffmpeg","-y","-v","error",
            "-i",str(mp3),"-ac","1","-ar","16000",str(wav))
        if await proc.wait():raise RuntimeError("ffmpeg conversion failed")
        import soundfile as sf
        audio,sr=await asyncio.to_thread(sf.read,wav,dtype="float32")
        return _to16k(audio,sr)

class VoiceEngine:
    def __init__(self):
        self.provider=os.getenv("TTS_PROVIDER","omnivoice").lower()
        self.omni=None
        self.loading=asyncio.Lock()
    async def synthesize(self,text,language):
        if self.provider=="edge":
            return await edge_generate(text,language)
        if self.provider!="omnivoice":
            raise ValueError("Set TTS_PROVIDER=omnivoice or edge")
        if self.omni is None:
            async with self.loading:
                if self.omni is None:
                    self.omni=await asyncio.to_thread(OmniVoiceEngine)
        return await asyncio.to_thread(self.omni.generate,text,language)
