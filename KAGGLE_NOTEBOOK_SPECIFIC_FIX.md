# Corrected Kaggle + Windows recovery for uploaded `jarvis-full-duplex.ipynb`

**Source of corrections:** the executed user notebook already showed two visible Tesla T4 GPUs; the PRIVATE Kaggle dataset is **raw folders**, not a ZIP; English/Tamil OmniVoice synthesis worked in the notebook; the older server was launched before extra CUDA libraries/environment changes; and the old configuration used Whisper Medium + Qwen3-1.7B on GPU0, OmniVoice on GPU1.

**Security:** the old notebook printed the bearer token in a cell output. ROTATE it and store the replacement as Kaggle Notebook Secret `VOICE_API_TOKEN`; update Windows `.env`. Do not share that original notebook with its outputs or commit it to GitHub.

## Recover in this order

1. In Windows, press Ctrl+C to stop `duplex_client.py`.
2. In Kaggle, stop any old uvicorn/cloudflared process; use **Restart Session** to reclaim CUDA allocations.
3. If you just installed dependencies, **do not reinstall unnecessarily**. The uploaded notebook successfully imported `omnivoice`, `faster_whisper`, `onnxruntime`, and `transformers` (version 5.23.0 at that time). A kernel restart clears Python variables, so rerun configuration cells before starting the server.
4. Run the source/GPU/voice/configuration cells below **in order**.
5. Start **one** server, test `/debug/llm`, test `/debug/tts`, only then reconnect Windows.

### Kaggle cell 1 — pull the patched repository

```python
from pathlib import Path
import os, subprocess
repo=Path("/kaggle/working/jarvis-realtime-voice")
if (repo/".git").is_dir():
    subprocess.run(["git","-C",str(repo),"pull","--ff-only"],check=True)
else:
    subprocess.run(["git","clone",
      "https://github.com/Godf-Cyb3r-1/jarvis-realtime-voice.git",str(repo)],check=True)
os.chdir(repo)
print("Repository:",repo)
```

### Kaggle cell 2 — verify GPUs and CUDA libraries

```python
import torch, ctranslate2, transformers, omnivoice, site
print("PyTorch GPUs:",torch.cuda.device_count())
print("CTranslate2 GPUs:",ctranslate2.get_cuda_device_count())
print("Transformers version:",transformers.__version__)
for i in range(torch.cuda.device_count()):
    free,total=torch.cuda.mem_get_info(i)
    print(i,torch.cuda.get_device_name(i),round(free/2**30,2),"GiB free")
assert torch.cuda.device_count()>=2, "This baseline expects T4 x2"
```

**Don't force transformers <5** after your successful OmniVoice install. The prior package-install sequence caused Transformers 5.23.0 to be active; it imported and synthesized in your notebook. If imports fail after a fresh session, install project requirements and OmniVoice, restart the kernel, and repeat from cell 1.

### Kaggle cell 3 — your actual raw Kaggle dataset (no voices.zip)

```python
from pathlib import Path
import subprocess, shutil
from soundfile import read
matches=[p for p in Path("/kaggle/input").rglob("jarvis-voice-references")
         if p.is_dir() and (p/"jarvis_en/new-ref.mp3").exists()
                       and (p/"jarvis_ta/ref.mp3").exists()]
assert matches, "Attach the private jarvis-voice-references dataset in Input"
dataset=matches[0]
voices=repo/"voices"
voices.mkdir(exist_ok=True)
for lang,mp3,txt in [
    ("en","jarvis_en/new-ref.mp3","jarvis_en/ref.txt"),
    ("ta","jarvis_ta/ref.mp3","jarvis_ta/ref.txt")
]:
    shutil.copy2(dataset/mp3,voices/f"{lang}.mp3")
    shutil.copy2(dataset/txt,voices/f"{lang}.txt")
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(voices/f"{lang}.mp3"),
                    "-ac","1","-ar","24000",str(voices/f"{lang}.wav")],check=True)
    y,sr=read(voices/f"{lang}.wav")
    print(lang, sr, "Hz", round(len(y)/sr,2),"seconds")
```

### Kaggle cell 4 — safe VRAM settings and secret

Create the Kaggle Secret `VOICE_API_TOKEN` first. Use the **same** replacement token in Windows `.env`.

```python
import os, site
from pathlib import Path
from kaggle_secrets import UserSecretsClient
os.environ["VOICE_API_TOKEN"]=UserSecretsClient().get_secret("VOICE_API_TOKEN")
assert len(os.environ["VOICE_API_TOKEN"])>=24
os.environ.update({
    "ASR_DEVICE":"cuda",
    "ASR_GPU_INDEX":"0",
    "ASR_MODEL":"small",
    "ASR_COMPUTE":"int8_float16",
    "ASR_BEAM_SIZE":"1",
    "ASR_LANGUAGE":"auto",
    "PARTIAL_ASR_ENABLED":"0",  # enable after first speech test
    "SMART_TURN_ENABLED":"0",   # temporarily use fallback turn detection
    "LLM_BACKEND":"local",
    "LOCAL_LLM_MODEL":"Qwen/Qwen3-0.6B",
    "LOCAL_LLM_DEVICE":"cuda:0",
    "LOCAL_LLM_GPU_INDEX":"0",
    "LOCAL_LLM_MAX_TOKENS":"100",
    "TTS_PROVIDER":"omnivoice",
    "TTS_GPU":"1",
    "VOICE_DIR":str(voices),
    "OMNIVOICE_STEPS":"12",
    "ALLOW_BACKCHANNEL":"0",
})
# Pass CUDA12 library paths to the newly launched server.
roots=[Path(x)/"nvidia" for x in site.getsitepackages()]
libs=[str(root/name/"lib") for name in ("cublas","cudnn","cuda_runtime")
      for root in roots if (root/name/"lib").is_dir()]
old=os.environ.get("LD_LIBRARY_PATH","")
os.environ["LD_LIBRARY_PATH"]=":".join(libs+([old] if old else []))
print("Environment ready; bearer token not printed")
```

### Kaggle cell 5 — offline checks and start ONE server

```python
import subprocess,requests,time
subprocess.run(["python","-m","unittest","discover","-s","tests","-p","test_*.py","-v"],check=True)
subprocess.run(["python","-m","compileall","-q",
  "gpu_guard.py","duplex_server.py","duplex_client.py",
  "tts_engine.py","local_llm.py"],check=True)
server_log=open("/kaggle/working/jarvis_server.log","w")
server=subprocess.Popen(["python","-m","uvicorn","duplex_server:app",
  "--host","127.0.0.1","--port","8091"],
  cwd=repo,env=os.environ.copy(),stdout=server_log,stderr=subprocess.STDOUT)
r=None
for attempt in range(120):
    if server.poll() is not None:break
    try:
        r=requests.get("http://127.0.0.1:8091/health",timeout=2)
        if r.ok:break
    except requests.RequestException:pass
    time.sleep(2)
if r is None or not r.ok:
    print(open("/kaggle/working/jarvis_server.log").read()[-4500:])
    raise RuntimeError("Server did not become healthy")
print("Health:",r.json())
```

The health output must show ASR GPU index 0, local LLM backend, OmniVoice, and two GPUs. Avoid starting a separate `synth=OmniVoiceEngine()` inside the notebook; that creates another voice model in an independent process.

### Kaggle cell 6 — prove Qwen and OmniVoice inside the server

```python
from pathlib import Path
from IPython.display import Audio, display
headers={"Authorization":"Bearer "+os.environ["VOICE_API_TOKEN"]}
llm=requests.post("http://127.0.0.1:8091/debug/llm",headers=headers,
    json={"text":"Say a short friendly greeting."},timeout=240)
print("LLM:",llm.status_code,llm.text[:800])
assert llm.ok and llm.json().get("reply")
for lang,text in [("en","Hello, I am Jarvis."),("ta","வணக்கம்! நான் ஜார்விஸ்.")]:
    tts=requests.post("http://127.0.0.1:8091/debug/tts",headers=headers,
        json={"text":text,"language":lang},timeout=240)
    print("TTS:",lang,tts.status_code)
    if not tts.ok:
        print(tts.text[:1000]);break
    fname=Path("/kaggle/working")/f"server_test_{lang}.wav"
    fname.write_bytes(tts.content)
    display(Audio(filename=str(fname)))
```

The original Tamil demo played with `rate=14200` despite returned 16-kHz PCM. Do not repeat that incorrect sample rate. The server diagnostic returns WAV with its proper rate.

### Kaggle cell 7 — restart the temporary tunnel

Use the Cloudflare tunnel cell from your original notebook **after** the new server starts. It will return a fresh `wss://...trycloudflare.com/ws` URL. Keep the token confidential. A Kaggle restart invalidates the old URL.

### Windows (PowerShell)

```powershell
cd C:\Users\LENOVO\jarvis-realtime-voice
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install -r requirements-duplex-client.txt

# Verify microphone + speaker work without Kaggle first:
.\.venv\Scripts\python.exe local_audio_smoke.py

# Edit just VOICE_WS_URL and VOICE_API_TOKEN; use fresh Kaggle values:
notepad .env

# Once BOTH Kaggle server diagnostics succeed:
.\.venv\Scripts\python.exe duplex_client.py
```

You should see `[stage:llm]` and `[stage:tts]`, then `[metric] first_pcm_after_commit_ms` after your final transcript.

After the first complete spoken response, stop/restart the server with `PARTIAL_ASR_ENABLED=1` and then `SMART_TURN_ENABLED=1` to turn the higher-cost duplex features on one by one. If errors reappear, revert the most recent change and inspect `/kaggle/working/jarvis_server.log`.

See [KAGGLE_GPU_RECOVERY.md](KAGGLE_GPU_RECOVERY.md) for additional troubleshooting. This has not been verified on your live Kaggle session.
