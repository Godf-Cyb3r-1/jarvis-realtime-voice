# Run Jarvis Duplex v2 using free Kaggle T4 x2 and Windows

**Status:** code committed and CPU policy tests provided; end-to-end GPU + microphone/audio hardware tests are **not yet executed**. This is an engineered duplex prototype; it cannot promise native-model-level conversational timing.

## 1. Kaggle notebook
1. Create a Kaggle notebook at https://www.kaggle.com/code.
2. Select accelerator GPU T4 x2 (or T4 if only one available).
3. Turn Internet on, if offered.
4. Upload the original `voices.zip` as a **PRIVATE** input dataset. Its internal layout must be `jarvis_en/new-ref.mp3`, `jarvis_en/ref.txt`, `jarvis_ta/ref.mp3`, `jarvis_ta/ref.txt`. These personal voice files are intentionally **not** in the public repository.
5. In Kaggle settings, add notebook secret `VOICE_API_TOKEN`: long random text (e.g. `secrets.token_urlsafe(32)` generated in your local terminal). Add `OPENROUTER_API_KEY` if using a free OpenRouter model.

### Kaggle cell A — obtain source and dependencies

```python
!git clone https://github.com/Godf-Cyb3r-1/jarvis-realtime-voice.git /kaggle/working/jarvis-realtime-voice
%cd /kaggle/working/jarvis-realtime-voice
%pip install -q -r requirements-duplex-server.txt
%pip install -q omnivoice
import torch
print("GPU available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
```

OmniVoice installation can require a compatible PyTorch/CUDA build. If the install fails, use the official https://github.com/k2-fsa/OmniVoice instructions appropriate for Kaggle rather than installing conflicting Torch wheel versions.

### Kaggle cell B — unpack private voice references

```python
from pathlib import Path
import zipfile, subprocess

root=Path("/kaggle/working/jarvis-realtime-voice")
voices=root/"voices"
voices.mkdir(exist_ok=True)
matches=list(Path("/kaggle/input").rglob("voices.zip"))
assert matches, "Upload voices.zip as a PRIVATE Kaggle dataset first"
with zipfile.ZipFile(matches[0]) as z:
    for lang,mp3,txt in [
        ("en","jarvis_en/new-ref.mp3","jarvis_en/ref.txt"),
        ("ta","jarvis_ta/ref.mp3","jarvis_ta/ref.txt"),
    ]:
        input_mp3=voices/f"{lang}.mp3"
        input_mp3.write_bytes(z.read(mp3))
        (voices/f"{lang}.txt").write_bytes(z.read(txt))
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(input_mp3),
            "-ac","1","-ar","24000",str(voices/f"{lang}.wav")],check=True)
print([(p.name,p.stat().st_size) for p in voices.iterdir()])
```

### Kaggle cell C — configure GPU workers and free LLM

```python
import os
from kaggle_secrets import UserSecretsClient
secret=UserSecretsClient()
os.environ["VOICE_API_TOKEN"]=secret.get_secret("VOICE_API_TOKEN")
os.environ["LLM_BASE_URL"]="https://openrouter.ai/api/v1"
os.environ["LLM_MODEL"]="openrouter/free"
os.environ["LLM_API_KEY"]=secret.get_secret("OPENROUTER_API_KEY")
os.environ["ASR_MODEL"]="medium"  # try large-v3-turbo for Tamil if latency permits
os.environ["ASR_DEVICE"]="cuda"  # faster-whisper defaults to GPU0
os.environ["ASR_COMPUTE"]="float16"
os.environ["ASR_LANGUAGE"]="auto"
os.environ["TTS_PROVIDER"]="omnivoice"
os.environ["TTS_GPU"]="1" if __import__("torch").cuda.device_count()>1 else "0"
os.environ["VOICE_DIR"]=str(voices)
os.environ["OMNIVOICE_STEPS"]="16"
os.environ["SMART_TURN_ENABLED"]="1"
os.environ["ALLOW_BACKCHANNEL"]="0"  # enable only after duplex verification
```

The free OpenRouter router is optional: set `LLM_BASE_URL` / `LLM_MODEL` to any other **OpenAI-compatible streaming** model you already have permission to use. `openrouter/free` is rate limited and may pick different models; it's not a guaranteed always-free unlimited reasoning service. Never place API keys in code or GitHub.

### Kaggle cell D — test models and start server

```python
!python -m unittest discover -s tests -p "test_*.py" -v
!python -m compileall -q duplex_policy.py smart_turn_adapter.py tts_engine.py duplex_audio.py duplex_server.py duplex_client.py
from tts_engine import OmniVoiceEngine
synth=OmniVoiceEngine()   # model loads on GPU1, English and Tamil reference prompts precomputed
pcm=synth.generate("Hello, I am Jarvis.", "en")
print("OmniVoice sample count:",len(pcm))
from IPython.display import Audio, display
display(Audio(pcm,rate=16000))
# Release GPU memory before launching a separate server process:
del synth
import gc
gc.collect()
__import__("torch").cuda.empty_cache()
```

If this fails, collect the actual exception (don't assume voice cloning is working).

```python
import subprocess,requests,time
server=subprocess.Popen(["python","-m","uvicorn","duplex_server:app",
    "--host","127.0.0.1","--port","8091"],
    stdout=open("/kaggle/working/jarvis_server.log","w"),
    stderr=subprocess.STDOUT)
time.sleep(8)
print(requests.get("http://127.0.0.1:8091/health",timeout=10).json())
```

If health fails: `print(open("/kaggle/working/jarvis_server.log").read()[-4000:])`.

### Kaggle cell E — temporary encrypted WSS tunnel

Only do this for development. Cloudflare Quick Tunnels are **public temporary URLs**, though this server additionally checks an Authorization header. Do not share the token or tunnel URL. They have no uptime guarantee.

```python
import subprocess,re,time
subprocess.run(["wget","-q",
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "-O","/kaggle/working/cloudflared"],check=True)
subprocess.run(["chmod","+x","/kaggle/working/cloudflared"],check=True)
log=open("/kaggle/working/cloudflared.log","w")
tunnel=subprocess.Popen(["/kaggle/working/cloudflared","tunnel",
    "--url","http://127.0.0.1:8091"],stdout=log,stderr=subprocess.STDOUT)
time.sleep(10)
text=open("/kaggle/working/cloudflared.log").read()
match=re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",text)
print("VOICE_WS_URL =",match.group(0).replace("https://","wss://")+"/ws" if match else text[-1600:])
```

If the tunnel fails, check Kaggle Internet availability and logs. The URL changes every session and dies when Kaggle disconnects.

## 2. Windows installation
Windows PowerShell:

```powershell
git clone https://github.com/Godf-Cyb3r-1/jarvis-realtime-voice.git
cd jarvis-realtime-voice
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-duplex-client.txt
Copy-Item .env.example .env
notepad .env
```

Inside `.env`, set:
- `VOICE_WS_URL=wss://<the-url-from-Kaggle>/ws`
- `VOICE_API_TOKEN=<the-same-Kaggle-token>`
- `AEC_ENABLED=0` while using **headphones**
- `VAD_MODE=2`

Remove placeholders / angle brackets when entering real values. Keep .env private.

Launch:

```powershell
.\.venv\Scripts\python.exe duplex_client.py
```

Optional speaker echo-cancellation trial: `.\.venv\Scripts\python.exe -m pip install pywebrtc-audio`, then `AEC_ENABLED=1` in Windows `.env`. Results depend on device timing and selected input/output drivers; headphones are the reliable baseline.

## 3. Voice experiments
- Say: "Jarvis, are you free now?" (English recognition + response)
- Speak Tamil, then Tanglish, and check whether the result is accurate.
- Pause mid-sentence: "I was thinking... [0.7 seconds] ...about yesterday." This should not finalize early if Smart Turn judges the phrase incomplete.
- While Jarvis is speaking, say "hmm": should briefly duck and resume.
- While Jarvis is speaking, say "Wait, that's not what I meant": should cancel its old audio and listen to your replacement.
- Measure: speech-end to first audio; interruption until Jarvis stops; Tamil ASR errors and voice fidelity.

See [research/RESEARCH_AND_PLAN.md](research/RESEARCH_AND_PLAN.md) for papers, limitations, acceptance criteria.

**Known blockers:** No proof of Kaggle GPU run or end-to-end microphone test yet. Echo cancellation and semantic overlap behavior may still require tuning. Smart Turn does not guarantee accurate Tamil endpoint classification. Kaggle free sessions expire; an always-on service needs another hosting plan.
