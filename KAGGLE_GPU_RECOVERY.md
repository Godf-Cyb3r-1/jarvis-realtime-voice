# Kaggle CUDA OOM + invalid device: recovery and isolated tests

This is for the Windows logs:
- `[error:asr_partial] CUDA failed with error out of memory`
- `[error:response] parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal`
- Final STT works but Jarvis never speaks.

These errors occur on the **Kaggle backend**. Reinstalling Windows speech recognition does not fix them.

## Step 0 — Stop the old processes and fully restart Kaggle
Press Ctrl+C in Windows to stop `duplex_client.py`.
In the current Kaggle notebook, stop the old server and tunnel if they are stored as variables:

```python
for name in ("server","tunnel"):
    if name in globals():
        try:globals()[name].terminate()
        except Exception:pass
```

Use Kaggle **Restart Session** (not just rerun a server cell). A Python `torch.cuda.empty_cache()` call cannot reclaim all allocations held by a still-running Qwen/Whisper/OmniVoice process or background thread.

Do not run multiple uvicorn servers or keep an old model from `synth=OmniVoiceEngine()` loaded inside the notebook after launching a separate server.

## Step 1 — Confirm what GPUs exist

```python
!nvidia-smi
import torch
import ctranslate2
print("Torch visible GPU count:",torch.cuda.device_count())
print("CTranslate2 visible GPU count:",ctranslate2.get_cuda_device_count())
print("CUDA_VISIBLE_DEVICES:",__import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
```

If only one GPU is visible, `TTS_GPU=1` is invalid. Do not assume two GPUs just because the notebook previously used T4 x2. CUDA indices are **logical, zero-based** and can be remapped by `CUDA_VISIBLE_DEVICES`.

## Step 2 — Pull the fixed source

```python
!git -C /kaggle/working/jarvis-realtime-voice pull --ff-only
%cd /kaggle/working/jarvis-realtime-voice
!python -m gpu_guard
!python -m unittest discover -s tests -p "test_*.py" -v
!python -m compileall -q gpu_guard.py duplex_server.py duplex_client.py tts_engine.py local_llm.py
```

If the repository directory is missing, use `!git clone https://github.com/Godf-Cyb3r-1/jarvis-realtime-voice.git /kaggle/working/jarvis-realtime-voice` first.

## Step 3 — Safe low-memory setup

Make sure your PRIVATE voices are present in `/kaggle/working/jarvis-realtime-voice/voices/` and the Kaggle secret `VOICE_API_TOKEN` is set.

```python
import os
from pathlib import Path
import torch
from kaggle_secrets import UserSecretsClient
os.environ["VOICE_API_TOKEN"]=UserSecretsClient().get_secret("VOICE_API_TOKEN")
root=Path("/kaggle/working/jarvis-realtime-voice")
voices=root/"voices"
assert all((voices/f).is_file() for f in ("en.wav","en.txt","ta.wav","ta.txt")), "Re-run reference voice extraction first"
os.environ["ASR_DEVICE"]="cuda"
os.environ["ASR_GPU_INDEX"]="auto"
os.environ["ASR_MODEL"]="small"        # medium was using too much VRAM
os.environ["ASR_COMPUTE"]="int8_float16"
os.environ["ASR_BEAM_SIZE"]="1"
os.environ["ASR_LANGUAGE"]="auto"
os.environ["PARTIAL_ASR_ENABLED"]="1"
os.environ["SMART_TURN_ENABLED"]="0"   # enable after normal speech works
os.environ["TTS_PROVIDER"]="omnivoice"
os.environ["TTS_GPU"]="auto"           # automatically selects GPU1 if two are visible
os.environ["VOICE_DIR"]=str(voices)
os.environ["OMNIVOICE_STEPS"]="12"
os.environ["ALLOW_BACKCHANNEL"]="0"
os.environ["LLM_BACKEND"]="local"       # no API key
os.environ["LOCAL_LLM_MODEL"]="Qwen/Qwen3-0.6B"
os.environ["LOCAL_LLM_GPU_INDEX"]="auto"
os.environ["LOCAL_LLM_MAX_TOKENS"]="100"
print("CUDA devices:",torch.cuda.device_count())
print("Server config prepared")
```

This preserves local **no-paid-API reasoning**, uses smaller Qwen/Whisper models and reduces model generation size. It still may OOM if there is only one visible GPU and OmniVoice cannot fit. The safest **two-GPU** allocation is GPU0 for Whisper Small + Qwen3-0.6B and GPU1 for OmniVoice.

For one visible GPU, prefer a **free hosted LLM** (`LLM_BACKEND=api` and an available free OpenAI-compatible provider with a valid key). That leaves GPU0 for Whisper/OmniVoice. Alternatively use `TTS_PROVIDER=edge` temporarily to prove the duplex pipeline, but that is **not** your reference cloned voice.

## Step 4 — Start only one server
Use the server-start cell from `KAGGLE_FULL_DUPLEX.md`. You should see `ready: true` plus fields `asr_gpu_index`, `asr_model`, `llm_backend`, and `gpu`.

```python
import requests
health=requests.get("http://127.0.0.1:8091/health",timeout=10).json()
print(health)
```

## Step 5 — Test Qwen without Windows
This is an isolated debug endpoint executed **in the server process** (so it doesn't load another Qwen copy into GPU memory).

```python
import os,requests
headers={"Authorization":"Bearer "+os.environ["VOICE_API_TOKEN"]}
llm_test=requests.post("http://127.0.0.1:8091/debug/llm",
    headers=headers,json={"text":"Hi Jarvis, say hello in one sentence."},timeout=180)
print("LLM status:",llm_test.status_code)
print(llm_test.text[:1200])
```

You need `"ok":true` and a nonempty `reply`. If it fails, **do not proceed to Windows yet**. Inspect the server log in the next section. OOM here means Qwen still competes with too much GPU usage.

## Step 6 — Test OmniVoice without Windows
This tests the *server's existing TTS engine* so it does not allocate a second copy of the model.

```python
voice_test=requests.post("http://127.0.0.1:8091/debug/tts",
    headers=headers,json={"text":"Hello, I'm Jarvis.","language":"en"},timeout=240)
print("TTS status:",voice_test.status_code)
if voice_test.ok:
    from IPython.display import Audio,display
    path="/kaggle/working/tts_probe.wav"
    Path(path).write_bytes(voice_test.content)
    display(Audio(filename=path))
else:
    print(voice_test.text[:1200])
```

You must hear audible WAV speech. If `/debug/llm` succeeds but `/debug/tts` fails, we have isolated the fault to OmniVoice or GPU allocation. You may explicitly use `TTS_PROVIDER=edge` for a short duplex smoke test, but it does not satisfy the cloned voice target.

## Step 7 — Restart Windows client
Keep Kaggle server and tunnel running. In Windows PowerShell:

```powershell
cd C:\Users\LENOVO\jarvis-realtime-voice
git pull --ff-only
.\.venv\Scripts\python.exe duplex_client.py
```

If Kaggle was restarted, its temporary Cloudflare tunnel URL may have changed. Update `VOICE_WS_URL` in Windows `.env`, and make sure its `VOICE_API_TOKEN` matches the Kaggle notebook secret.

After you say "Hello Jarvis", look for:

```text
[ready] Duplex transport connected. Speak naturally.
[partial] Hello Jarvis
[transcript] Hello Jarvis
[stage:llm] Generating reply
... streamed reply text ...
[stage:tts] Synthesizing en voice
[metric] first_pcm_after_commit_ms: ...
```

If there is a `[stage:llm]` with no reply, inspect Qwen or API. If reply text appears but `[stage:tts]` hangs, inspect OmniVoice. If `first_pcm_after_commit_ms` prints but no sound is audible, check Windows audio output and playback devices.

## Step 8 — Collect the failure without exposing secrets
```python
print(open("/kaggle/working/jarvis_server.log",encoding="utf-8").read()[-4500:])
```

Do not share notebook secrets, environment values containing tokens, or reference recordings. You may send the error text and the redacted `/health` response.

### Why the crash happened
Whisper Medium and Qwen3-1.7B were both loading GPU0 at full precision, potentially simultaneously with repeated interim ASR, and OmniVoice had an explicit GPU1 assignment. If device1 wasn't visible or memory was exhausted, CUDA failures could cascade. The patch now uses lower-memory defaults, a thread-level ASR lock (canceled async tasks cannot overlap GPU execution), device inventory and safe selection, and explicit LLM/TTS debug endpoints.

This is a **code-level fix**, not yet proven on the user's live Kaggle session. The goal is to get one successful LLM test and one successful TTS test before calling the full duplex pipe working.
