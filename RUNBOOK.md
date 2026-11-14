# Jarvis listening + speaking MVP: setup

This is an **untested prototype**, not yet validated full duplex or a production service. Use headphones: echo cancellation is not implemented.

## Kaggle (GPU)
Create a Kaggle notebook with GPU and Internet enabled. In cells:

```python
!git clone https://github.com/Godf-Cyb3r-1/jarvis-realtime-voice.git /kaggle/working/jarvis-realtime-voice
%cd /kaggle/working/jarvis-realtime-voice
%pip install -r requirements-server.txt
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
```

Set environment variables **without printing secrets**: `VOICE_API_TOKEN` (random long token), `LLM_BASE_URL` (OpenAI-compatible API base), `LLM_MODEL`, `LLM_API_KEY`. OpenCode itself is not automatically an HTTP API. Optional: `ASR_MODEL=small`, `ASR_DEVICE=cuda`, `TTS_PROVIDER=edge` (default; cloud synthetic voice, not cloned).

Start:

```python
import subprocess
server=subprocess.Popen(["python","-m","uvicorn","server:app","--host","127.0.0.1","--port","8091"])
```

Test health with `requests.get("http://127.0.0.1:8091/health").json()`.

## Windows
Clone repo, create Python 3.11 virtual environment, install `requirements-client.txt`. Copy `.env.example` to `.env`. Set `VOICE_WS_URL=wss://YOUR-SECURE-RELAY/ws` and the same `VOICE_API_TOKEN`. Run `python client.py`.

Kaggle's loopback server is not reachable from Windows: configure a **private authenticated WSS-capable relay/tunnel** to port 8091 first. The project does not provision the tunnel. Never expose an unauthenticated GPU service. Kaggle sessions are temporary.

## Verify
1. GPU health responds.
2. Windows client says Connected.
3. Speak English: see partial/final ASR.
4. See LLM reply and hear TTS.
5. Interrupt Jarvis while speaking using headphones.
6. Test Tamil/Tanglish and note errors.

## Limitations
- Microphone stays active during playback, but echo cancellation is absent.
- Silence-based turn detector may prematurely finalize an utterance.
- ASR partials repeatedly decode accumulated audio, not a native streaming decoder.
- Default Edge TTS requires internet and does not use your reference voices.
- OmniVoice adapter in older gpu_server.py is unverified and **not integrated in this MVP**.
- No persistent memory or semantic turn-taking/backchannels.
- Token in WebSocket query can appear in proxy logs; use a private relay and rotate it if leaked.
