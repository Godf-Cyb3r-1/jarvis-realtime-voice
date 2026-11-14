# Jarvis Realtime Voice

Experimental **continuous listening + speaking** prototype, not just a TTS demo.

- `client.py`: Windows microphone capture (20 ms PCM frames), WebRTC voice activity detection, live listening during playback, and interruption attempts.
- `server.py`: GPU faster-whisper ASR, interim and final transcriptions, OpenAI-compatible LLM calls, sentence-chunked Edge TTS, and streaming audio messages.
- `requirements-client.txt` / `requirements-server.txt`: separate Windows and Kaggle dependencies.
- `RUNBOOK.md`: setup and verification instructions.
- `KAGGLE_VERIFY.md`: private English/Tamil voice reference preparation.
- `gpu_server.py`: older, **unverified** OmniVoice experiment; **not** used by the new listening-speaking path.

**Current TTS is Edge, not your cloned voice.** OmniVoice integration and model-specific inference verification are still pending. Do not claim that the full-duplex pipeline or voice cloning is validated on Kaggle.

Limitations: no acoustic echo cancellation (use headphones), no semantic turn-taking model, no backchannel agent, no persistent memory, no guaranteed always-on Kaggle hosting. A secure WebSocket relay is required to connect Windows to Kaggle.

Start with [RUNBOOK.md](RUNBOOK.md). Never commit tokens or reference recordings to this public repository.
