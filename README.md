# Jarvis Realtime Voice — Full-Duplex Research Prototype

**Current version:** concurrent listening and speaking implemented, research-driven upgrades in progress. This is an engineered L0 duplex assistant, **not** a trained native full-duplex speech model and **not** yet validated end-to-end on Kaggle or Windows.

## Start here
- **[KAGGLE_FULL_DUPLEX.md](KAGGLE_FULL_DUPLEX.md)** — detailed GPU + Windows setup from scratch, using your private English and Tamil references.
- **[research/RESEARCH_AND_PLAN.md](research/RESEARCH_AND_PLAN.md)** — papers, architecture choices, human-like behavior test plan.
- **[tests/test_duplex_policy.py](tests/test_duplex_policy.py)** — offline tests for natural pauses, listener backchannels and genuine interruptions.

## New modules (use these, not old client.py/server.py)

| File | Purpose |
|---|---|
| `duplex_client.py` | Windows live audio capture and playback, VAD, barge-in and overlap tracking |
| `duplex_audio.py` | Single full-duplex input/output device, optional WebRTC acoustic echo cancellation |
| `duplex_server.py` | Kaggle ASR and semantic endpoint inference, token-streamed LLM, TTS speech streams |
| `duplex_policy.py` | Distinguish acknowledgements, actual interruptions, pauses and turn endings |
| `smart_turn_adapter.py` | Optional open Smart Turn v3.2 ONNX |
| `tts_engine.py` | Official in-process OmniVoice cloning + explicit Edge TTS alternative |
| `requirements-duplex-*.txt` | Separate client and GPU-server dependencies |

Old `client.py`, `server.py` and `gpu_server.py` are superseded prototypes and should not be used for the human-like duplex setup.

## Free stack
- **ASR:** faster-whisper on Kaggle GPU0. English/Tamil/Tanglish accuracy needs live evaluation.
- **Turn detection:** Smart Turn v3.2 CPU ONNX, with conservative pause fallback.
- **Voice:** OmniVoice zero-shot cloned English/Tamil on GPU1, reference prompts loaded privately from Kaggle.
- **LLM:** OpenAI-compatible model endpoint (for example OpenRouter free tier, rate limited).
- **Audio:** local Windows WebRTC VAD, optional `pywebrtc-audio` AEC.
- **Transport:** WSS, per-session bearer token in Authorization header, optional Cloudflare development tunnel.

## Truth-in-testing
Unit tests and Python syntax checks are not a substitute for Kaggle T4 inference, Tamil QA, an actual microphone-to-speaker call, echo-cancellation benchmarks and user barge-in timing. See acceptance criteria before calling this human-like.
