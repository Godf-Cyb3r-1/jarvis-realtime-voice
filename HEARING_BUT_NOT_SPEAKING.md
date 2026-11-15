# Jarvis hears me but never speaks: 3-stage Windows check

This is for the **two-Kaggle** setup. It tests Kaggle B voice separately from Kaggle A hearing and from Groq reasoning. It prints no private API keys.

In Windows PowerShell:

```powershell
cd C:\Users\LENOVO\jarvis-realtime-voice
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install -r requirements-realtime-windows.txt

# Test speaker first
.\.venv\Scripts\python.exe realtime_voice_smoke.py --stage audio

# Test Groq independently
.\.venv\Scripts\python.exe realtime_voice_smoke.py --stage groq

# Test Kaggle B directly and listen to generated cloned speech
.\.venv\Scripts\python.exe realtime_voice_smoke.py --stage tts
```

**Interpretation:**

- The audio test makes a 440 Hz tone. If silent, run `python -m sounddevice` and check Windows sound output. Set `SPEAKER_DEVICE=NUMBER` in `.env` for the main full-duplex app. If you change the Windows output device, restart `realtime_jarvis.py`.
- The Groq test should produce a spoken-style text reply. If HTTP 401, update `GROQ_API_KEY` **locally**. If HTTP 429, the free-tier rate limit may be reached.
- The TTS test connects directly to `TTS_WS_URL` (no mic, no Groq), asks for a short sentence, reports first audio packet, saves `jarvis_tts_smoke.wav`, and plays it via the default output device.
- If TTS returns `[tts] ...` with nonzero RMS but no audible output, your sound device/routing is likely wrong. If it reports no audio, check **Kaggle B** logs: `print(SERVER_LOG.read_text(errors="replace")[-3000:])`.
- If both smoke tests work but `realtime_jarvis.py` stays silent, inspect the Windows stages **after** speaking and allowing 2–3 seconds of silence:

```text
[listening] turn=...
[turn-end] silence=...
[final] transcript | ASR=...
[speak 0] ...
[tts] seg=... compute=...
[audio] received first speech PCM from Kaggle B
[latency] speech-end-to-audio-callback=...
```

If only `[listening]` / `[partial-stable]` are present and no `[turn-end]`, the local VAD isn't observing a pause. Fix microphone selection or background noise. If `[turn-end]` shows but `[final]` doesn't, inspect Kaggle A logs. If `[final]` shows but `[speak 0]` doesn't, investigate Groq and floor policy. If `[speak 0]` shows but `[tts]` doesn't, investigate Kaggle B queue and server.

For first diagnosis, temporarily turn off competing speculative work in Windows `.env`:

```dotenv
ENABLE_PREFILL=0
ENABLE_BACKCHANNELS=0
TTS_PREWARM_ACKS=0
```

Restart Jarvis and verify ordinary speech first. Re-enable features *one at a time* after both direct tests succeed. Normal listening is not affected by disabling speculative prefill.

**Important:** This is a diagnostic tool, not proof of a working end-to-end voice system. Do not paste your `.env`, API keys, or Kaggle secrets into chat. Copy only the non-secret logs/errors.
