# Fix: Jarvis recognizes speech but keeps interrupting / remains silent

This specific patch addresses a real log with **37 listening turns**, many
`low_rms` rejected ASR finals, false `[barge-in]` events during Jarvis speech,
and an approximately **3,000ms** speech-end-to-first-audio-callback time.

## Changes committed

- `voice_gate.py`: WebRTC VAD must also pass a configurable normalized
  microphone RMS threshold. A stricter threshold applies while TTS plays.
- `realtime_jarvis.py`: Keeps each STT turn's original overlap/voice-duration
  state. Valid late final transcripts are no longer thrown away merely because
  the microphone started a later turn. Truly overlapping newer speech can defer
  a response; an empty next ASR result recovers the deferred answer.
- Prevents late streaming partials **after turn end** from triggering Groq prefill.
- `duplex_audio.py`: Prints actual Windows input/output device names,
  and exposes playback gain to diagnose unintended ducking.
- `realtime_voice_smoke.py --stage duplex`: tests the **same PortAudio
  full-duplex stream** the Jarvis client uses, distinct from regular playback.
- Regression tests added for noise gate and late final race. No live
  microphone/Kaggle/Groq measurement has been performed on your hardware.

## Update Windows only

Stop Jarvis with Ctrl+C, then:

```powershell
cd C:\Users\LENOVO\jarvis-realtime-voice
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install -r requirements-realtime-windows.txt
notepad .env
```

Use/adjust these local `.env` settings (keep your existing secrets intact):

```dotenv
VAD_ONSET_FRAMES=5
VAD_RMS_MIN=0.008
VAD_BARGE_RMS_MIN=0.020
VAD_BARGE_ONSET_FRAMES=7
VAD_NOISE_RATIO=2.5
ENABLE_PREFILL=0
ENABLE_BACKCHANNELS=0
TTS_PREWARM_ACKS=0
HEADPHONES=1
```

The low-energy threshold `0.008` is a **starting value**, not an empirically
calibrated setting for your microphone. If a genuine quiet voice no longer
starts a listening turn, lower `VAD_RMS_MIN` to `0.004`. If fake barge-ins
persist while wearing headphones, increase `VAD_BARGE_RMS_MIN` from
`0.020` to `0.035`, checking that genuine interruptions still register.
With laptop speakers and no acoustic echo cancellation, no RMS threshold can
reliably distinguish speaker leakage from real user speech.

**Do not restart Kaggle A/B** for this Windows-only patch; keep both tunnels
running.

## Verify that audio actually reaches the correct Windows output

**Stop the Jarvis process first** before running these direct tests:

```powershell
# Pure Windows output test (no microphone or cloud)
.\.venv\Scripts\python.exe realtime_voice_smoke.py --stage audio

# SAME duplex input/output audio stream Jarvis uses
.\.venv\Scripts\python.exe realtime_voice_smoke.py --stage duplex

# Genuine cloned voice from Kaggle B, bypassing Groq and STT
.\.venv\Scripts\python.exe realtime_voice_smoke.py --stage tts
```

If the first test is audible but `--stage duplex` is silent:

```powershell
.\.venv\Scripts\python.exe -m sounddevice
```

Find the real output and input device IDs, add e.g.
`SPEAKER_DEVICE=3`, `MIC_DEVICE=1` to `.env` (the numbers depend on
**your** Windows device list). Do not invent IDs; prefer headphone output.
Test duplex again. If both tones work but the speech test fails, check Kaggle B
server logs and saved `jarvis_tts_smoke.wav`.

## Retest a single real turn

```powershell
.\.venv\Scripts\python.exe realtime_jarvis.py
```

Say: `Hello Jarvis`, then wait 3 seconds. Expected stage order:

```text
[audio-device] Input: ...
[audio-device] Output: ...
[listening] turn=1 | overlap=False | onset_rms=... | noise_floor=...
[turn-end] silence=...ms
[final] Hello Jarvis. | ASR=...ms | lang=en
[reply-start] from STT turn 1: Hello Jarvis.
[speak 0] ...
[tts] seg=0 compute=...ms audio=...ms
[audio] first speech PCM received; output_gain=1.00
[latency] speech-end-to-audio-callback=...ms
```

If `output_gain=0.25` while no one is interrupting Jarvis, its audio is being
unintentionally ducked. If gain is 1.00 but duplex-tone is silent, select the
correct speaker output. If the final arrives but no `[reply-start]`, review
`[final-deferred]` and `[overlap]` lines.

Once this single-turn test is audible, re-enable speculative `ENABLE_PREFILL=1`
and `ENABLE_BACKCHANNELS=1` **one at a time**, testing between changes.

The observed ~3 seconds to first output have not been solved by this patch;
this patch prioritizes reliable turn-taking and audible sound. Optimize STT,
Groq and OmniVoice streaming after the speaker path is confirmed.
