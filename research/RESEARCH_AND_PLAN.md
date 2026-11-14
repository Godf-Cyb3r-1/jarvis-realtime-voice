# Jarvis: research-driven duplex architecture

## Definition of success
A real-time conversation interface must distinguish turns, hesitations, acknowledgements, substantive interruptions and ambient speech. An always-open microphone by itself does not prove human-like duplex performance.

## Sources and conclusions
1. [Moshi (2024)](https://arxiv.org/abs/2410.00037): parallel user and assistant audio streams and time-aligned textual inner monologue. Native duplex reference; pre-trained demo voice/model are not drop-in Tamil cloned Jarvis.
2. [Freeze-Omni (ICML 2025)](https://proceedings.mlr.press/v267/wang25aw.html): attaching speech understanding and generation to a frozen LLM is viable; motivates modular LLM.
3. [Full-Duplex-Bench (2025)](https://arxiv.org/abs/2503.04721): evaluate pause handling, backchannel, interruptions and taking the floor, not just single-turn ASR/TTS.
4. [Full-Duplex-Bench v1.5 (2025)](https://arxiv.org/abs/2507.23159): distinguish user interrupt, listener backchannel, third-party conversation and ambient audio.
5. [Speaking While Listening (EMNLP 2026)](https://arxiv.org/abs/2606.19453): Idle, Listen, Speak, Wait, Dual state model; differentiate L0 modular duplex from native L2 model.
6. [DuplexSLA (2026)](https://arxiv.org/abs/2605.20755): speech, language and action streams synchronized for tools; research direction, not deployed in this prototype.
7. [Streaming User Transcription for Duplex Models (2026)](https://arxiv.org/abs/2609.15759): incremental ASR and duplex speech must co-exist, prompting continuously updated transcripts.
8. [NemotronLabs VoiceChat (2026)](https://arxiv.org/abs/2609.21967): native duplex plus tool channel, but the 11B model and vendor reference deployment exceed the reliable free T4 configuration.
9. [Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn): audio prosody-informed turn completeness inference, CPU ONNX.
10. [OmniVoice (2026)](https://arxiv.org/abs/2604.00688): 600+ language zero-shot voice cloning; use official in-process API and cached voice prompt.
11. [WebRTC audio processor Python bindings](https://github.com/strands-labs/pywebrtc-audio): optional local AEC for barge-in without hearing self-playback.

## Design choices for Kaggle T4 x2

```text
  Windows: 16 kHz mic ─── VAD + optional WebRTC AEC ─── WSS audio ──┐
           headphones/speakers ◀── cancellable PCM playback          │
                                                                      ▼
  Kaggle:  GPU0 faster-whisper ASR ─► Smart Turn CPU ─► floor control (WAIT/LISTEN)
            (partial + final)                              │
                                                            ▼
           OpenAI-compatible streamed free/owned LLM ─► sentence queue
                                                            │
           GPU1 OmniVoice voice clone ◀─────────────────────┘
                  │
                  └── PCM 16k audio chunks ──────────► Windows
```

WebSocket uses Authorization header, not a token in query strings. Playback and listening are independent. A generation counter prevents old audio from continuing after an interruption. The local player ducks volume during an uncertain overlap; the server classifies short backchannels before yielding, while sustained talk forces a local interrupt.

## Explicit limitations
- **Engineered/modular duplex**, not a jointly trained speech-to-speech model.
- Smart Turn language generalization to Tamil/Tanglish needs measurement. Fallback is still silence-based at 1.1s/2.3s.
- Whisper sliding re-decode is not a trained native streaming recognizer, and can hallucinate/flip language.
- OmniVoice is sentence-wise synthesis, not causal waveform-token streaming; first-audio latency depends on generated phrase and GPU.
- WebRTC AEC is optional, hardware-dependent and unvalidated on the user's device.
- Overlap classification is heuristic + ASR, not native overlap intent understanding.
- Automatic backchannels are optional and simplistic, disabled by default.
- Kaggle is temporary and not an always-on server; free LLM endpoints have rate limits.
- CPU unit tests and syntax checks do not prove GPU/model/runtime quality.

## Acceptance tests (record actual timings)
1. English question: transcript WER and speech first-audio latency.
2. Tamil question: Tamil output and stable language choice.
3. Tanglish: mixed-language comprehension, pronunciation and ASR substitution.
4. Hesitation: pause mid-thought (500–900ms) must not trigger an answer.
5. Backchannel: 'hmm', 'yeah', 'ஆமா' while Jarvis speaks should continue.
6. Interruption: 'Wait, not that one' while Jarvis speaks must cancel old buffered audio.
7. Speaker echo: test both headphones and speakers; AEC optional.
8. Side conversation/music: measure false activation; current policy may fail.
9. Kaggle restart/reconnect and model quota exhaustion.
10. Measure VAD onset, SmartTurn inference, LLM first token, OmniVoice first PCM, interruption-to-silence, and speech-to-speech latency.

Release criteria: repeat tests, gather traces, compare against baseline and only then label human-like duplex.
