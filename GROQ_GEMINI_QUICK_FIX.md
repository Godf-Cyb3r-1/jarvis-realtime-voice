# Temporary free hosted brain: Groq or Gemini (Nov 2026)

**Why:** Your Windows logs show local Qwen3-0.6B repeating the microphone transcript ("Hello!" to "Hello", "What the fuck?" to "What the fuck?"), wrong Whisper short-utterance language guesses (Spanish/Hindi), and first PCM arriving about **1.1–1.6 seconds after server commit for English and up to ~5 seconds for Tamil**. The true delay from you stopping speaking to audible audio is longer because of turn detection, network, and playback.

## Updated implementation

- `hosted_llm.py` streams *actual conversational replies*, not a local Qwen model. Provider choice controlled with `LLM_BACKEND=groq|gemini`. Models as of November 2026: Groq `openai/gpt-oss-20b` with low reasoning effort, Gemini `gemini-3.5-flash-lite`. Both may have free-tier usage limits.
- `asr_quality.py` detects unsupported Whisper language guesses and retries English for the user-selected English/Tamil use case. `ASR_HOTWORDS=Jarvis`; confidence/no-speech filter rejects some hallucinations. It cannot recover words that were never audible.
- Whisper Medium on GPU0, OmniVoice on GPU1; the hosted LLM needs no GPU memory.
- Faster (850 ms) fallback turn ending; turn-quality trade-off. First TTS chunk is produced from shorter streamed clauses; OmniVoice step count can be tuned from 8 to 12.
- New metrics `llm_first_token_after_commit_ms`, `tts_generate_ms`, and `first_pcm_after_commit_ms`. PCM is delivered closer to playback pace to limit queue buildup.

## Exactly what to do

1. **On Windows**, Ctrl+C to stop Jarvis.
2. **Create a free API key** using [Groq Console](https://console.groq.com/keys), or [Google AI Studio](https://aistudio.google.com/apikey) if you prefer Gemini.
3. In **Kaggle**, Add-ons → Secrets: add **`GROQ_API_KEY`** (or **`GEMINI_API_KEY`**) and enable notebook access. Keep your private **`VOICE_API_TOKEN`** secret.
4. **Restart Kaggle session** to free the old Whisper/Qwen/OmniVoice CUDA processes and stop the old server/tunnel.
5. Open the updated notebook: [notebooks/Jarvis_Kaggle_RUN_IN_ORDER.ipynb](notebooks/Jarvis_Kaggle_RUN_IN_ORDER.ipynb). Use Kaggle's Import Notebook to import the downloaded file if needed. Attach the same existing **private** reference-voice dataset. Run **all cells top-to-bottom**, and stop on any error.
6. Notebook Step 5 defaults to **`HOSTED_BACKEND="groq"`**. To use Gemini, change it to **`"gemini"`** and supply the Gemini key. No local LLM is loaded.
7. Verify `/debug/llm` returns a meaningful response and `/debug/tts` plays both English/Tamil. Only then start the tunnel and copy its new `wss://` URL.
8. **Windows PowerShell**:

```powershell
cd C:\Users\LENOVO\jarvis-realtime-voice
git pull --ff-only
notepad .env
# Update VOICE_WS_URL if the Kaggle tunnel changed.
# Keep VOICE_API_TOKEN identical to Kaggle's secret.
.\.venv\Scripts\python.exe duplex_client.py
```

**Do NOT enter Groq/Gemini keys into Windows `.env`.** The hosted model is called by Kaggle only.

## If speech recognition is wrong

- **English-only trial**: in Notebook Step 5 set `"ASR_LANGUAGE": "en"`, restart server (do not start a second one).
- **Tamil-only trial**: use `"ASR_LANGUAGE": "ta"`.
- **English + Tamil/Tanglish**: `"ASR_LANGUAGE": "auto"`, `"ASR_ALLOWED_LANGUAGES": "en,ta"`. This reduces accidental Hindi/Spanish decoding, but short/code-mixed phrases may still be wrong.
- Increase `ASR_BEAM_SIZE` or model size if accuracy is worse, but both cost latency.
- Use headphones: speaker audio reaching the microphone causes false-triggered transcripts.

## Latency interpretation

`first_pcm_after_commit_ms` starts **after** the server decides the turn is finished, and stops when it sends the *first audio packet*. It **does not** measure the full conversational response time from mouth-to-ear. Check `llm_first_token_after_commit_ms` and `tts_generate_ms` to separate LLM and OmniVoice costs. OmniVoice is still sentence/phrase synthesis rather than native token-streamed TTS; the result is not yet human-level full duplex. Slashing the turn pause can cause premature cutoffs; 850 ms is a tuning compromise, not an accuracy guarantee.

If Groq returns 401/429, check key/quota or switch Notebook Step 5 to Gemini. If Gemini returns 429, check free-tier rate limit. Neither key should be posted in screenshots or GitHub.
