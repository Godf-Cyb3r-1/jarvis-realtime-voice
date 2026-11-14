# Setup and verification

This is an experimental integration, **not a tested full-duplex voice assistant**. OmniVoice's Python interface in gpu_server.py must be checked against the installed upstream release before running.

## Kaggle / Colab
1. Enable a CUDA GPU and upload the project with your reference WAV and transcript files.
2. Install OmniVoice following its upstream repository's instructions, then install fastapi uvicorn numpy soundfile.
3. Set VOICE_API_TOKEN to a long random secret.
4. Start: uvicorn gpu_server:app --host 127.0.0.1 --port 8091
5. Test locally within the notebook using an authenticated POST to /synthesize.
6. Use a permitted private HTTPS tunnel or authenticated relay to reach the notebook from Windows.

## Windows
1. Install Python 3.11, create a virtual environment, and pip install -r requirements.txt.
2. Copy .env.example to .env; set VOICE_API_URL and VOICE_API_TOKEN.
3. Run python voice_client.py.

## Limitations
- No GPU is required on Windows.
- The cloud endpoint is not yet deployed.
- This uses full-utterance synthesis, not low-latency incremental TTS.
- No ASR, semantic turn-taking, or interruption handling is wired into this GitHub version yet.
- The English and Tamil voice reference assets must be added before synthesis.
