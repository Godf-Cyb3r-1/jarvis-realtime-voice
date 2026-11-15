"""Lightweight diagnostics/filters for Whisper short-utterance hallucinations.

Only English/Tamil/Tanglish requested. Language auto-detection on isolated
short phrases can confuse Hindi/Spanish; redo unsupported detections in English
by default. Set ASR_LANGUAGE=ta when explicitly testing Tamil. Confidence
filters are tunable and can reject quiet real speech: test with recordings.
"""
import os
import numpy as np

def transcribe_audio(model, pcm, env=None):
    env = os.environ if env is None else env
    audio=np.asarray(pcm,dtype=np.float32).reshape(-1)/32768.0
    if audio.size < 16000//4:
        return "", "too_short"
    rms=float(np.sqrt(np.mean(audio*audio)))
    if rms < float(env.get("ASR_MIN_RMS","0.003")):
        return "", "low_rms"
    force=env.get("ASR_LANGUAGE","auto").lower()
    language=None if force=="auto" else force
    kwargs=dict(beam_size=int(env.get("ASR_BEAM_SIZE","3")),temperature=0.0,
                condition_on_previous_text=False,vad_filter=False,
                language=language,hotwords=env.get("ASR_HOTWORDS","Jarvis"),
                no_speech_threshold=float(env.get("ASR_NO_SPEECH_THRESHOLD","0.6")),
                log_prob_threshold=float(env.get("ASR_LOG_PROB_THRESHOLD","-1.0")))
    segments,info=model.transcribe(audio,**kwargs)
    detected=getattr(info,"language","") or ""
    allowed={x.strip() for x in env.get("ASR_ALLOWED_LANGUAGES","en,ta").split(",") if x.strip()}
    if language is None and detected not in allowed:
        # We are intentionally only handling English and Tamil in this prototype.
        # For other languages the user must configure ASR_ALLOWED_LANGUAGES.
        segments,info=model.transcribe(audio,**dict(kwargs,language="en"))
        detected="en(fallback)"
    minimum=float(env.get("ASR_MIN_AVG_LOGPROB","-1.25"))
    valid=[]
    for seg in segments:
        avg=float(getattr(seg,"avg_logprob",0))
        nospeech=float(getattr(seg,"no_speech_prob",0))
        if avg < minimum or (nospeech > 0.8 and avg < -0.65):
            continue
        piece=str(getattr(seg,"text","")).strip()
        if piece:
            valid.append(piece)
    result=" ".join(valid).strip()
    if not result:
        return "", f"rejected:{detected}"
    return result, detected
