"""Noise-aware local speech-onset filter for full-duplex microphone capture.

WebRTC VAD sometimes identifies quiet device noise or leaked speaker audio as
speech. Combine VAD with PCM energy and slightly stricter barge-in criteria.
No model calls or network dependencies; levels are configurable per microphone.
"""
import os
import numpy as np

def pcm_rms(frame: bytes) -> float:
    """Normalized RMS of 16-bit mono PCM. Range 0..1."""
    if not frame or len(frame) % 2:
        return 0.0
    x = np.frombuffer(frame, dtype="<i2").astype(np.float32)
    return float(np.sqrt(np.mean(x*x, dtype=np.float64)) / 32768.0) if len(x) else 0.0

class SpeechGate:
    """Per-frame noise floor, gating and onset for WebRTC VAD.

    No long suppression windows: a genuine voice can interrupt as soon as it
    crosses the sustained onset threshold. With speakers and no AEC, no
    acoustic-energy-only gate can reliably distinguish echo from the user.
    """
    def __init__(self, env=None):
        env=os.environ if env is None else env
        self.min_rms=float(env.get("VAD_RMS_MIN","0.008"))
        self.barge_min_rms=float(env.get("VAD_BARGE_RMS_MIN","0.020"))
        self.onset_frames=max(1,int(env.get("VAD_ONSET_FRAMES","5")))
        self.barge_onset_frames=max(1,int(env.get("VAD_BARGE_ONSET_FRAMES","7")))
        self.noise_floor=0.001
        self._onset=0
        self.noise_ratio=float(env.get("VAD_NOISE_RATIO","2.5"))
        self.last_rms=0.0

    def accept(self, frame: bytes, vad_speech: bool, *, bot_active: bool=False) -> bool:
        rms=pcm_rms(frame)
        self.last_rms=rms
        # Slowly update an estimate from frames that WebRTC labels nonspeech.
        if not vad_speech:
            self.noise_floor=0.98*self.noise_floor+0.02*rms
        threshold=max(self.barge_min_rms if bot_active else self.min_rms,
                      self.noise_floor*self.noise_ratio)
        return bool(vad_speech and rms>=threshold)

    def start_ready(self,voice: bool, *, bot_active: bool=False) -> bool:
        self._onset=(self._onset+1) if voice else 0
        required=self.barge_onset_frames if bot_active else self.onset_frames
        if self._onset>=required:
            self._onset=0
            return True
        return False

    def reset_onset(self):
        self._onset=0
