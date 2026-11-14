"""Optional audio-native Smart Turn v3.2 ONNX endpoint detector.

Model is downloaded from its publisher only when SMART_TURN_ENABLED=1.
BSD-2-Clause weights: pipecat-ai/smart-turn-v3.
"""
import os
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=1)
def _load():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import WhisperFeatureExtractor
    filename=os.getenv("SMART_TURN_WEIGHTS","smart-turn-v3.2-cpu.onnx")
    path=hf_hub_download(repo_id="pipecat-ai/smart-turn-v3",filename=filename)
    options=ort.SessionOptions()
    options.inter_op_num_threads=1
    options.intra_op_num_threads=2
    session=ort.InferenceSession(path,sess_options=options,providers=["CPUExecutionProvider"])
    return session, WhisperFeatureExtractor(chunk_length=8)

def completion_probability(samples: np.ndarray) -> float | None:
    if os.getenv("SMART_TURN_ENABLED","0")!="1":
        return None
    session,extractor=_load()
    audio=np.asarray(samples,dtype=np.float32)[-8*16000:]
    if not len(audio):
        return None
    if len(audio)<8*16000:
        audio=np.pad(audio,(8*16000-len(audio),0))
    features=extractor(audio,sampling_rate=16000,return_tensors="np",
        padding="max_length",max_length=8*16000,truncation=True,do_normalize=True).input_features.astype(np.float32)
    return float(session.run(None,{"input_features":features})[0].reshape(-1)[0])
