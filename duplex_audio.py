"""Single synchronized microphone+speaker stream with optional WebRTC AEC.

16kHz mono, 20ms blocks. Unlike sd.play(), speaker playback can be ducked
or aborted without stopping the microphone.
"""
import asyncio
from collections import deque
import os
import threading
import time
import numpy as np
import sounddevice as sd

RATE=16000
FRAME=320

def _device_from_env(name):
    """Choose PortAudio device ID or substring without changing OS defaults."""
    value=os.getenv(name,"").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return value


class FullDuplexAudio:
    def __init__(self, loop):
        self.loop=loop
        self.mic=asyncio.Queue(maxsize=200)
        self._lock=threading.Lock()
        self._pending=deque()
        self._tail=b""
        self._gain=1.0
        self._recent_audio=False
        self._first_output_time=None
        self._stream=None
        self.aec=None
        if os.getenv("AEC_ENABLED","0")=="1":
            try:
                from pywebrtc_audio import AudioProcessor
                self.aec=AudioProcessor(sample_rate=RATE,noise_suppression=True,
                    echo_cancellation=True,auto_gain_control=False,
                    stream_delay_ms=int(os.getenv("AEC_DELAY_MS","40")))
            except Exception as exc:
                raise RuntimeError("AEC_ENABLED=1 but pywebrtc-audio could not initialize") from exc

    @property
    def playing(self):
        with self._lock:
            return bool(self._pending or self._tail or self._recent_audio)

    @property
    def first_output_time(self):
        """Monotonic time of the first PCM block given to the OS audio callback."""
        with self._lock:
            return self._first_output_time

    def duck(self, gain=0.3):
        with self._lock:
            self._gain=float(gain)

    def restore(self):
        self.duck(1.0)

    def flush(self):
        with self._lock:
            self._pending.clear()
            self._tail=b""
            self._recent_audio=False
            self._first_output_time=None
            self._gain=1.0

    def enqueue(self, samples):
        pcm=np.asarray(samples,dtype=np.int16).reshape(-1).tobytes()
        with self._lock:
            self._pending.append(pcm)

    def _output(self,byte_count):
        with self._lock:
            out=bytearray()
            while len(out)<byte_count:
                if not self._tail:
                    if not self._pending:break
                    self._tail=self._pending.popleft()
                take=min(byte_count-len(out),len(self._tail))
                out.extend(self._tail[:take])
                self._tail=self._tail[take:]
            self._recent_audio=bool(out)
            if out and self._first_output_time is None:
                self._first_output_time=time.monotonic()
            gain=self._gain
        if len(out)<byte_count:out.extend(bytes(byte_count-len(out)))
        data=np.frombuffer(out,dtype="<i2").copy()
        if gain!=1.0:
            data=np.clip(data.astype(np.float32)*gain,-32768,32767).astype(np.int16)
        return data

    def _callback(self,indata,outdata,frames,timing,status):
        outgoing=self._output(frames*2)
        outdata[:]=outgoing.tobytes()
        incoming=np.frombuffer(indata,dtype="<i2").copy()
        if self.aec is not None:
            try:
                incoming=np.asarray(self.aec.process(incoming,outgoing),dtype=np.int16)
            except Exception:
                # Do not kill the audio stream if an AEC frame is invalid.
                pass
        payload=incoming.tobytes()
        def put():
            if not self.mic.full():self.mic.put_nowait(payload)
        self.loop.call_soon_threadsafe(put)

    def __enter__(self):
        mic=_device_from_env("MIC_DEVICE")
        speaker=_device_from_env("SPEAKER_DEVICE")
        device=(mic,speaker) if mic is not None or speaker is not None else None
        self._stream=sd.RawStream(samplerate=RATE,channels=(1,1),
            dtype="int16",blocksize=FRAME,callback=self._callback,
            latency="low",device=device)
        self._stream.start()
        return self

    def __exit__(self,*_):
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self.flush()
