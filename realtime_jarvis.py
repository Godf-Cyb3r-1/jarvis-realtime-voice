"""Windows: full-duplex controller for two independent Kaggle WebSockets.

Mic and speaker never wait for Groq or OmniVoice. Groq API key remains LOCAL.
Speculative LLM/TTS prefill may run on stable partial text, but prefilled
audio is played ONLY if the final transcript matches exactly.
"""
import asyncio
import base64
from collections import deque
import hashlib
import json
import os
import sqlite3
import time

import numpy as np
from dotenv import load_dotenv
import webrtcvad
import websockets

from duplex_audio import FullDuplexAudio
from duplex_policy import overlap_action
from hosted_llm import SYSTEM_PROMPT, stream_reply, configuration
from realtime_protocol import (RATE, FRAME_MS, Hypothesis, normalized,
    endpoint_ready, backchannel_due, safe_to_prefill, prefill_matches, speech_chunks)

load_dotenv()

def need(name):
    value=os.getenv(name,"").strip()
    if not value:
        raise RuntimeError(f"Missing {name} in local .env")
    return value

def ws_url(name):
    value=need(name)
    if not (value.startswith("wss://") or value.startswith("ws://127.0.0.1:")):
        raise RuntimeError(f"{name} must be WSS (except 127.0.0.1 for local tests)")
    return value

class Memory:
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS conversation("
                        "id INTEGER PRIMARY KEY, role TEXT, content TEXT, timestamp REAL)")
        self.db.commit()

    def messages(self):
        rows=self.db.execute("SELECT role,content FROM "
                             "(SELECT id,role,content FROM conversation "
                             "ORDER BY id DESC LIMIT 14) ORDER BY id").fetchall()
        return [{"role":"system","content":SYSTEM_PROMPT+
                 " You can pause naturally; no repeated 'ayyo' or role labels."}]+[
                 {"role":r,"content":t} for r,t in rows]

    def add(self,role,text):
        self.db.execute("INSERT INTO conversation(role,content,timestamp) VALUES(?,?,?)",
                        (role,text,time.time()))
        self.db.commit()

class Jarvis:
    def __init__(self,stt,tts,loop):
        self.stt=stt
        self.tts=tts
        self.audio=FullDuplexAudio(loop)
        self.vad=webrtcvad.Vad(int(os.getenv("VAD_MODE","2")))
        self.stt_lock=asyncio.Lock()
        self.tts_lock=asyncio.Lock()
        self.memory=Memory(os.getenv("JARVIS_MEMORY_DB","jarvis_conversation.sqlite3"))
        self.turn=0
        self.capturing=False
        self.vad_run=0
        self.silence_ms=0
        self.voice_ms=0
        self.probed=False
        self.probability=None
        self.recent=deque(maxlen=12)
        self.stable=""
        self.overlap=False
        self.yielded=False
        self.last_ack=-1000.0
        self.response_task=None
        self.active_audio_turn=-1
        self.tts_pending=0
        self.turn_end_at={}
        self.output_measured=set()
        self.spec_task=None
        self.spec_queue=None
        self.spec_text=""
        self.spec_turn=-1
        self.generation=0
        self.heard_since=0.0
        self.enabled_prefill=os.getenv("ENABLE_PREFILL","1")=="1"
        self.enabled_ack=os.getenv("ENABLE_BACKCHANNELS","1")=="1"
        self.headphones=os.getenv("HEADPHONES","1")=="1"

    async def to_stt(self,kind,**fields):
        async with self.stt_lock:
            await self.stt.send(json.dumps({"type":kind,**fields}))

    async def to_tts(self,kind,**fields):
        async with self.tts_lock:
            await self.tts.send(json.dumps({"type":kind,**fields}))

    async def cancel_reply(self):
        self.generation+=1
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
        self.response_task=None
        old=self.active_audio_turn
        self.active_audio_turn=-1
        self.tts_pending=0
        self.audio.flush()
        if old>=0:
            await self.to_tts("tts_cancel",turn=old)

    async def start_utterance(self):
        self.turn+=1
        self.capturing=True
        self.silence_ms=0
        self.voice_ms=0
        self.heard_since=time.monotonic()
        self.probed=False
        self.probability=None
        self.stable=""
        self.overlap=(self.audio.playing or self.tts_pending>0 or
                      bool(self.response_task and not self.response_task.done()))
        self.yielded=False
        self.discard_spec()
        await self.to_stt("start",turn=self.turn)
        for frame in self.recent:
            await self.audio_frame(frame)
        if self.overlap:
            self.audio.duck(float(os.getenv("OVERLAP_DUCK_GAIN","0.25")))
        print(f"\n[listening] turn={self.turn} | overlap={self.overlap}")

    def discard_spec(self):
        if self.spec_task and not self.spec_task.done():
            self.spec_task.cancel()
        self.spec_task=None
        self.spec_queue=None
        self.spec_text=""
        self.spec_turn=-1

    async def audio_frame(self,frame):
        await self.to_stt("audio",turn=self.turn,
                          pcm16=base64.b64encode(frame).decode("ascii"))

    async def microphone(self):
        onset=int(os.getenv("VAD_ONSET_FRAMES","4"))
        while True:
            frame=await self.audio.mic.get()
            if len(frame)!=RATE*FRAME_MS//1000*2:
                continue
            voiced=self.vad.is_speech(frame,RATE)
            self.recent.append(frame)
            if not self.capturing:
                self.vad_run=self.vad_run+1 if voiced else 0
                if self.vad_run>=onset:
                    self.vad_run=0
                    await self.start_utterance()
                continue
            await self.audio_frame(frame)
            if voiced:
                self.silence_ms=0
                self.voice_ms+=FRAME_MS
                self.probed=False
                self.probability=None
                if self.overlap and not self.yielded and self.voice_ms>=int(
                        os.getenv("FORCE_BARGE_MS","850")):
                    self.yielded=True
                    await self.cancel_reply()
                    print("[barge-in] yielded floor to user")
            else:
                self.silence_ms+=FRAME_MS
                if self.silence_ms>=320 and not self.probed:
                    self.probed=True
                    await self.to_stt("pause_probe",turn=self.turn)
                if backchannel_due(self.voice_ms/1000,self.silence_ms,
                    time.monotonic()-self.last_ack,self.stable,
                    enabled=self.enabled_ack and self.headphones and not self.overlap):
                    self.last_ack=time.monotonic()
                    ack="சரி." if any("\u0b80"<=c<="\u0bff" for c in self.stable) else "Mm-hmm."
                    ack_turn=self.turn*10
                    self.active_audio_turn=ack_turn
                    self.tts_pending+=1
                    await self.to_tts("tts_speak",turn=ack_turn,segment=-1,
                                      language="ta" if ack=="சரி." else "en",text=ack,
                                      cache_key="ack-ta" if ack=="சரி." else "ack-en")
                    print("[backchannel]",ack)
                if endpoint_ready(self.silence_ms,self.stable,self.probability):
                    self.capturing=False
                    self.turn_end_at[self.turn]=time.monotonic()
                    if len(self.turn_end_at)>50:
                        self.turn_end_at.clear()
                    await self.to_stt("end",turn=self.turn)
                    print(f"[turn-end] silence={self.silence_ms}ms | stable={self.stable!r}")
                    self.recent.clear()

    async def listen_stt(self):
        async for msg in self.stt:
            e=json.loads(msg)
            kind=e.get("type")
            turn=int(e.get("turn",-1))
            if kind=="ready":
                print("[stt] connected:",e.get("rate"),"Hz")
            elif kind=="partial" and turn==self.turn:
                stable=e.get("stable","")
                if stable!=self.stable:
                    self.stable=stable
                    if stable:
                        print("[partial-stable]",stable)
                    if self.spec_task is not None and not normalized(stable).startswith(
                            normalized(self.spec_text)):
                        # Recognition changed its mind: invalidate speculation.
                        self.discard_spec()
                    if (stable and self.enabled_prefill and not self.overlap and
                            self.spec_task is None and safe_to_prefill(stable)):
                        self.spec_turn=turn
                        self.spec_text=stable
                        self.spec_queue=asyncio.Queue(maxsize=512)
                        self.spec_task=asyncio.create_task(self.speculate(turn,stable,self.spec_queue))
            elif kind=="turn_probability" and turn==self.turn:
                self.probability=e.get("probability")
            elif kind=="final":
                print(f"[final] {e.get('text','')} | ASR={e.get('asr_ms')}ms | lang={e.get('language')}")
                # Preserve final events even if microphone has started a later turn.
                task=asyncio.create_task(self.handle_final(turn,e.get("text","").strip()))
                task.add_done_callback(lambda t: print("[final-error]",t.exception())
                                       if not t.cancelled() and t.exception() else None)
            elif kind in ("error","warning"):
                print("[stt-"+kind+"]",e.get("message"))
        raise ConnectionError("STT WebSocket disconnected")

    async def speculate(self,turn,stable,q):
        """Warm Groq and possibly first TTS phrase while user is still talking."""
        first=""
        all_text=""
        try:
            async for delta in stream_reply(self.memory.messages()+[
                    {"role":"user","content":stable}]):
                if turn!=self.turn:
                    return
                await q.put(delta)
                all_text+=delta
                if not first:
                    ready,_=speech_chunks(all_text)
                    if ready:
                        first=ready[0]
                        key=self.cache_key(turn,first)
                        await self.to_tts("tts_prefill",cache_key=key,
                                          text=first,language=self.language(first))
            await q.put(None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print("[prefill] failed, normal response will still run:",str(exc)[:130])
            await q.put(None)

    @staticmethod
    def language(text):
        return "ta" if any("\u0b80"<=ch<="\u0bff" for ch in text) else "en"

    @staticmethod
    def cache_key(turn,phrase):
        return hashlib.sha256(f"{turn}|{phrase}".encode("utf-8")).hexdigest()[:32]

    async def handle_final(self,turn,text):
        if turn>self.turn or not text:
            if not text:
                print("[asr] ignored empty/uncertain speech")
            self.audio.restore()
            return
        if self.overlap and not self.yielded:
            decision=overlap_action(text,self.voice_ms)
            if decision=="CONTINUE":
                self.audio.restore()
                print("[overlap] listener acknowledgment; continuing Jarvis")
                self.discard_spec()
                return
        # Don't let an older final cancel a newer reply.
        if turn<self.turn:
            return
        await self.cancel_reply()
        reply_turn=turn*10+1
        self.active_audio_turn=reply_turn
        usable=(self.spec_task is not None and self.spec_queue is not None
                and self.spec_turn==turn and prefill_matches(self.spec_text,text))
        q=self.spec_queue if usable else None
        if not usable:
            self.discard_spec()
        else:
            print("[prefill] confirmed exact transcript; reusing Groq stream")
            # Speculative producer is intentionally allowed to continue.
        self.memory.add("user",text)
        self.response_task=asyncio.create_task(self.answer(reply_turn,text,q))
        self.audio.restore()

    async def answer(self,reply_turn,text,queue):
        version=self.generation
        first=time.monotonic()
        parts=[]
        buffer=""
        segment=0
        produced=None
        try:
            if queue is None:
                produced=self.stream_fresh(text)
            else:
                produced=self.stream_prefilled(queue)
            async for delta in produced:
                if version!=self.generation:
                    return
                parts.append(delta)
                ready,buffer=speech_chunks(buffer+delta)
                for phrase in ready:
                    if version!=self.generation:
                        return
                    await self.speak(reply_turn,segment,phrase)
                    segment+=1
            ready,_=speech_chunks(buffer,force=True)
            for phrase in ready:
                if version!=self.generation:
                    return
                await self.speak(reply_turn,segment,phrase)
                segment+=1
            if parts and version==self.generation:
                answer="".join(parts).strip()
                self.memory.add("assistant",answer)
                print(f"[groq] queued {segment} phrases in {int((time.monotonic()-first)*1000)}ms")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print("[groq-error]",str(exc)[:350])

    async def stream_fresh(self,text):
        async for delta in stream_reply(self.memory.messages()):
            yield delta

    async def stream_prefilled(self,q):
        while True:
            delta=await q.get()
            if delta is None:
                break
            yield delta

    async def speak(self,reply_turn,segment,text):
        print(f"[speak {segment}]",text)
        self.tts_pending+=1
        await self.to_tts("tts_speak",turn=reply_turn,segment=segment,
                          text=text,language=self.language(text),
                          cache_key=self.cache_key(reply_turn//10,text))

    async def measure_first_output(self,turn):
        """Callback-level first output, not a server-only packet metric."""
        user_turn=turn//10
        ended=self.turn_end_at.get(user_turn)
        if ended is None:
            return
        for _ in range(50):
            if turn!=self.active_audio_turn:
                return
            first=self.audio.first_output_time
            if first is not None:
                print(f"[latency] speech-end-to-audio-callback={round((first-ended)*1000)}ms")
                return
            await asyncio.sleep(0.02)

    async def listen_tts(self):
        async for raw in self.tts:
            e=json.loads(raw)
            kind=e.get("type")
            if kind=="ready":
                print("[tts] connected:",e.get("rate"),"Hz")
                await self.to_tts("tts_prefill",cache_key="ack-en",text="Mm-hmm.",language="en")
            elif kind=="tts_audio":
                if int(e.get("turn",-1))!=self.active_audio_turn:
                    continue
                pcm=np.frombuffer(base64.b64decode(e["pcm16"]),dtype="<i2").copy()
                self.audio.enqueue(pcm)
                received=int(e["turn"])
                if received%10==1 and received not in self.output_measured:
                    self.output_measured.add(received)
                    asyncio.create_task(self.measure_first_output(received))
            elif kind=="tts_info":
                if int(e.get("turn",-1))==self.active_audio_turn:
                    print(f"[tts] seg={e.get('segment')} compute={e.get('synthesis_ms')}ms "
                          f"audio={e.get('duration_ms')}ms")
            elif kind in ("error","warning"):
                print("[tts-"+kind+"]",e.get("message"))
            elif kind=="prefill_ready":
                key=e.get("cache_key","")
                if key=="ack-en":
                    await self.to_tts("tts_prefill",cache_key="ack-ta",text="சரி.",language="ta")
                elif key=="ack-ta":
                    print("[prefill] both short acknowledgments cached")
                else:
                    print("[prefill] speculative first clause prepared")
            elif kind=="tts_done":
                if int(e.get("turn",-1))==self.active_audio_turn:
                    self.tts_pending=max(0,self.tts_pending-1)
        raise ConnectionError("TTS WebSocket disconnected")

    async def run(self):
        print("[Jarvis] continuous ear + local Groq + remote cloned voice")
        print("[Jarvis] Headphones strongly recommended; Ctrl+C to stop.")
        try:
            with self.audio:
                # Structured concurrency cancels microphone/other socket tasks
                # if either remote Kaggle service disconnects.
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(self.microphone())
                    tasks.create_task(self.listen_stt())
                    tasks.create_task(self.listen_tts())
        finally:
            self.memory.db.close()

async def main():
    # Fail immediately if no local Groq key. Re-read .env on reconnect so
    # updated temporary Kaggle tunnel URLs take effect without restarting Jarvis.
    configuration("groq")
    attempt=0
    while True:
        load_dotenv(override=True)
        stt_url=ws_url("STT_WS_URL")
        tts_url=ws_url("TTS_WS_URL")
        try:
            async with (websockets.connect(stt_url,additional_headers={
                            "Authorization":"Bearer "+need("STT_WS_TOKEN")},
                            max_size=8*1024*1024,ping_interval=20,ping_timeout=30) as stt,
                        websockets.connect(tts_url,additional_headers={
                            "Authorization":"Bearer "+need("TTS_WS_TOKEN")},
                            max_size=16*1024*1024,ping_interval=20,ping_timeout=30) as tts):
                print("[network] Both Kaggle sessions connected")
                attempt=0
                jarvis=Jarvis(stt,tts,asyncio.get_running_loop())
                await jarvis.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            attempt+=1
            delay=min(2**min(attempt,4),20)
            print(f"[network] {type(exc).__name__}: {str(exc)[:190]}")
            print(f"[network] Retrying in {delay}s; update .env if Kaggle restarted")
            await asyncio.sleep(delay)

if __name__=="__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\nJarvis stopped.")
