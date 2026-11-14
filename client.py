"""Windows continuous listening, VAD, playback, interruption. Headphones required."""
import asyncio,base64,json,os,queue,threading,time
from urllib.parse import quote
import numpy as np,sounddevice as sd,webrtcvad,websockets
from dotenv import load_dotenv
load_dotenv()
RATE=16000;MS=20;N=320
class Speaker:
    def __init__(self):
        self.q=queue.Queue();self.version=0;self.playing=threading.Event()
        threading.Thread(target=self.run,daemon=True).start()
    def stop(self):
        self.version+=1;sd.stop();self.playing.clear()
        while not self.q.empty():
            try:self.q.get_nowait()
            except queue.Empty:break
    def run(self):
        while True:
            v,a,rate=self.q.get()
            if v!=self.version:continue
            self.playing.set()
            try:sd.play(a,rate,blocking=True)
            except Exception as e:print("Playback:",e)
            finally:self.playing.clear()
async def main():
    url=os.getenv("VOICE_WS_URL","");token=os.getenv("VOICE_API_TOKEN","")
    if not url or not token:raise RuntimeError("Configure VOICE_WS_URL and VOICE_API_TOKEN")
    if not(url.startswith("wss://") or url.startswith("ws://127.0.0.1:")):raise RuntimeError("Use WSS for remote audio")
    url+=("&" if "?" in url else "?")+"token="+quote(token,safe="")
    frames=asyncio.Queue(maxsize=500);loop=asyncio.get_running_loop()
    vad=webrtcvad.Vad(int(os.getenv("VAD_MODE","2")))
    speaker=Speaker()
    def callback(data,count,tm,status):
        frame=bytes(data)
        def push():
            if not frames.full():frames.put_nowait(frame)
        loop.call_soon_threadsafe(push)
    async with websockets.connect(url,max_size=16*1024*1024) as ws:
        async def receive():
            async for raw in ws:
                msg=json.loads(raw);typ=msg.get("type")
                if typ in ("partial","transcript","reply"):print(f"[{typ}]",msg["text"])
                elif typ=="audio":
                    a=np.frombuffer(base64.b64decode(msg["pcm_f32"]),dtype=np.float32).copy()
                    speaker.q.put((speaker.version,a,int(msg["rate"])))
                elif typ=="error":print("[error]",msg.get("message"))
                elif typ=="ready":print("Connected. Speak naturally; Ctrl+C to exit.")
        reader=asyncio.create_task(receive())
        speaking=False;voiced=0;silence=0;preroll=[];last=time.monotonic()
        try:
            with sd.RawInputStream(samplerate=RATE,channels=1,dtype="int16",blocksize=N,callback=callback):
                while True:
                    frame=await frames.get()
                    active=vad.is_speech(frame,RATE)
                    if not speaking:
                        preroll=(preroll+[frame])[-15:]
                        voiced=voiced+1 if active else max(0,voiced-1)
                        if voiced>=3:
                            if speaker.playing.is_set():
                                speaker.stop();await ws.send(json.dumps({"type":"interrupt"}))
                            await ws.send(json.dumps({"type":"start"}))
                            for x in preroll:
                                await ws.send(json.dumps({"type":"audio","pcm16":base64.b64encode(x).decode()}))
                            speaking=True;silence=0;last=time.monotonic();preroll=[]
                    else:
                        await ws.send(json.dumps({"type":"audio","pcm16":base64.b64encode(frame).decode()}))
                        silence=0 if active else silence+MS
                        if time.monotonic()-last>float(os.getenv("PARTIAL_INTERVAL","1.4")):
                            await ws.send(json.dumps({"type":"partial"}));last=time.monotonic()
                        if silence>=int(os.getenv("END_SILENCE_MS","900")):
                            await ws.send(json.dumps({"type":"end"}))
                            speaking=False;voiced=0;preroll=[]
        finally:
            reader.cancel();speaker.stop()
if __name__=="__main__":
    try:asyncio.run(main())
    except KeyboardInterrupt:print("Stopped")
