"""Local Windows playback client for GPU-hosted OmniVoice."""
import io, os, re
import httpx
import soundfile as sf
import sounddevice as sd
from dotenv import load_dotenv
load_dotenv()

def speak(text: str, language: str | None = None):
    endpoint=os.environ["VOICE_API_URL"].rstrip("/")
    token=os.environ["VOICE_API_TOKEN"]
    if not (endpoint.startswith("https://") or endpoint.startswith("http://127.0.0.1:")):
        raise ValueError("Use HTTPS for remote voice synthesis")
    language=language or ("ta" if re.search(r"[\u0B80-\u0BFF]", text) else "en")
    with httpx.Client(timeout=180) as client:
        response=client.post(endpoint+"/synthesize",headers={"Authorization":"Bearer "+token},json={"text":text,"language":language})
        response.raise_for_status()
    audio, rate=sf.read(io.BytesIO(response.content),dtype="float32")
    sd.play(audio,rate,blocking=True)

if __name__=="__main__":
    speak("Hello, I am Jarvis. This is a reference voice test.")
