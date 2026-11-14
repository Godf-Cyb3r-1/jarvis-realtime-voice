# Kaggle verification of reference assets and GPU service

**Current status:** GPU inference API in `gpu_server.py` is not yet validated against the installed OmniVoice package. Do not assume it will work unchanged.

1. Create Kaggle notebook; select GPU; upload the original `voices.zip` as a private notebook input.
2. Check GPU:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
```

3. Extract the original files and convert to the filenames expected by the service:

```python
from pathlib import Path
import zipfile, subprocess
zip_path=next(Path("/kaggle/input").rglob("voices.zip"))
root=Path("/kaggle/working/jarvis")
(root/"voices").mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
    for lang,audio,text in [
        ("en","jarvis_en/new-ref.mp3","jarvis_en/ref.txt"),
        ("ta","jarvis_ta/ref.mp3","jarvis_ta/ref.txt")]:
        src=root/"voices"/f"{lang}.mp3"
        src.write_bytes(z.read(audio))
        (root/"voices"/f"{lang}.txt").write_bytes(z.read(text))
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(src),
                        "-ac","1","-ar","24000",str(root/"voices"/f"{lang}.wav")],check=True)
print([p.name for p in (root/"voices").iterdir()])
```

4. Verify audio and transcript:

```python
import soundfile as sf
for lang in ("en","ta"):
    audio,sr=sf.read(str(root/"voices"/f"{lang}.wav"))
    print(lang,sr,round(len(audio)/sr,2),"seconds")
    print((root/"voices"/f"{lang}.txt").read_text())
```

5. Install OmniVoice using its actual upstream instructions. **Check model class, checkpoint, generation parameters, sample rate and dependencies** before attempting `gpu_server.py`. The current server contains an unverified model interface.
6. Once verified, run the server and send an authenticated POST to `http://127.0.0.1:8091/synthesize` inside the notebook. Do not expose it publicly. A separate encrypted relay is needed for Windows playback.

**Privacy:** Never commit the voice recordings or API token to this public repository.
