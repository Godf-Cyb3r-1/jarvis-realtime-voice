"""Deterministic speech-floor policy; no model/GPU dependencies."""
from dataclasses import dataclass
import re
import os

BACKCHANNELS = {
    "hmm", "hm", "mm", "mhm", "mmhm", "uh huh", "uh-huh", "aha",
    "yeah", "yep", "okay", "ok", "right", "sure", "yes", "uh", "ah",
    "ஆமா", "ஆமாம்", "சரி", "ம்ம்ம்", "அப்படியா", "seri", "aama",
    "ama", "correct", "got it",
}
YIELD_COMMANDS = (
    "stop", "wait", "hold on", "hang on", "pause", "listen",
    "no no", "not that", "one second", "actually no",
    "நிறுத்து", "பொறு", "ஒரு நிமிஷம்", "இல்ல", "venam", "niruthu",
    "illa", "oru nimisham", "wait wait",
)

def normalized(text: str) -> str:
    return re.sub(r"[^\w\s\u0b80-\u0bff-]", "", text.casefold()).strip()

def overlap_action(text: str, duration_ms: int) -> str:
    """CONTINUE for listener signals, YIELD for substantive user speech."""
    t = normalized(text)
    if any(t == term or t.startswith(term + " ") for term in YIELD_COMMANDS):
        return "YIELD"
    if t in BACKCHANNELS and duration_ms <= 1500:
        return "CONTINUE"
    if not t:
        return "HOLD"
    if len(t.split()) >= 3 or duration_ms >= 850:
        return "YIELD"
    # Short acknowledgement while Jarvis speaks is not an interruption.
    return "CONTINUE"

@dataclass(frozen=True)
class TurnSettings:
    min_pause_ms: int = 300
    timeout_ms: int = 2300
    completion_threshold: float = 0.56

def should_finish(silence_ms: int, probability: float | None, settings: TurnSettings = TurnSettings()) -> bool:
    """Smart Turn only influences the middle of the pause window."""
    if silence_ms < settings.min_pause_ms:
        return False
    if silence_ms >= settings.timeout_ms:
        return True
    if probability is None:
        return silence_ms >= int(os.getenv("ENDPOINT_FALLBACK_MS","850"))
    return probability >= settings.completion_threshold

def split_ready(buffer: str, force: bool = False) -> tuple[list[str], str]:
    """Flush a speakable clause as soon as punctuation finishes it."""
    chunks = []
    while True:
        m = re.search(r"[.!?।！？](?:\s+|$)", buffer)
        if m:
            cut = m.end()
        elif len(buffer) >= 58:
            cut = max(buffer.rfind(",", 20, 58), buffer.rfind(" ", 38, 58))
            if cut < 0:
                break
            cut += 1
        else:
            break
        candidate = buffer[:cut].strip()
        if candidate:
            chunks.append(candidate)
        buffer = buffer[cut:]
    if force and buffer.strip():
        chunks.append(buffer.strip())
        buffer = ""
    return chunks, buffer
