"""Protocol and turn-taking policy for the distributed Jarvis research build.

Shared by Windows, Kaggle STT and Kaggle TTS. Pure Python, no GPU deps.
Source: WhisperStreaming local agreement, Smart Turn, full-duplex dialogue work.
"""
import re
from dataclasses import dataclass

RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = RATE * FRAME_MS // 1000
MAX_AUDIO_BYTES = FRAME_SAMPLES * 2 * 10  # 200ms max frame packet

def normalized(text: str) -> str:
    return re.sub(r"[^\w\s\u0b80-\u0bff]", "", text.casefold()).strip()

def common_prefix_words(a: str, b: str) -> str:
    """LocalAgreement-2 approximation: stable matching words across passes."""
    aa, bb = a.split(), b.split()
    common = []
    for x, y in zip(aa, bb):
        if normalized(x) != normalized(y):
            break
        common.append(y)
    # Last word may still be unfinished, so hold one word unless terminal.
    if common and not re.search(r"[.!?\u0964]$", common[-1]):
        common = common[:-1]
    return " ".join(common).strip()

@dataclass
class Hypothesis:
    text: str = ""
    stable: str = ""
    revision: int = 0

    def observe(self, text: str) -> tuple[str, bool]:
        self.revision += 1
        stable = common_prefix_words(self.text, text) if self.text else ""
        changed = normalized(stable) != normalized(self.stable)
        # An ASR revision can invalidate previously stable words. Propagate
        # the empty/reduced prefix; never keep a contradictory old hypothesis.
        self.stable = stable
        self.text = text
        return self.stable, changed

INCOMPLETE_ENDINGS = (
    "and", "or", "but", "because", "when", "if", "so", "then",
    "the", "a", "to", "that", "en", "அப்புறம்", "ஆனா", "ஏன்னா",
    "இன்னும்", "அதுக்கு", "அதனால",
)

def endpoint_ready(silence_ms: int, stable: str = "", probability=None) -> bool:
    if silence_ms < 320:
        return False
    if silence_ms >= 2200:
        return True
    if probability is not None:
        if probability >= 0.72:
            return True
        if probability < 0.3 and silence_ms < 1600:
            return False
    words = normalized(stable).split()
    if words and words[-1] in INCOMPLETE_ENDINGS:
        return silence_ms >= 1450
    return silence_ms >= 780

def safe_to_prefill(stable: str) -> bool:
    """Speculate only on reasonably informative, stable user text."""
    words = normalized(stable).split()
    return len(words) >= 7 and len(stable) >= 32

def prefill_matches(stable: str, final: str) -> bool:
    """Never play an unconfirmed speculative response."""
    return bool(stable and normalized(stable) == normalized(final))

def speech_chunks(text: str, force: bool = False):
    """Small, pronounceable pieces. Leave unfinished text pending."""
    chunks = []
    # Use punctuation OR soft length limit so TTS can begin while LLM streams.
    while text:
        marker = re.search(r"[.!?\u0964](?=\s|$)", text)
        if marker and marker.end() <= 115:
            cut = marker.end()
        elif len(text) > 65:
            cut = text.rfind(" ", 30, 66)
            if cut <= 0:
                break
        else:
            break
        phrase = text[:cut].strip()
        if phrase:
            chunks.append(phrase)
        text = text[cut:].strip()
    if force and text.strip():
        chunks.append(text.strip())
        text = ""
    return chunks, text

def backchannel_due(spoken_seconds: float, silence_ms: int, since_last_seconds: float,
                    stable: str, enabled: bool = True) -> bool:
    """Rare listener acknowledgment, never every silence."""
    return (enabled and spoken_seconds > 4.0 and 180 <= silence_ms <= 650
            and since_last_seconds > 9.0 and len(normalized(stable).split()) >= 6
            and not endpoint_ready(silence_ms, stable))
