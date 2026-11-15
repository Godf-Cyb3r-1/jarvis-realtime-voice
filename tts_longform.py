"""Full-text TTS chunking; no GPU dependencies.

The debug endpoint used to silently truncate inputs to 200 characters.
Keep all supplied words and synthesize short, natural segments independently
to avoid long single-pass duration estimation or clipped output.
"""
import re

SUPPORTED_ALIASES = {
    "[chuckle]": "[laughter]",
    "[short pause]": "...",
}

def normalize_expression_tags(text: str) -> str:
    for unsupported, replacement in SUPPORTED_ALIASES.items():
        text = re.sub(re.escape(unsupported), lambda _: replacement, text,
                      flags=re.IGNORECASE)
    return text

def chunk_text(text: str, max_chars: int = 100) -> list[str]:
    """Split on sentence/pause boundaries, then on word boundaries.

    Never intentionally discard input words or split a word. Words longer
    than the target chunk size are retained as one chunk.
    """
    if not isinstance(text, str):
        raise TypeError("TTS text must be a string")
    if max_chars < 25:
        raise ValueError("max_chars must be >= 25")
    text = normalize_expression_tags(text).strip()
    if not text:
        return []
    # Text after ., !, ?, Tamil danda or line boundaries can begin a new chunk.
    sentences = re.split(r"(?<=[.!?\u0964\uff01\uff1f])\s+|\n+", text)
    output = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        current = []
        size = 0
        for word in words:
            new_size = size + len(word) + (1 if current else 0)
            if current and new_size > max_chars:
                output.append(" ".join(current))
                current = []
                size = 0
            current.append(word)
            size += len(word) + (1 if len(current) > 1 else 0)
        if current:
            output.append(" ".join(current))
    return output

def combine_pcm(pieces, sample_rate: int = 16000, gap_ms: int = 100):
    import numpy as np
    if not pieces:
        raise ValueError("No generated TTS audio chunks")
    arrays = [np.asarray(piece,dtype=np.float32).reshape(-1) for piece in pieces]
    if any(not len(a) for a in arrays):
        raise ValueError("OmniVoice returned an empty audio chunk")
    gap = np.zeros(int(sample_rate*gap_ms/1000),dtype=np.float32)
    parts=[]
    for index,a in enumerate(arrays):
        if index:
            parts.append(gap)
        parts.append(a)
    return np.concatenate(parts)
