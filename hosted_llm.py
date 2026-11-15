"""Stream replies from temporary free-tier Groq or Gemini APIs.

No LLM GPU load on Kaggle. Credentials are read from environment/Secrets, never
embedded in source. Uses documented OpenAI-compatible streaming endpoints.
"""
import json
import os
import httpx

SYSTEM_PROMPT = (
    "You are Jarvis, an intelligent, lively, warm conversational companion. "
    "Speak naturally and briefly (usually one or two sentences) in English, "
    "Tamil or Tanglish matching the user. Never repeat, paraphrase or merely "
    "translate the user's sentence as your reply. Respond to the meaning. "
    "For a greeting, respond conversationally and invite discussion. "
    "For a complaint, acknowledge it and help solve it. "
    "If the transcript seems wrong or unclear, ask for clarification rather "
    "than confidently inventing an answer. No generic assistant disclaimers, "
    "markdown, stage tags, or unnecessary lengthy explanations. "
    "Keep responses speakable; don't overuse fillers."
)

def configuration(backend=None, environ=None):
    env = os.environ if environ is None else environ
    selected = (backend or env.get("LLM_BACKEND", "groq")).lower().strip()
    if selected == "groq":
        base = "https://api.groq.com/openai/v1"
        model = env.get("GROQ_MODEL", "openai/gpt-oss-20b")
        key = env.get("GROQ_API_KEY", "")
        key_name = "GROQ_API_KEY"
    elif selected == "gemini":
        base = "https://generativelanguage.googleapis.com/v1beta/openai"
        model = env.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        key = env.get("GEMINI_API_KEY", "")
        key_name = "GEMINI_API_KEY"
    elif selected == "api":
        base = env.get("LLM_BASE_URL", "").rstrip("/")
        model = env.get("LLM_MODEL", "")
        key = env.get("LLM_API_KEY", "")
        key_name = "LLM_API_KEY"
        if not base.startswith("https://"):
            raise RuntimeError("LLM_BASE_URL must be HTTPS")
    else:
        raise ValueError("LLM_BACKEND must be groq, gemini, api, or local")
    if not key.strip():
        raise RuntimeError(f"Missing {key_name}: set this Kaggle secret before starting Jarvis")
    if not model.strip():
        raise RuntimeError(f"No model configured for {selected}")
    return selected, base, model, key

async def stream_reply(messages, backend=None, environ=None, transport=None):
    provider, base, model, key = configuration(backend, environ)
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    request = {
        "model": model,
        "messages": messages[-16:],
        "stream": True,
        "temperature": 0.6,
        "max_tokens": int((os.environ if environ is None else environ).get("LLM_MAX_TOKENS", "140")),
    }
    timeout = httpx.Timeout(connect=10.0, read=45.0, write=15.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        async with client.stream("POST", base + "/chat/completions",
                                 headers=headers, json=request) as response:
            if response.status_code >= 400:
                detail = (await response.aread()).decode("utf-8", errors="replace")[:350]
                raise RuntimeError(f"{provider} API HTTP {response.status_code}: {detail}")
            emitted = False
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if not payload:
                    continue
                try:
                    obj = json.loads(payload)
                    choice = obj["choices"][0]
                    delta = choice.get("delta", {}).get("content")
                except (ValueError, IndexError, KeyError, TypeError):
                    continue
                if isinstance(delta, str) and delta:
                    emitted = True
                    yield delta
            if not emitted:
                raise RuntimeError(f"{provider} returned no spoken reply text")
