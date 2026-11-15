"""Offline hosted-provider and ASR filtering tests: no API key, GPU or network."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import httpx

from hosted_llm import configuration, stream_reply, SYSTEM_PROMPT
from asr_quality import transcribe_audio

class HostedProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_groq_current_id(self):
        _, base, model, key=configuration("groq",{"GROQ_API_KEY":"mock-secret"})
        self.assertEqual(base,"https://api.groq.com/openai/v1")
        self.assertEqual(model,"openai/gpt-oss-20b")
        self.assertEqual(key,"mock-secret")

    def test_gemini_compatible_endpoint(self):
        _, base, model, _=configuration("gemini",{"GEMINI_API_KEY":"mock-secret"})
        self.assertEqual(base,"https://generativelanguage.googleapis.com/v1beta/openai")
        self.assertEqual(model,"gemini-3.5-flash-lite")

    def test_missing_secret_fails(self):
        with self.assertRaisesRegex(RuntimeError,"GROQ_API_KEY"):
            configuration("groq",{})

    async def test_sse_reply_and_prompt(self):
        def handler(request):
            body=json.loads(request.content)
            self.assertEqual(body["model"],"openai/gpt-oss-20b")
            self.assertEqual(request.headers["authorization"],"Bearer mock-secret")
            return httpx.Response(200,text=(
                'data: {"choices":[{"delta":{"content":"Hey! "}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"How are you?"}}]}\n\n'
                'data: [DONE]\n\n'
            ))
        transport=httpx.MockTransport(handler)
        chunks=[]
        async for fragment in stream_reply([{"role":"system","content":SYSTEM_PROMPT},
                          {"role":"user","content":"Hello."}],
                          environ={"LLM_BACKEND":"groq","GROQ_API_KEY":"mock-secret"},
                          transport=transport):
            chunks.append(fragment)
        self.assertEqual("".join(chunks),"Hey! How are you?")

    async def test_api_error_not_silently_echoed(self):
        transport=httpx.MockTransport(lambda req:httpx.Response(429,text="quota exceeded"))
        with self.assertRaisesRegex(RuntimeError,"HTTP 429"):
            async for _ in stream_reply([{"role":"user","content":"hello"}],
                    environ={"LLM_BACKEND":"groq","GROQ_API_KEY":"mock-secret"},
                    transport=transport):pass

class WhisperQualityTests(unittest.TestCase):
    def _segment(self,text,avg=0,nospeech=0):
        return SimpleNamespace(text=text,avg_logprob=avg,no_speech_prob=nospeech)
    def test_forces_english_after_wrong_spanish_detection(self):
        class Model:
            def __init__(self):self.calls=[]
            def transcribe(self,audio,**kwargs):
                self.calls.append(kwargs["language"])
                if kwargs["language"] is None:
                    return iter([SimpleNamespace(text="a todos.",avg_logprob=0,no_speech_prob=0)]),SimpleNamespace(language="es")
                return iter([SimpleNamespace(text="Hello Jarvis.",avg_logprob=0,no_speech_prob=0)]),SimpleNamespace(language="en")
        m=Model()
        text,det=transcribe_audio(m,np.ones(16000,dtype=np.int16)*3000,{"ASR_LANGUAGE":"auto","ASR_ALLOWED_LANGUAGES":"en,ta"})
        self.assertEqual(text,"Hello Jarvis.")
        self.assertEqual(m.calls,[None,"en"])
    def test_silence_is_ignored(self):
        class Bad:
            def transcribe(self,*args,**kwargs):raise AssertionError("ASR should never run")
        t,reason=transcribe_audio(Bad(),np.zeros(16000,dtype=np.int16),{})
        self.assertEqual(t,"")
    def test_low_confidence_segment_rejected(self):
        class Model:
            def transcribe(self,*args,**kwargs):
                return iter([SimpleNamespace(text="Thank you.",avg_logprob=-1.8,no_speech_prob=0.9)]),SimpleNamespace(language="en")
        text,_=transcribe_audio(Model(),np.ones(16000,dtype=np.int16)*3000,{"ASR_LANGUAGE":"en"})
        self.assertEqual(text,"")

if __name__=="__main__":
    unittest.main()
