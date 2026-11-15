"""Exercise real Jarvis.handle_final using AST extraction without audio/network.

Reproduces the user's 37-turn log: a valid final from turn N may be received
after the microphone started turn N+1. The old code unconditionally discarded
such results, causing Jarvis to hear but never answer.
"""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from duplex_policy import overlap_action

ROOT=Path(__file__).resolve().parents[1]

def load_final_handler():
    source=(ROOT/"realtime_jarvis.py").read_text(encoding="utf-8")
    module=ast.parse(source)
    cls=next(node for node in module.body if isinstance(node,ast.ClassDef)
             and node.name=="Jarvis")
    func=next(node for node in cls.body if isinstance(node,ast.AsyncFunctionDef)
              and node.name=="handle_final")
    out=ast.Module(body=[func],type_ignores=[])
    scope={"overlap_action":overlap_action}
    exec(compile(out,"realtime_jarvis.py","exec"),scope)
    return scope["handle_final"]

class FinalRaceTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler=staticmethod(load_final_handler())

    def make_fake(self,turn=2,voice_ms=0,capturing=False,meta=None):
        return SimpleNamespace(
            turn=turn, voice_ms=voice_ms, capturing=capturing,
            turn_meta=meta or {}, last_accepted_turn=0,
            deferred_final=None, commit_final=AsyncMock(),
            audio=SimpleNamespace(restore=lambda:None))

    async def test_late_valid_final_still_gets_answer(self):
        fake=self.make_fake(turn=2,
            meta={1:{"overlap":False,"yielded":False,"voice_ms":450}})
        await self.handler(fake,1,"Hello, Jarvis.")
        fake.commit_final.assert_awaited_once_with(1,"Hello, Jarvis.")

    async def test_proper_overlap_uses_original_turn_not_current_vad(self):
        fake=self.make_fake(turn=8,
            meta={7:{"overlap":True,"yielded":False,"voice_ms":400}})
        await self.handler(fake,7,"Mm-hmm")
        fake.commit_final.assert_not_awaited()

    async def test_new_genuine_speech_defers_old_answer(self):
        fake=self.make_fake(turn=4,voice_ms=500,capturing=True,
            meta={3:{"overlap":False,"yielded":False,"voice_ms":500}})
        await self.handler(fake,3,"A real question")
        self.assertEqual(fake.deferred_final,(3,"A real question"))
        fake.commit_final.assert_not_awaited()

    async def test_empty_followup_recovers_deferred_answer(self):
        fake=self.make_fake(turn=4,voice_ms=0,capturing=False,
            meta={4:{"overlap":False,"yielded":False,"voice_ms":0}})
        fake.deferred_final=(3,"I asked something real.")
        await self.handler(fake,4,"")
        fake.commit_final.assert_awaited_once_with(3,"I asked something real.")
        self.assertIsNone(fake.deferred_final)

    async def test_older_result_after_newer_answer_is_stale(self):
        fake=self.make_fake(turn=4,meta={3:{"overlap":False,"yielded":False,
                                            "voice_ms":250}})
        fake.last_accepted_turn=4
        await self.handler(fake,3,"old text")
        fake.commit_final.assert_not_awaited()

if __name__=="__main__":
    unittest.main()
