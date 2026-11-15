import ast
import json
import unittest
from pathlib import Path

from realtime_protocol import (Hypothesis, common_prefix_words, endpoint_ready,
                               safe_to_prefill, prefill_matches, speech_chunks,
                               backchannel_due)

ROOT=Path(__file__).resolve().parents[1]

class DistributedConversationPolicyTests(unittest.TestCase):
    def test_local_agreement_avoids_unstable_last_word(self):
        self.assertEqual(common_prefix_words("Hello Jarvis what's", "Hello Jarvis what"),
                         "Hello")

    def test_revisions_invalidate_wrong_stable_hypothesis(self):
        h=Hypothesis()
        self.assertEqual(h.observe("Please open the blue folder")[0],"")
        stable,changed=h.observe("Please open the blue folder.")
        self.assertTrue(changed)
        self.assertIn("Please open",stable)
        new,changed=h.observe("Turn left at the blue folder")
        self.assertEqual(new,"")
        self.assertTrue(changed)

    def test_long_pause_only_when_incomplete(self):
        self.assertFalse(endpoint_ready(700,"I want to"))
        self.assertFalse(endpoint_ready(1100,"because"))
        self.assertTrue(endpoint_ready(1600,"because"))
        self.assertTrue(endpoint_ready(800,"Hello there"))

    def test_acoustic_completion_overrides_short_pause(self):
        self.assertFalse(endpoint_ready(250,"Hello",probability=0.9))
        self.assertTrue(endpoint_ready(350,"Hello",probability=0.9))
        self.assertFalse(endpoint_ready(900,"hello",probability=0.1))

    def test_unconfirmed_speculation_must_not_play(self):
        self.assertTrue(safe_to_prefill("Can you please tell me about the current project status"))
        self.assertTrue(prefill_matches("What is the weather?", "What is the weather?"))
        self.assertFalse(prefill_matches("Turn on the", "Turn off the lights"))
        self.assertFalse(prefill_matches("What is the", "What is the price?"))

    def test_first_phrase_streams_before_remaining_response(self):
        pieces,rest=speech_chunks("Hey! Good to hear you. I'm finishing")
        self.assertEqual(pieces,["Hey!","Good to hear you."])
        self.assertEqual(rest,"I'm finishing")
        p2,rest=speech_chunks(rest,force=True)
        self.assertEqual(p2,["I'm finishing"])
        self.assertEqual(rest,"")

    def test_backchannel_not_automatic_after_every_pause(self):
        self.assertFalse(backchannel_due(2,300,20,"You can listen to me now"))
        self.assertFalse(backchannel_due(6,300,2,"You can listen to me now as well"))
        self.assertTrue(backchannel_due(6,350,20,"You can listen to me right now as well"))
        self.assertFalse(backchannel_due(6,800,20,"You can listen to me right now as well"))

class NotebookSyntaxTests(unittest.TestCase):
    for_name=("Jarvis_Kaggle_A_Continuous_STT.ipynb",
              "Jarvis_Kaggle_B_OmniVoice_TTS.ipynb")

    def test_both_notebooks_compile_and_are_clean(self):
        for name in self.for_name:
            with self.subTest(notebook=name):
                data=json.loads((ROOT/"notebooks"/name).read_text(encoding="utf-8"))
                self.assertEqual(data["nbformat"],4)
                scripts=[c for c in data["cells"] if c["cell_type"]=="code"]
                self.assertGreaterEqual(len(scripts),5)
                for c in scripts:
                    self.assertFalse(c.get("outputs"))
                    ast.parse("".join(c["source"]))

    def test_no_groq_key_in_gpu_notebooks(self):
        for name in self.for_name:
            source=(ROOT/"notebooks"/name).read_text(encoding="utf-8")
            self.assertNotIn('get_secret(\\\"GROQ_API_KEY\\\")',source)
            self.assertIn("STT_WS_TOKEN" if "_A_" in name else "TTS_WS_TOKEN",source)

if __name__=="__main__":
    unittest.main()
