import unittest
from duplex_policy import TurnSettings, overlap_action, should_finish, split_ready

class DuplexPolicyTests(unittest.TestCase):
    def test_short_backchannels_not_interruptions(self):
        for text in ("hmm", "yeah", "சரி", "aama", "uh-huh"):
            self.assertEqual(overlap_action(text, 400), "CONTINUE")
    def test_explicit_interruptions(self):
        for text in ("wait", "stop", "நிறுத்து", "not that", "one second"):
            self.assertEqual(overlap_action(text, 300), "YIELD")
    def test_substantive_interruption(self):
        self.assertEqual(overlap_action("Actually I meant the other thing", 950), "YIELD")
    def test_uncertain_overlap(self):
        self.assertEqual(overlap_action("", 300), "HOLD")
    def test_pause_does_not_equal_turn_end(self):
        self.assertFalse(should_finish(330, 0.20))
        self.assertTrue(should_finish(410, 0.92))
        self.assertFalse(should_finish(250, 0.99))
        self.assertTrue(should_finish(2400, 0.01))
    def test_fallback(self):
        self.assertFalse(should_finish(600, None))
        self.assertTrue(should_finish(1200, None))
    def test_llm_chunks(self):
        a, b = split_ready("Hi! How are you? I am")
        self.assertEqual(a, ["Hi!", "How are you?"])
        self.assertEqual(b, " I am")
        a, b = split_ready(b, force=True)
        self.assertEqual(a, ["I am"])
        self.assertEqual(b, "")

if __name__ == "__main__":
    unittest.main()
