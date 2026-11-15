import unittest
import numpy as np
from tts_longform import chunk_text, combine_pcm, normalize_expression_tags


class LongFormSpeechTests(unittest.TestCase):
    def test_full_english_paragraph_is_not_cut_off(self):
        text = (
            "[laughter] How have you been holding up with everything lately? "
            "I know the past few weeks have been super busy on your end... "
            "[sigh] Honestly, I feel like this whole month just flew right past us! "
            "Whenever you finally catch a free moment to breathe, let's grab a coffee "
            "or hop on a quick call so you can tell me all about how the new project "
            "is going. [laughter] No rush at all though—just reply whenever you get a chance!"
        )
        self.assertGreater(len(text), 200)
        chunks = chunk_text(text, max_chars=100)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(chunks))
        self.assertEqual(" ".join(chunks), " ".join(text.split()))
        self.assertIn("whenever you get a chance!", chunks[-1])

    def test_tamil_and_expression_aliases(self):
        text = ("[chuckle] சரி, புரிஞ்சிது! நான் இப்போ உள்ள இருக்க தகவல்களை "
                "சரி பார்த்து உங்களுக்கு உடனே சொல்றேன். [short pause] "
                "கொஞ்சம் நேரம் காத்திருங்க, இணைப்பு இப்போ தான் தொடங்குது.")
        chunks=chunk_text(text, max_chars=68)
        joined=" ".join(chunks)
        self.assertIn("[laughter]", joined)
        self.assertNotIn("[short pause]",joined)
        self.assertIn("இணைப்பு இப்போ தான் தொடங்குது.",joined)
        self.assertEqual(joined, " ".join(normalize_expression_tags(text).split()))

    def test_audio_is_concatenated_with_a_short_gap(self):
        x=np.ones(16000, dtype=np.float32)
        y=np.ones(8000, dtype=np.float32)*0.4
        combined=combine_pcm([x,y], sample_rate=16000, gap_ms=100)
        self.assertEqual(len(combined), 25600)
        self.assertEqual(float(combined[16000]), 0.0)
        self.assertAlmostEqual(float(combined[-1]),0.4,places=6)

    def test_does_not_split_individual_words(self):
        long_word="வணக்கம்இதுஒருநீளமானதமிழ்சொல்"
        chunks=chunk_text(long_word+" thank you.",max_chars=25)
        self.assertTrue(chunks[0].startswith(long_word))
        self.assertEqual(" ".join(chunks),long_word+" thank you.")


if __name__ == "__main__":
    unittest.main()
