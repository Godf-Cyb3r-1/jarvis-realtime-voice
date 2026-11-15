"""Regression checks for noisy mic, low-RMS false turns, and real barge-ins."""
import unittest
import numpy as np
from voice_gate import pcm_rms, SpeechGate


def pcm(level, n=320):
    # Constant amplitude synthetic 16kHz PCM. Works for RMS tests even if VAD
    # is mocked; this does not simulate realistic spoken language.
    value=int(round(level*32767))
    return (np.ones(n,dtype="<i2")*value).tobytes()


class NoiseGateTests(unittest.TestCase):
    def setUp(self):
        self.g=SpeechGate({
            "VAD_RMS_MIN":"0.008",
            "VAD_BARGE_RMS_MIN":"0.020",
            "VAD_ONSET_FRAMES":"5",
            "VAD_BARGE_ONSET_FRAMES":"7",
            "VAD_NOISE_RATIO":"2.5",
        })

    def test_rms_normalized(self):
        self.assertEqual(pcm_rms(bytes(640)),0.0)
        self.assertAlmostEqual(pcm_rms(pcm(0.03)),0.03,delta=0.001)

    def test_vad_false_positive_low_rms_must_not_start(self):
        for _ in range(100):
            v=self.g.accept(pcm(0.002),True,bot_active=False)
            self.assertFalse(v)
            self.assertFalse(self.g.start_ready(v))
        self.assertEqual(self.g.last_rms<0.008,True)

    def test_real_voice_starts_after_sustained_onset(self):
        for _ in range(4):
            v=self.g.accept(pcm(0.045),True,bot_active=False)
            self.assertTrue(v)
            self.assertFalse(self.g.start_ready(v))
        self.assertTrue(self.g.start_ready(True))

    def test_spurious_speaker_leakage_does_not_interrupt(self):
        for _ in range(100):
            v=self.g.accept(pcm(0.012),True,bot_active=True)
            self.assertFalse(v)
            self.assertFalse(self.g.start_ready(v,bot_active=True))

    def test_loud_user_can_still_barge_in(self):
        for _ in range(6):
            v=self.g.accept(pcm(0.050),True,bot_active=True)
            self.assertTrue(v)
            self.assertFalse(self.g.start_ready(v,bot_active=True))
        self.assertTrue(self.g.start_ready(True,bot_active=True))

    def test_nonspeech_resets_onset(self):
        for _ in range(3):
            self.g.start_ready(True)
        self.assertFalse(self.g.start_ready(False))
        for _ in range(4):
            self.assertFalse(self.g.start_ready(True))
        self.assertTrue(self.g.start_ready(True))

    def test_acoustic_floor_rises_with_true_background_noise(self):
        for _ in range(300):
            self.g.accept(pcm(0.009),False,bot_active=False)
        self.assertGreater(self.g.noise_floor,0.007)
        self.assertFalse(self.g.accept(pcm(0.015),True,bot_active=False))


if __name__=="__main__":
    unittest.main()
