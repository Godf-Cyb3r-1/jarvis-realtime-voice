"""CUDA availability checks, no GPUs required."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gpu_guard

def fake_torch(count):
    class Cuda:
        @staticmethod
        def device_count():return count
        @staticmethod
        def mem_get_info(i):return (10*2**30,15*2**30)
        @staticmethod
        def get_device_name(i):return f"Mock GPU {i}"
    return SimpleNamespace(cuda=Cuda())

class MappingTests(unittest.TestCase):
    def test_auto_single_gpu(self):
        with patch.dict(sys.modules,{"torch":fake_torch(1)}),patch.dict(os.environ,{"TTS_GPU":"auto"}):
            self.assertEqual(gpu_guard.index_for("TTS_GPU",default=1),0)

    def test_explicit_invalid_gpu_clamps(self):
        with patch.dict(sys.modules,{"torch":fake_torch(1)}),patch.dict(os.environ,{"TTS_GPU":"1"}):
            self.assertEqual(gpu_guard.index_for("TTS_GPU",default=1),0)

    def test_dual_gpu_separation(self):
        with patch.dict(sys.modules,{"torch":fake_torch(2)}),patch.dict(os.environ,{"TTS_GPU":"auto"}):
            self.assertEqual(gpu_guard.index_for("TTS_GPU",default=1),1)

    def test_no_gpu_fails(self):
        with patch.dict(sys.modules,{"torch":fake_torch(0)}):
            with self.assertRaisesRegex(RuntimeError,"No compatible CUDA"):
                gpu_guard.index_for("TTS_GPU")

    def test_ctranslate_min_count(self):
        ct=SimpleNamespace(get_cuda_device_count=lambda:1)
        with patch.dict(sys.modules,{"torch":fake_torch(2),"ctranslate2":ct}),patch.dict(os.environ,{"ASR_GPU_INDEX":"auto"}):
            self.assertEqual(gpu_guard.index_for("ASR_GPU_INDEX",default=0,require_ctranslate=True),0)

    def test_gpu_inventory(self):
        ct=SimpleNamespace(get_cuda_device_count=lambda:2)
        with patch.dict(sys.modules,{"torch":fake_torch(2),"ctranslate2":ct}):
            info=gpu_guard.inventory()
            self.assertEqual(info["torch_cuda_count"],2)
            self.assertEqual(info["ctranslate2_cuda_count"],2)

if __name__=="__main__":
    unittest.main()
