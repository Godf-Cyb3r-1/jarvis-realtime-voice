"""CI checks the Kaggle notebook is clean, ordered and Python-syntax valid."""
import ast
import json
from pathlib import Path
import unittest

NOTEBOOK=Path(__file__).resolve().parents[1]/"notebooks/Jarvis_Kaggle_RUN_IN_ORDER.ipynb"

class OrderedNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code=["".join(c["source"]) for c in cls.notebook["cells"]
                  if c["cell_type"]=="code"]

    def test_notebook_format(self):
        self.assertEqual(self.notebook["nbformat"],4)
        self.assertGreaterEqual(len(self.code),10)
        self.assertTrue(all(not c.get("outputs") for c in self.notebook["cells"]
                            if c["cell_type"]=="code"))

    def test_python_syntax(self):
        for i,source in enumerate(self.code):
            with self.subTest(cell=i):
                ast.parse(source)

    def test_stages_are_in_order(self):
        joined="\n".join(self.code)
        ordered=["git", "-C", "importlib.util", "torch.cuda.device_count()",
                 "dataset_candidates", "get_secret(\"VOICE_API_TOKEN\")",
                 "unittest", "uvicorn\", \"duplex_server:app",
                 "/debug/llm", "/debug/tts", "cloudflared"]
        positions=[]
        for token in ordered:
            p=joined.find(token)
            self.assertNotEqual(p,-1,token)
            positions.append(p)
        self.assertEqual(positions,sorted(positions))

    def test_no_embedded_output_or_secret(self):
        txt=NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("CBpwoJtquO",txt)
        self.assertNotIn("PRIVATE WINDOWS TOKEN:",txt)

if __name__=="__main__":
    unittest.main()
