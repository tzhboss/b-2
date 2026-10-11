"""Dependency-free checks for staging and observation bookkeeping; no model/GPU."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent

class Checks(unittest.TestCase):
    def test_original_sources_preserved(self):
        import hashlib
        pins = json.loads((HERE/"author-source-sha256.json").read_text())
        for name,digest in pins.items():
            self.assertEqual(hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest(),digest)

    def test_syntax(self):
        for name in ["run_native.py","observations.py"]:
            ast.parse((HERE/name).read_text())

    def test_observation_aggregation_and_cursor(self):
        # Stub only imports; exercise the actual dependency-free bookkeeping.
        names = ["torch","src","src.models","src.models.models"]
        old = {k:sys.modules.get(k) for k in names}
        try:
            for k in names:
                sys.modules[k] = types.ModuleType(k)
            sys.modules["src"].models = sys.modules["src.models"]
            sys.modules["src.models"].models = sys.modules["src.models.models"]
            sys.modules["src.models.models"].KGEditor = object
            spec = importlib.util.spec_from_file_location("observations_test",HERE/"observations.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            f = dict(input_ids=[101,103,102],label=2,labels=[1])
            class Dataset:
                features=[f,dict(f)]
                def __len__(self):
                    return len(self.features)
            dataset = Dataset()
            class Native:
                memory_perm=[5,7,9]
                memory_pos=0
                stable_batch_size=2
                def __call__(self, features):
                    self.memory_pos += self.stable_batch_size
                    return {"sentinel": "unchanged", "size":len(features)+2}
            native = Native()
            observer = m.ObservedCollator(native,dataset,[0,1],"edit_test")
            batch = observer([f])
            self.assertEqual(native.memory_pos,2)
            self.assertEqual(batch["sentinel"],"unchanged")
            self.assertEqual(batch["_observations"][0]["source_row_candidates"],[5])
            self.assertEqual(batch["_observations"][1]["source_row_candidates"],[7])
            self.assertEqual(batch["_observations"][2]["source_row_candidates"],[0,1])
            self.assertEqual(f,dict(input_ids=[101,103,102],label=2,labels=[1]))
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/"rows.jsonl"
                rows=[dict(role="edit",rank_before=10,rank_after=1,source_row_candidates=[0]),
                      dict(role="locality",rank_before=2,rank_after=4,source_row_candidates=[5]),
                      dict(role="locality",rank_before=2,rank_after=2,source_row_candidates=[5])]
                p.write_text("".join(json.dumps(x)+"\n" for x in rows))
                result=m.summarize(p)
                self.assertEqual(result["edit"]["article_ER_roc"],.9)
                self.assertEqual(result["locality"]["article_RK_at3"],.5)
                self.assertEqual(result["locality"]["article_RK_roc"],1/3)
                self.assertEqual(result["locality"]["unique_source_rows"],1)
        finally:
            for k,v in old.items():
                if v is None:
                    sys.modules.pop(k,None)
                else:
                    sys.modules[k]=v

if __name__ == "__main__":
    unittest.main()
