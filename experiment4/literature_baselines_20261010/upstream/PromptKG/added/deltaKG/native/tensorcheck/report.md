# CPU tensor observation verification

Status: **PASS**. No remaining CPU import/tensor blocker observed. This is a small synthetic CPU fixture, not a complete GPU run or article reproduction.

Both adapter files were preserved before the import-only correction in ../../provenance/adapter-before-importlib-fix/ with before/after SHA256 manifest. Replaced the shadow-sensitive dotted import by importlib.import_module('src.models.models') in run_native.py and observations.py. No author file changed.

The runner, observations, config and eight pinned author inputs are frozen. frozen-input-sha256.json and frozen-input-recheck.json show **11/11 byte hashes unchanged** across this check. The frozen input set excludes sidecar provenance and these new tensorcheck artifacts. No frozen file was modified during this continuation.

Executed environment/kgeditor38/bin/python with CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1 and one CPU thread. PyTorch reports 1.12.0+cu113; model parameters stayed on cpu. The installed library's HAMI preload emits cuInit messages during import; these messages are retained in attempt1.json, and do not constitute GPU execution by the fixture.

check_cpu.py runs the actual author BaseKEModule._eval through a tiny EXBert (2 layers, hidden16, FFN32, extension4, 10 entity candidates) and actual OneShotLearner with two editable matrices. It compares the original native path with the observation adapter loaded from the frozen files:

- All native returned ranks, complete entity ordering, edit input IDs, labels and edit ranks agree exactly.
- Both generated weight shifts agree exactly.
- Before/after exported target logits, ranks and top3 identities equal tensors captured independently from native forwards.
- Base model/editor state tensors remain unchanged; final RNG states agree exactly.
- Four edit/locality records are exported and aggregated; CPU import of both adapters and native_parser passes.
- The three dependency-free source/bookkeeping tests also passed under the installed Python3.8 environment; earlier logs remain in native/source_tests-python38-attempt1.json.

No extra model forward was inserted into the observed evaluation path. No GPU job, training, environment change, canonical modification or job-manifest widening was performed. Main controls the registered full runs. Checkpoint restore, full real-data training and article score agreement remain properties of those runs and are not established by this fixture.
