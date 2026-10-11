#!/usr/bin/env python3
"""Bounded CPU fixtures of frozen audit_training, not native artifact or score replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "code/audit_kgeditor_native.py"
PIN = "4153692fac983667dd8a4ae90d034f60a038dc3bdaf93525e7a98d60393914a5"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == PIN
loader = importlib.util.spec_from_file_location("reviewed_auditor", SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)


def exercise(name, *, wrong_selection=False, short_training=False, missing_epoch=False):
    with tempfile.TemporaryDirectory(prefix=name + "-", dir=HERE) as td:
        run = Path(td) / "runs" / "synthetic"
        logger = run / "logger/version_0"
        cp = logger / "checkpoints/model.ckpt"
        cp.parent.mkdir(parents=True)
        cp.write_bytes(b"MOCK checkpoint bytes; torch.load is replaced with synthetic CPU metadata")
        event = logger / "events.out.tfevents.synthetic"
        event.write_text("MOCK event bytes; EventAccumulator is replaced with synthetic scalars")
        (run / "run.log").write_text("Synthetic fixture, no producer executed.\n")
        job = {"id": "synthetic", "dataset": "E-FB15k237"}
        spec = m.SPECS[job["dataset"]]
        per_epoch = spec["steps"] // 20
        chosen_epoch = 3
        score = .9
        effective = dict(m.COMMON, ex_size=spec["ex_size"], kb_layer=spec["kb_layer"],
            data_type=job["dataset"][2:],
            ex_model_checkpoint=str(run.parents[1] / "models/KGEditor-HF" / job["dataset"]),
            model_name_or_path=str(run / "base_tokenizer"), data_dir=str(run / "data_view"),
            dirpath=str(run / "logger"))
        complete = dict(validation_exposure="native exposed evaluation",
            global_step=spec["steps"] - (per_epoch if short_training else 0), current_epoch=20,
            best_model_path=str(cp), best_model_score=score)
        summary = dict(evaluation_label=m.LABEL, checkpoint=str(cp),
            export=str(run / "native-exposed-examples.jsonl"),
            source_preflight=str(run / "source-preflight.json"), config=effective,
            checkpoint_sha256=m.sha(cp))
        terminal = dict(id=job["id"], exit_code=0, source_inputs_unchanged=True,
            log_sha256=m.sha(run / "run.log"))
        for filename, value in [("training-complete.json", complete),
            ("native-exposed-metrics.json", summary), ("terminal.json", terminal)]:
            (run / filename).write_text(json.dumps(value))
        ckpt = dict(epoch=chosen_epoch, global_step=(chosen_epoch+1)*per_epoch,
            hyper_parameters=effective, state_dict={"ex_model.weight": 0, "ex_learner.weight": 0},
            callbacks={"mock-ModelCheckpoint": dict(best_model_score=score, best_model_path=str(cp))})
        best_logged_epoch = 10 if wrong_selection else chosen_epoch
        monitors = [types.SimpleNamespace(step=(e+1)*per_epoch-1,
            value=score if e == best_logged_epoch else .4) for e in range(20)]
        epochs = [types.SimpleNamespace(step=(e+1)*per_epoch-1,value=e)
                  for e in range(19 if missing_epoch else 20)]
        fake_torch = types.ModuleType("torch")
        fake_torch.load = lambda *args, **kwargs: ckpt
        fake_torch.is_tensor = lambda x: False
        class Accumulator:
            def __init__(self, *args, **kwargs): pass
            def Reload(self): return self
            def Tags(self): return {"scalars": ["Eval/hits3", "epoch"]}
            def Scalars(self, tag): return monitors if tag == "Eval/hits3" else epochs
        mods = {name: types.ModuleType(name) for name in ["tensorboard", "tensorboard.backend",
            "tensorboard.backend.event_processing", "tensorboard.backend.event_processing.event_accumulator"]}
        mods["tensorboard.backend.event_processing.event_accumulator"].EventAccumulator = Accumulator
        mods["torch"] = fake_torch
        audit = m.Audit(name)
        with patch.dict(sys.modules, mods):
            m.audit_training(run, audit, job, spec, {})
        result = audit.result()
        result["synthetic_selection"] = dict(checkpoint_epoch=chosen_epoch,
            checkpoint_global_step=ckpt["global_step"], logged_best_epoch=best_logged_epoch,
            checkpoint_epoch_logged_score=monitors[chosen_epoch].value,
            declared_callback_best_score=score)
        # Temporary fixture paths are ephemeral; all synthetic values needed to reproduce are above.
        return result


results = [exercise("valid_early_best_after_full20epochs"),
    exercise("reject_19epoch_total", short_training=True),
    exercise("reject_missing_epoch19", missing_epoch=True),
    exercise("known_gap_selected_epoch_different_from_logged_best", wrong_selection=True)]
assert results[0]["failure_count"] == 0 and not results[0]["missing_evidence"]
assert "exact 20-epoch optimizer step count" in results[1]["failures"]
assert "logged epochs cover0..19" in results[2]["failures"]
assert results[3]["failure_count"] == 0 and not results[3]["missing_evidence"]
assert results[3]["synthetic_selection"]["checkpoint_epoch_logged_score"] != .9
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == PIN
output = dict(scope="MOCKED audit_training branch ONLY; no full audit or native artifact PASS",
    source_sha256=PIN, fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    gpu_launches=0, score_replays=0, cases=results,
    interpretation="Positive and negative contracts hold; wrong selected-epoch association is accepted, reproducing the bounded reviewer finding.")
(HERE / "selection-fixture-results.json").write_text(json.dumps(output, indent=2)+"\n")
print(json.dumps({"source_sha256":PIN, "fixture_sha256":output["fixture_sha256"],
    "cases":[{"name":x["run"],"checks":x["checks"],"failure_count":x["failure_count"],
    "failures":x["failures"]} for x in results],"harness_assertions":"PASS", "native_artifact_pass":False}))
