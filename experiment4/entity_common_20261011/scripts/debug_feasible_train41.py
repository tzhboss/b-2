"""Retain failure diagnostics; never run a graph model or change settings."""
import json
import control_feasible_train41 as control
original=control.optimize_train_row
def diagnostic(*args,**kwargs):
    result=original(*args,**kwargs)
    print(json.dumps(dict(receipt=result[1],rows=result[0].rows.tolist()),indent=2),flush=True)
    return result
control.optimize_train_row=diagnostic
control.controls()
