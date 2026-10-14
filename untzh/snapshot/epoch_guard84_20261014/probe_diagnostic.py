"""Compare saved decision probe endpoints to real final metrics; no scoring."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
def one(folder):
    summary=json.loads((folder/'summary.json').read_text());shadow=json.loads((folder/'shadow.json').read_text())['records']
    output={}
    for policy in ['local','cumulative']:
        epochs=json.loads((folder/'models'/policy/'epochs.json').read_text());last=epochs[-1]
        chosen=next(x for x in last['options'] if x['multiplier']==last['chosen'])
        baseline=shadow[-1]['probe'];sc=chosen['probe'];metrics=summary['metrics'][policy]
        output[policy]=dict(final_train_probe_ratios=[a/b for a,b in zip(sc['new_mrr'],baseline['new_mrr'])],
            final_valid_ratios=metrics['directional_ratios'],
            old_probe_drops=sc['old_drops'],shadow_old_probe_drops=baseline['old_drops'],
            final_full_train_drops=metrics['old_train']['drops']['filtered_train_truths']['drops'],
            ordinary_full_train_drops=summary['metrics']['none']['old_train']['drops']['filtered_train_truths']['drops'],
            average_multiplier=summary['costs']['training'][policy]['mean_multiplier'])
    return output
if (root/'seed29').exists():
    output={str(seed):one(root/f'seed{seed}') for seed in [29,43]}
else:output=one(root)
(root/'probe-vs-endpoint.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
