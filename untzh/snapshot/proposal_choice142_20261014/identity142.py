import json
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
models={a:torch.load(root/'models'/a/'epoch-20.pt',map_location='cpu',weights_only=True) for a in ['fixed_replay','pareto_choice','random_matched']}
identity={a:all(torch.equal(models[a][k],models['fixed_replay'][k]) for k in ['E','R']) for a in ['pareto_choice','random_matched']}
records=json.loads((root/'models/pareto_choice/steps.json').read_text())
summary=dict(parameter_equal_to_fixed_replay=identity,steps=len(records),
             admissible_counts={str(a):sum(a in r['admissible'] for r in records) for a in [0,1,2]},
             selected_counts={str(a):sum(r['action']==a for r in records) for a in [0,1,2]},
             not_independent_method_confirmation=all(identity.values()),
             reuse_field_in_original_summary_only_compares_three_fixed_arms=True)
(root/'identity-and-decision.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary))
