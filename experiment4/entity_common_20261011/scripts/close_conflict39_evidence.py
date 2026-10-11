"""Archive terminal039 without changing its frozen diagnostic inputs."""
import json
from pathlib import Path
import yaml

root=Path(__file__).resolve().parents[1]
state=yaml.safe_load((root/'experiments/KGE-ENTITY-COMMON-RANK-CONFLICT-DIAGNOSTIC-039/experiment.yaml').read_text())
assert state['status']=='completed_audited_train_diagnostic_only' and not state['active_processes']
report=root/'results/KGE-ENTITY-COMMON-RANK-CONFLICT-DIAGNOSTIC-039/mechanism-disclosure.json'
d=json.loads(report.read_text())
assert d['status']=='PASS_COMPLETE_TRAIN_SLOT_DIAGNOSIS_DISCLOSURE' and d['cpu_score_cells_replayed']==5764224
paragraph='''

039 is complete and audited for all three seeds/four policies. Its terminal
canonical has no active processes. Independent CPU fixtures replayed5,764,224
scores in384 complete query rows; full TRAIN scans were GPU-only. At least
95.4%-100% of original blocked source replacements have a fixed-old capacity
barrier (94.8%-100% after exemptions). New-row reordering cannot remove those
particular constraints. No weights were updated and no utility was established.
See039 report.md and mechanism-disclosure.json. Preserve TRUE-only013, task
metric019/020, fixed64-old-row032 and capacity037/038 negative decisions.

A distinct candidate to assess is direct newTRAIN retrieval optimization inside
the already audited rank-feasible set, instead of nearest-source projection.
It cannot remove immutable capacity barriers; its hypothesis is recovery of
useful scores on remaining queries/directions. It requires actual objective
controls, unchanged full/scoped references, bounds and common utility gates.
No expanded radius, relaxed guard or selected favorable seed is authorized by
this diagnosis. This candidate is not yet a useful method or novelty claim.
'''
for name in ('plan/review/outline-mechanism-status.md','plan/review/shared-graph-disposition.md'):
    p=root/name
    text=p.read_text()
    assert '039 is complete and audited' not in text
    p.write_text(text.rstrip()+paragraph,encoding='utf8')
errors=root/'.learnings/ERRORS.md'
errors.parent.mkdir(exist_ok=True)
with errors.open('a',encoding='utf8') as f:
    f.write('\n\n## 2026-10-11 follow-up source lookup\nTwo read-only source lookups used guessed paths. Actual capacity draft is plan/review/rank-capacity-followup-draft.md; task metric implementation is learned_task_projection. Discover file inventory before follow-up reads. No scientific inputs changed.\n')
print('CONFLICT39_TERMINAL_DOCUMENTATION_UPDATED')
