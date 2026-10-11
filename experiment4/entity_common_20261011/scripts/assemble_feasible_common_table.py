"""Create a new common-table snapshot; all41 historical rows must be exact."""
import argparse,json
from pathlib import Path
from hashlib import sha256
from build_common_table_v3 import build

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=a.root.resolve();destination=root/a.output
assert not destination.exists(),'Retain every assembly receipt; choose a fresh path.'
old=root/'tables/common-comparison-v4/table.json';original=json.loads(old.read_text());old_digest=sha256(old.read_bytes()).hexdigest()
assert original['completed_jobs']==117 and original['planned_jobs']==123 and len(original['rows'])==41
build(root,'configs/common-table-v5.json')
new=root/'tables/common-comparison-v5/table.json';updated=json.loads(new.read_text())
by_key=lambda rows:{(r['method'],r['model'],r['dimension']):r for r in rows}
before=by_key(original['rows']);after=by_key(updated['rows']);assert len(after)==43
assert all(after[key]==value for key,value in before.items()),'Historical row or failure status changed'
assert sha256(old.read_bytes()).hexdigest()==old_digest,'Original table mutated'
extra=[r for key,r in after.items() if key not in before]
assert sorted(r['method'] for r in extra)==['FeasibleTrainFull','FeasibleTrainScoped']
assert updated['planned_jobs']==129
for row in extra:
    assert [j['seed'] for j in row['per_seed_status']]==[17,29,43]
    if not row['audited_seeds']:assert row['metrics']=={} and not row['complete_seed_set']
receipt=dict(status='PASS_COMMON_TABLE_EXTENSION_AND_ALL41_HISTORICAL_ROWS_EXACT',
       historical_rows_unchanged=41,completed_jobs=updated['completed_jobs'],planned_jobs=129,
       new_rows=[dict(method=r['method'],audited_seeds=r['audited_seeds'],complete_seed_set=r['complete_seed_set'],per_seed_status=r['per_seed_status']) for r in extra],
       source_table_sha256=old_digest,new_table_sha256=sha256(new.read_bytes()).hexdigest(),scientific_utility_asserted=False)
destination.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
