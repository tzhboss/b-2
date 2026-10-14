"""Read targeted metadata; use existing score lists, without neural rescoring."""
import json
import time
from pathlib import Path
from raw_icews79 import raw_lines

ROOT = Path(__file__).resolve().parent
target = ROOT / 'competitor-identity-context-round9.json'
if target.exists():
    print('EXISTING_OUTPUT_NO_RESCAN')
    raise SystemExit(0)
names = {'National Salvation Government', 'Democratic Alliance (Africa)'}
crossings = json.loads((ROOT / 'fixed40-new-outrankers.json').read_text(encoding='utf-8'))
wanted = {}
for rec in crossings['records']:
    for c in rec['new_outrankers']:
        for name in names.intersection(c['fact']):
            wanted.setdefault(c['train_first_year'], set()).add(name)
started = time.perf_counter()
records = {name: [] for name in names}
for year, subset in sorted(wanted.items()):
    paths = [p for p in (ROOT / 'sources').glob(f'events.{year}.*') if p.suffix in {'.zip', '.tab'}]
    assert len(paths) == 1, paths
    reader = raw_lines(paths[0])
    headers = next(reader)
    sh, th = headers.index('Source Name'), headers.index('Target Name')
    for row in reader:
        for name in subset.intersection({row[sh], row[th]}):
            if len(records[name]) < 5:
                records[name].append(dict(zip(headers, row)))
        if all(len(records[name]) >= 5 for name in subset):
            break
    print('YEAR', year, {name: len(records[name]) for name in subset}, flush=True)
out = {'records': records, 'seconds': time.perf_counter() - started,
       'scope': 'Up to five raw rows in TRAIN-first year, identity metadata only; not formation evidence or article bodies.'}
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('COMPLETE', out['seconds'], flush=True)
