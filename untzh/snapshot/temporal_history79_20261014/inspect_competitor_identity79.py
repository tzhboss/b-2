"""Find raw country/sector metadata for selected temporally incompatible competitors."""
import json
import time
from pathlib import Path
from raw_icews79 import raw_lines

ROOT = Path(__file__).resolve().parent
target = ROOT / 'competitor-identity-context.json'
if target.exists():
    print('EXISTING_OUTPUT_NO_RESCAN')
    raise SystemExit(0)
crossings = json.loads((ROOT / 'fixed40-new-outrankers.json').read_text(encoding='utf-8'))
names = {'Liberal National Party', 'Council for National Security', 'Terrorist (Jabhat al-Nusra)'}
wanted = {}
for rec in crossings['records']:
    for cand in rec['new_outrankers']:
        for name in names.intersection(cand['fact']):
            wanted.setdefault(cand['train_first_year'], set()).add(name)
started = time.perf_counter()
records = {name: [] for name in names}
for year, year_names in sorted(wanted.items()):
    paths = [p for p in (ROOT / 'sources').glob(f'events.{year}.*') if p.suffix in {'.zip', '.tab'}]
    assert len(paths) == 1, paths
    reader = raw_lines(paths[0])
    headers = next(reader)
    sh, th = headers.index('Source Name'), headers.index('Target Name')
    for row in reader:
        for name in year_names.intersection({row[sh], row[th]}):
            if len(records[name]) < 5:
                records[name].append(dict(zip(headers, row)))
        if all(len(records[name]) >= 5 for name in year_names):
            break
    print('YEAR', year, {name: len(records[name]) for name in year_names}, flush=True)
out = {'records': records, 'seconds': time.perf_counter() - started,
       'scope': 'First up to five raw rows in TRAIN-first year for identity disambiguation. Raw names/countries/sectors are coded metadata, not independent truth or actor IDs.'}
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('COMPLETE', out['seconds'], flush=True)
