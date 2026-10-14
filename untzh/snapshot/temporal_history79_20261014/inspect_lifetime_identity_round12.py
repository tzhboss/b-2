"""Read raw identity context for two cached competitors; never rescore models."""
import json
import time
from pathlib import Path
from raw_icews79 import raw_lines

ROOT = Path(__file__).resolve().parent
target = ROOT / 'lifetime-identity-context-round12.json'
if target.exists():
    print('EXISTING_OUTPUT_NO_RESCAN')
    raise SystemExit(0)
names = {'Chiang Kai-shek', 'Yitzhak Rabin'}
rows = {name: [] for name in names}
paths = [p for p in (ROOT / 'sources').glob('events.2005.*') if p.suffix in {'.zip', '.tab'}]
assert len(paths) == 1, paths
started = time.perf_counter()
reader = raw_lines(paths[0])
headers = next(reader)
sh, th = headers.index('Source Name'), headers.index('Target Name')
for row in reader:
    for name in names.intersection({row[sh], row[th]}):
        if len(rows[name]) < 5:
            rows[name].append(dict(zip(headers, row)))
    if all(len(rows[name]) >= 5 for name in names):
        break
out = {
    'records': rows,
    'seconds': time.perf_counter() - started,
    'scope': 'First five available raw2005 rows per actor, identity and coding context only; not source article bodies, full actor history, or truth labels.',
    'training': False, 'neural_rescoring': False
}
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(out, ensure_ascii=False))
