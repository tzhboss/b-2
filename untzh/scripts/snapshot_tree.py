"""Export a read-only lightweight snapshot of an experiment tree."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile

EXCLUDED_DIRS = {'.git', '__pycache__', '.cache', '.venv', 'venv', 'node_modules',
                 'cache', 'models', 'checkpoints', 'ranks', 'rank', 'logs',
                 'artifacts', 'environment', 'repos', 'data', 'datasets',
                 'prepared', 'inputs', 'shadow'}
EXTENSIONS = {'.py', '.md', '.yaml', '.yml', '.toml', '.sh', '.bash', '.json',
              '.txt', '.rst', '.ini', '.cfg', '.csv', '.tsv', '.patch', '.diff',
              '.bib', '.tex'}
ROW_NAMES = {'steps.json', 'online-steps.json', 'trajectory.json',
             'proposal-records.json',
             'validation-selection.json', 'fixed40-new-outrankers.json',
             'fixed40-filtered-competitors.json', 'prediction-entities.json',
             'temporal-witness-candidates.json', 'sample-coded-event-context.json',
             'all-sample-coded-context.json'}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def reason(path):
    name = path.name.lower()
    if path.is_symlink():
        return 'symlink'
    if name == '.env' or name.startswith('.env.') or path.suffix.lower() in {'.key', '.pem'}:
        return 'credential-file'
    if path.suffix.lower() == '.log' or name.endswith('-profile.txt'):
        return 'runtime-log-or-profile'
    if name in ROW_NAMES or name.startswith('raw-matched-tuples-') or 'history-examples' in name:
        return 'row-level-runtime-or-raw-data'
    if name.startswith('representative-search-receipts') or re.fullmatch(r'round\d+-web-receipts(?:-tail)?\.json', name):
        return 'raw-source-retrieval-receipt'
    if path.suffix.lower() not in EXTENSIONS and name not in {'LICENSE', 'NOTICE', 'Makefile', 'Dockerfile', '.gitignore'}:
        return 'binary-or-runtime-file'
    if path.stat().st_size > 1024 * 1024:
        return 'over-1-MiB'
    return None


def directory_size(path):
    files = total = 0
    for base, dirs, names in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(base) / name).is_symlink()]
        for name in names:
            file = Path(base) / name
            if not file.is_symlink():
                files += 1
                total += file.stat().st_size
    return files, total


def snapshot(source, output, archive):
    source = source.resolve(strict=True)
    output = output.resolve()
    if output == source or source in output.parents or output.exists() or archive.exists():
        raise ValueError('Use a new output directory and archive outside the source tree')
    output.mkdir(parents=True)
    selected, excluded, dependencies, registrations = [], [], [], []
    taken_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for base, dirs, names in os.walk(source, followlinks=False):
        for name in sorted(dirs.copy()):
            directory = Path(base) / name
            if name.lower() in EXCLUDED_DIRS or directory.is_symlink():
                count, size = (0, 0) if directory.is_symlink() else directory_size(directory)
                excluded.append({'path': directory.relative_to(source).as_posix() + '/',
                                 'reason': 'excluded-runtime-directory', 'files': count, 'bytes': size})
                dirs.remove(name)
        dirs.sort()
        for name in sorted(names):
            path = Path(base) / name
            rel = path.relative_to(source).as_posix()
            why = reason(path)
            if why:
                excluded.append({'path': rel, 'reason': why, 'bytes': path.stat().st_size})
                continue
            before = path.stat()
            content = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Source changed during read: ' + rel)
            target = output / 'snapshot' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            selected.append({'path': rel, 'bytes': len(content),
                             'sha256': hashlib.sha256(content).hexdigest()})
            if path.suffix == '.py':
                text = content.decode('utf-8')
                for number, line in enumerate(text.splitlines(), 1):
                    for value in re.findall(r'/data/lc/[^\s\x22\x27)\]}>,;]+', line):
                        dependencies.append({'file': rel, 'line': number, 'reference': value,
                                             'inside_snapshot_source': value.startswith(str(source) + '/')})
    for folder in sorted(p for p in source.iterdir() if p.is_dir()):
        rel = folder.name
        paths = [item['path'] for item in selected if item['path'].split('/')[0] == rel]
        stage = None
        status_file = output / 'snapshot' / rel / 'status.json'
        if status_file.exists():
            stage = json.loads(status_file.read_text(encoding='utf-8')).get('stage')
        protocol = next((p for p in [f'{rel}/plan/experiment-protocol.md', f'{rel}/protocol.md', f'{rel}/plan.md', f'{rel}/README.md'] if p in paths), None)
        metadata = {'id': rel, 'snapshot_at_utc': taken_at,
                    'execution': {'observed_stage': stage, 'receipt': '../../snapshot/' + rel + '/status.json' if status_file.exists() else None},
                    'scientific_validity': None,
                    'disposition': 'archival development evidence; not promoted to formal results',
                    'protocol': '../../snapshot/' + protocol if protocol else None,
                    'source_scripts': ['../../snapshot/' + p for p in paths if p.endswith('.py')],
                    'source_root': str(folder),
                    'external_artifacts': '../../provenance/upload-manifest.json'}
        dump(output / 'experiments' / rel / 'experiment.yaml', metadata)
        registrations.append({'id': rel, 'registration': 'experiments/' + rel + '/experiment.yaml'})
    manifest = {'schema_version': 1, 'source_root': str(source), 'snapshot_at_utc': taken_at,
                'selected_files': len(selected), 'selected_bytes': sum(x['bytes'] for x in selected),
                'directories': len(registrations), 'files': selected, 'excluded': excluded,
                'notes': ['Source files are preserved byte-for-byte.', 'Hashes verify transfer integrity only; no scientific audit, training or evaluation was performed.',
                          'Excluded directory sizes are inventories, not content hashes. Models and caches have no second backup in GitHub.']}
    dump(output / 'provenance' / 'upload-manifest.json', manifest)
    dump(output / 'provenance' / 'dependency-locations.json', {'references': dependencies,
        'notes': ['Static references from archived Python sources, not a complete dependency resolver.',
                  'Original server paths are retained for provenance. Configure paths before execution elsewhere.',
                  'Dependencies outside this snapshot must be fetched from experiment4 or the server.']})
    dump(output / 'experiments' / 'index.json', registrations)
    script_target = output / 'scripts' / 'snapshot_tree.py'
    script_target.parent.mkdir(parents=True)
    shutil.copyfile(__file__, script_target)
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(output, arcname='untzh')
    print(json.dumps({'selected_files': len(selected), 'selected_bytes': manifest['selected_bytes'],
                      'directories': len(registrations), 'excluded_entries': len(excluded),
                      'archive_bytes': archive.stat().st_size,
                      'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    args = parser.parse_args()
    snapshot(args.source_root, args.output_root, args.archive)
