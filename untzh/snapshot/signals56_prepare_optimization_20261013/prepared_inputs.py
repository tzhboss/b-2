"""Lightweight invalidation for reuse of this fixed snapshot/seed preparation."""
from pathlib import Path
import optimized_experiment as experiment


def contract():
    paths = [
        experiment.SRC / 'entity_common_20261011/artifacts/incremental/newentityonly-distmult-s17/initial.pt',
        experiment.SRC / 'entity_diagnostic42_20261012/artifacts/data/train.npz',
        experiment.SRC / 'matched_protection52_20261013/artifacts/tape/learning.npz',
        experiment.SRC / 'full_increment_screen49_20261012/artifacts/evaluation/A.npz',
    ]
    files = {}
    for path in paths:
        st = path.stat()
        files[str(path)] = {'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    return {'recipe':'signals56-snapshot0-to1-reference-v1',
            'readonly_inputs':files,'old_count':7505,'entity_count':11258,
            'protection_seed':550017,'steps':740,'answers_per_step':32,
            'reference_draws':4,'scale_count':256,
            'scale_eta':1.4826,'scale_epsilon':1e-8,'scale_adequate':32}


if __name__ == '__main__':
    import json
    root = Path(__file__).resolve().parent
    experiment.write(root/'prepared-contract.json',contract())
    assert json.loads((root/'prepared-contract.json').read_text()) == contract()
    assert all((root/'cache'/f'{j:04d}.npz').is_file() for j in range(740))
    print('REUSE_INPUTS_AND_CACHE_READY')
