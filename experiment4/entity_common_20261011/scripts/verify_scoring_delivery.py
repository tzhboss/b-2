"""Verify scorer evidence and a prose-only common-table correction."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    from build_common_table_v2 import build

    table_config = 'configs/common-table-v4.json'
    before_config = json.loads(subprocess.check_output(['git', 'show', 'HEAD:'+table_config], cwd=root))
    after_config = json.loads((root/table_config).read_text())
    key = 'KGE-ENTITY-COMMON-SHARED-GRAPH-UTILITY-032'
    before_config['utility_descriptions'][key]['text'] = after_config['utility_descriptions'][key]['text']
    assert before_config == after_config, 'Table config changed beyond the disclosed032 coverage prose'
    frozen_checks = 0
    for experiment in ['KGE-ENTITY-COMMON-FULL-INDEX-EQUIVALENCE-034', 'KGE-ENTITY-COMMON-RANK-CAPACITY-037', 'KGE-ENTITY-COMMON-RANK-CAPACITY-UTILITY-038']:
        canonical = yaml.safe_load((root/'experiments'/experiment/'experiment.yaml').read_text())
        for name, value in canonical['frozen_input_sha256'].items():
            assert digest(root/name) == value, (experiment, name)
            frozen_checks += 1
    summaries = {}
    for name in ['cpu-bitwise-shapes', 'cuda-bitwise-shapes', 'cpu-real-train', 'cuda-real-train']:
        path = root/'results/engineering-adaptive-scoring'/f'{name}.json'
        result = json.loads(path.read_text())
        assert result['status'].startswith('PASS') and result['model_updates'] == 0
        for code, value in result['code_sha256'].items():
            assert digest(root/code) == value
        for source, value in result.get('input_sha256', {}).items():
            assert digest(Path(source)) == value
        if 'shapes' in result:
            assert len(result['shapes']) == 14 and result['total_cells'] == 135514
            assert all(row['bit_patterns_checked'] and row['exact_original_and_numpy'] for row in result['shapes'])
        else:
            assert result['query_groups'] == 68270 and result['reference_members'] == 235882
            assert sum(row['score_cells'] for row in result['measurements']) == 204810
            assert all(row['bitwise_original_and_numpy'] for row in result['measurements'])
            assert not result['test_scored'] and not result['valid_scored']
            if name.startswith('cuda'):
                assert result['cuda_visible_devices'] == '6' and result['gpu_name'] == 'NVIDIA GeForce RTX 3090'
        summaries[name] = digest(path)
    table_path = root/'tables/common-comparison-v4/table.json'
    previous = json.loads(table_path.read_text())
    build(root, table_config)
    current = json.loads(table_path.read_text())
    index = lambda table: {(row['method'], row['model'], row['dimension']): row for row in table['rows']}
    old, new = index(previous), index(current)
    assert old.keys() == new.keys()
    fields = ['metrics', 'query_checks', 'new_directional_metrics', 'training_qualifications', 'audited_seeds', 'complete_seed_set']
    for key in old:
        for field in fields:
            assert old[key][field] == new[key][field], (key, field)
    assert previous['development_utility_screen'] == current['development_utility_screen']
    assert previous['additional_development_utility'] == current['additional_development_utility']
    report = (table_path.parent/'report.md').read_text()
    assert '18个任务全量VALID复算通过' not in report and '032逐seed独立复算全部VALID' not in report
    assert '旧／新VALID各1024条查询及完整候选池' in report
    result = dict(status='PASS_SCORING_EVIDENCE_AND_PROSE_ONLY_TABLE_CORRECTION', frozen_file_checks=frozen_checks,
                  unchanged_metric_rows=len(old), unchanged_development_gates_and_results=True,
                  scoring_summaries_sha256=summaries, table_sha256=digest(table_path),
                  scientific_utility_asserted=False, trajectory_equivalence_asserted=False)
    destination = root/'results/engineering-adaptive-scoring/verification.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.root.resolve())
