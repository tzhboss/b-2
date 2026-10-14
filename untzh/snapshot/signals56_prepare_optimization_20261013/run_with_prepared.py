"""Run the same fixed 056 recipe in a new directory using existing preparation."""
import argparse
import json
from pathlib import Path
import optimized_experiment as experiment
from prepared_inputs import contract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--prepared',type=Path,default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    prepared = args.prepared.resolve()
    output = args.output.resolve()
    allowed = Path('/data/lc/untzh').resolve()
    if allowed not in output.parents or output == prepared:
        raise ValueError('Use a separate new output directory under /data/lc/untzh')
    saved = json.loads((prepared/'prepared-contract.json').read_text())
    if saved != contract():
        raise ValueError('Preparation inputs/settings changed; rebuild affected preparation')
    report = json.loads((prepared/'prepare.json').read_text())
    if report['steps'] != 740 or any(not (prepared/'cache'/f'{j:04d}.npz').is_file() for j in range(740)):
        raise ValueError('Incomplete prepared cache')
    output.mkdir(exist_ok=False)
    experiment.ROOT = output

    def reused_prepare(*_):
        result = {'reused':True,'source':str(prepared),
                  'original_preparation_seconds':report['seconds'],
                  'new_preparation_seconds':0.0,'steps':740,
                  'cache_bytes':report['cache_bytes']}
        experiment.write(output/'prepare.json',result)
        return result

    def reused_cache(step):
        return {k:experiment.torch.as_tensor(v,device=experiment.DEVICE)
                for k,v in experiment.load(prepared/'cache'/f'{step:04d}.npz').items()}

    experiment.prepare = reused_prepare
    experiment.cache = reused_cache
    experiment.main()


if __name__ == '__main__':
    main()
