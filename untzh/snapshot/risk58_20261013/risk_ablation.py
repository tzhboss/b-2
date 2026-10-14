"""Use cached risk features/labels to test learner necessity and feature groups."""
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
import experiment58 as ex


def main():
    started = time.monotonic()
    data = ex.base.load(ex.ROOT / 'cache/cohort.npz')
    frames = [ex.base.load(ex.ROOT / 'cache' / f'frame-{i:02d}.npz') for i in range(21)]
    train = data['split'] == 0
    X = np.concatenate([frames[i]['X'][train] for i in range(8)])
    y = np.concatenate([(frames[i + 1]['ranks'][train] > 10).astype(int) for i in range(8)])
    safe = np.concatenate([frames[i]['ranks'][train] <= 10 for i in range(8)])
    models = {name: joblib.load(ex.ROOT / 'cache' / f'{name}.joblib') for name in ['logistic', 'tree']}
    definitions = {'tree_static': list(range(8)) + [20], 'tree_current': list(range(15)) + [20],
                   'tree_no_epoch': list(range(20)), 'tree_new_event': list(range(21))}
    fit = {}
    for name, cols in definitions.items():
        mask = safe if name == 'tree_new_event' else np.ones(len(y), bool)
        target = y[mask]
        weights = np.where(target, len(target) / (2 * target.sum()), len(target) / (2 * (1 - target).sum()))
        model = make_pipeline(ColumnTransformer([('columns', 'passthrough', cols)]),
                              HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, max_depth=3,
                                  learning_rate=.08, l2_regularization=1., early_stopping=False, random_state=17))
        start = time.monotonic()
        model.fit(X[mask], target, histgradientboostingclassifier__sample_weight=weights)
        models[name] = model
        joblib.dump(model, ex.ROOT / 'cache' / f'{name}.joblib')
        fit[name] = {'seconds': time.monotonic() - start, 'columns': [ex.FEATURE_NAMES[i] for i in cols],
                     'train_examples': len(target), 'positive_events': int(target.sum())}
        print('ABLATION_FIT', name, fit[name]['seconds'], flush=True)
    report = {'fit': fit, 'per_epoch': {}, 'operational': {}, 'cost_seconds': {}}
    rows = {}
    rng = np.random.default_rng(17)
    mask = data['split'] == 2
    for epoch in range(10, 20):
        Xi = frames[epoch]['X'][mask]
        current_safe = frames[epoch]['ranks'][mask] <= 10
        next_drop = frames[epoch + 1]['ranks'][mask] > 10
        new_drop = current_safe & next_drop
        scores = ex.rule_scores(Xi)
        scores['random'] = rng.random(len(Xi))
        for name, model in models.items():
            tick = time.monotonic()
            scores[name] = model.predict_proba(Xi)[:, 1]
            report['cost_seconds'][name] = report['cost_seconds'].get(name, 0.) + time.monotonic() - tick
        report['per_epoch'][str(epoch)] = {}
        for name, score in scores.items():
            report['per_epoch'][str(epoch)][name] = {
                'all_next_drop': ex.selection_metrics(next_drop, score),
                'new_drop_among_safe': ex.selection_metrics(next_drop, score, current_safe)}
            rows.setdefault(name, []).append((score, next_drop, new_drop, current_safe))
    for name, epochs in rows.items():
        summary = {}
        for fraction in [.05, .1, .2]:
            selected = new_hits = all_hits = new_total = all_total = 0
            epoch_recalls = []
            for score, target, new, safe in epochs:
                k = int(np.ceil(len(score) * fraction))
                chosen = np.argsort(-score, kind='stable')[:k]
                selected += k
                new_hits += int(new[chosen].sum()); all_hits += int(target[chosen].sum())
                new_total += int(new.sum()); all_total += int(target.sum())
                if new.any():
                    epoch_recalls.append(float(new[chosen].sum() / new.sum()))
            summary[str(fraction)] = {'selected_events': selected, 'new_drop_hits': new_hits,
                                      'new_drop_events': new_total, 'new_drop_recall': new_hits / new_total,
                                      'all_drop_recall': all_hits / all_total,
                                      'new_drop_precision': new_hits / selected,
                                      'macro_epoch_new_drop_recall': float(np.mean(epoch_recalls))}
        values = [report['per_epoch'][str(i)][name]['new_drop_among_safe']['average_precision'] for i in range(10, 20)]
        summary['macro_epoch_AP_new_drop'] = float(np.mean([v for v in values if v is not None]))
        report['operational'][name] = summary
    report['seconds'] = time.monotonic() - started
    ex.dump('risk-ablations.json', report)
    print('ABLATION_COMPLETE', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
