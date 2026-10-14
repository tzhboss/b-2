"""Explicit adaptation: unique-query scale pools and fixed initial RMS calibration."""
import math
from collections import defaultdict
import numpy as np


def build_scales(records, *, eta, epsilon, adequate):
    """records: (canonical query tuple(anchor,relation,role), detached old sample scores).

    An identical repeated query must give identical ordered scale scores. Each unique
    query contributes its sample once to fallback pools, independent of replay frequency.
    """
    if not math.isfinite(eta) or eta <= 0 or not math.isfinite(epsilon) or epsilon <= 0 or adequate < 1:
        raise ValueError('invalid scale configuration')
    unique = {}
    for key, values in records:
        key = tuple(key)
        a = np.asarray(values, dtype=np.float64)
        if len(key) != 3 or key[2] not in ('head','tail') or a.ndim != 1 or not len(a) or not np.isfinite(a).all():
            raise ValueError('invalid eligible query scale record')
        if key in unique and not np.array_equal(unique[key],a):
            raise ValueError('inconsistent repeated query scale scores')
        unique[key] = a.copy()
    if not unique:
        raise ValueError('no eligible query records')
    rel, role = defaultdict(list), defaultdict(list)
    all_scores = []
    for key in sorted(unique):
        a = unique[key]
        rel[key[1:]].append(a); role[key[2]].append(a); all_scores.append(a)

    def pool(items):
        a = np.concatenate(items)
        return {'n':len(a),'mad':float(np.median(np.abs(a-np.median(a)))) if len(a)>=adequate else None}

    rel = {key:pool(value) for key,value in rel.items()}
    role = {key:pool(value) for key,value in role.items()}
    global_pool = pool(all_scores)
    result = {}
    for key,a in unique.items():
        q = np.quantile(a,[.25,.75],method='linear')
        iqr = float(q[1]-q[0])
        levels = [rel[key[1:]],role[key[2]],global_pool]
        eligible = [(level,item) for level,item in enumerate(levels) if item['mad'] is not None]
        level,item = eligible[0] if eligible else (None,{'mad':0.})
        result[key] = {'scale':max(iqr,eta*item['mad'],epsilon),'iqr':iqr,
                       'fallback_level':level,'fallback_mad':item['mad']}
    return result


def coefficient(base_norms, protect_norms, *, rho):
    if len(base_norms) != len(protect_norms) or not base_norms or not math.isfinite(rho) or rho <= 0:
        raise ValueError('invalid calibration cohort')
    for values in (base_norms,protect_norms):
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError('invalid gradient norm')
    base = math.hypot(*base_norms)/math.sqrt(len(base_norms))
    protect = math.hypot(*protect_norms)/math.sqrt(len(protect_norms))
    if not base or not protect:
        raise ValueError('zero RMS calibration signal; no automatic fallback')
    value = rho*base/protect
    if not math.isfinite(value) or value <= 0:
        raise ValueError('invalid calibrated coefficient')
    return {'lambda':value,'base_rms':base,'protection_rms':protect,'rho':rho,
            'batches':len(base_norms),'rule':'initial new-row Euclidean gradient RMS; not Adam or author recipe'}
