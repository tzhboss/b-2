"""Declared adaptation choices; not the author's matching/scale implementation."""
import hashlib
import json
import math
from collections import defaultdict
import numpy as np


def stream(seed, *identity):
    blob = json.dumps([int(seed), *identity], sort_keys=True, separators=(',', ':')).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(blob).digest()[:16], 'big'))


class PrefixStructure:
    def __init__(self, triples, old_ids, new_ids):
        self.old = tuple(sorted(set(map(int, old_ids))))
        self.new = tuple(sorted(set(map(int, new_ids))))
        if set(self.old) & set(self.new):
            raise ValueError('old/new IDs overlap')
        ids = set(self.old) | set(self.new)
        self.facts = tuple(sorted(set(tuple(map(int, x)) for x in triples)))
        self.degree = defaultdict(int)
        signatures = defaultdict(set)
        self.truth = defaultdict(set)
        for h, r, t in self.facts:
            if h not in ids or t not in ids or min(h, r, t) < 0:
                raise ValueError('invalid original TRAIN triple')
            self.degree[h, 'head'] += 1
            self.degree[t, 'tail'] += 1
            signatures[h].add((r, 'head'))
            signatures[t].add((r, 'tail'))
            self.truth[h, r, 'tail'].add(t)
            self.truth[t, r, 'head'].add(h)
        self.signature = {e: tuple(sorted(signatures[e])) for e in ids}
        # These keys depend only on fixed TRAIN structure, never on scores or gold.
        self._role_cache = {}

    def role_cache(self, role):
        if role not in self._role_cache:
            cells = [defaultdict(list) for _ in range(3)]
            for e in self.old:
                for level in range(3):cells[level][self.key(e,role,level)].append(e)
            new_keys = {e:tuple(self.key(e,role,level) for level in range(3)) for e in self.new}
            self._role_cache[role] = cells,new_keys
        return self._role_cache[role]

    def key(self, e, role, level):
        degree = self.degree[e, role]
        bucket = degree.bit_length()  # 0, 1, 2--3, 4--7, ...
        return [(role, bucket, self.signature[e]), (role, bucket), (role,)][level]

    def cohorts(self, query, gold, *, seed, draws, scale_count):
        h, r, role = query
        if role not in ('head', 'tail') or draws <= 0 or scale_count <= 0:
            raise ValueError('invalid query/config')
        excluded = self.truth.get((h, r, role), set()) | {int(gold)}
        newcomers = [e for e in self.new if e not in excluded]
        old = [e for e in self.old if e not in excluded]
        if not newcomers or not old:
            return {'eligible': False, 'reason': 'empty_new_or_old', 'new': newcomers, 'old_pool': old}
        fixed_cells,new_keys = self.role_cache(role)
        cells = []
        for level in range(3):
            current = dict(fixed_cells[level])
            affected = {self.key(e,role,level) for e in excluded if e in self.signature and e in self.old}
            for key in affected:
                pool = [e for e in current.get(key,[]) if e not in excluded]
                if pool:current[key] = pool
                else:current.pop(key,None)
            cells.append(current)
        assignments = defaultdict(list)
        levels = []
        for i, e in enumerate(newcomers):
            for level in range(3):
                key = new_keys[e][level]
                if key in cells[level]:
                    assignments[level, key].append(i)
                    levels.append(level)
                    break
        references = []
        for draw in range(draws):
            row = [None] * len(newcomers)
            for (level, key), positions in sorted(assignments.items()):
                pool = cells[level][key]
                rng = stream(seed, list(query), 'reference', draw, level, key)
                first = list(map(int, rng.permutation(pool)))[:len(positions)]
                if len(first) < len(positions):
                    first += list(map(int, rng.choice(pool, size=len(positions)-len(first), replace=True)))
                for index, value in zip(positions, first):
                    row[index] = value
            references.append(row)
        scale = list(map(int, stream(seed, list(query), 'scale').choice(old, size=min(scale_count, len(old)), replace=False)))
        return {'eligible': True, 'new': newcomers, 'old_pool': old,
                'references': references, 'scale': scale, 'levels': levels}


def detached_scale(local_scores, ordered_fallback_pools, *, eta, epsilon, adequate):
    if not math.isfinite(eta) or eta <= 0 or not math.isfinite(epsilon) or epsilon <= 0 or adequate < 1:
        raise ValueError('invalid explicit scale settings')
    local = np.asarray(local_scores, dtype=float)
    if local.ndim != 1 or len(local) == 0 or not np.isfinite(local).all():
        raise ValueError('invalid local scale scores')
    iqr = float(np.quantile(local, .75, method='linear') - np.quantile(local, .25, method='linear'))
    mad, selected = 0., None
    for index, pool in enumerate(ordered_fallback_pools):
        a = np.asarray(pool, dtype=float)
        if a.ndim != 1:
            raise ValueError('invalid fallback pool')
        a = a[np.isfinite(a)]
        if len(a) >= adequate:
            mad = float(np.median(np.abs(a-np.median(a))))
            selected = index
            break
    return {'scale': max(iqr, eta*mad, epsilon), 'iqr': iqr,
            'fallback_mad': mad, 'fallback_index': selected}
