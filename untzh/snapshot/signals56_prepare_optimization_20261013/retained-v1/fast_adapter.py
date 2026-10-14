"""Reuse fixed TRAIN matching groups; preserve ordered reference draws exactly."""
from collections import defaultdict
import numpy as np
from reference_adapter import PrefixStructure as ReferenceStructure, stream


class PrefixStructure(ReferenceStructure):
    def __init__(self, triples, old_ids, new_ids):
        super().__init__(triples, old_ids, new_ids)
        self._old_set = set(self.old)
        self._old_array = np.asarray(self.old, dtype=np.int64)
        self._new_array = np.asarray(self.new, dtype=np.int64)
        self._matching_groups = {}

    def matching_groups(self, role):
        if role not in self._matching_groups:
            cells, new_keys = self.role_cache(role)
            groups = defaultdict(list)
            for i, e in enumerate(self.new):
                for level, key in enumerate(new_keys[e]):
                    if key in cells[level]:
                        groups[level, key].append(i)
                        break
            pools = [{k: np.asarray(v, dtype=np.int64) for k, v in c.items()} for c in cells]
            self._matching_groups[role] = pools, {k: np.asarray(v) for k, v in groups.items()}
        return self._matching_groups[role]

    def cohorts(self, query, gold, *, seed, draws, scale_count):
        h, r, role = query
        if role not in ('head', 'tail') or draws <= 0 or scale_count <= 0:
            raise ValueError('invalid query/config')
        excluded = self.truth.get((h, r, role), set()) | {int(gold)}
        excluded_array = np.asarray(sorted(excluded), dtype=np.int64)
        keep_new = ~np.isin(self._new_array, excluded_array)
        newcomers = self._new_array[keep_new]
        old = self._old_array[~np.isin(self._old_array, excluded_array)]
        if not len(newcomers) or not len(old):
            return {'eligible': False, 'reason': 'empty_new_or_old',
                    'new': newcomers.tolist(), 'old_pool': old.tolist()}
        cells, groups = self.matching_groups(role)
        changed = {}
        for e in excluded & self._old_set:
            for level in range(3):
                key = self.key(e, role, level)
                identity = level, key
                if identity not in changed:
                    pool = cells[level][key]
                    changed[identity] = pool[~np.isin(pool, excluded_array)]

        def pool_for(level, key):
            return changed.get((level, key), cells[level].get(key))

        assignments = defaultdict(list)
        for (level, key), positions in groups.items():
            positions = positions[keep_new[positions]]
            if not len(positions):
                continue
            pool = pool_for(level, key)
            if pool is None or not len(pool):
                # Signature -> degree -> role form nested groups. All members
                # of a default group share each coarser key.
                e = int(self._new_array[positions[0]])
                for level in range(level + 1, 3):
                    key = self.key(e, role, level)
                    pool = pool_for(level, key)
                    if pool is not None and len(pool):
                        break
            assignments[level, key].append(positions)
        index = np.full(len(self.new), -1, dtype=np.int64)
        index[keep_new] = np.arange(len(newcomers))
        assignments = {k: index[np.sort(np.concatenate(v))] for k, v in assignments.items()}
        levels = np.empty(len(newcomers), dtype=np.int64)
        for (level, _), positions in assignments.items():
            levels[positions] = level
        references = np.empty((draws, len(newcomers)), dtype=np.int64)
        for draw in range(draws):
            for (level, key), positions in sorted(assignments.items()):
                pool = pool_for(level, key)
                rng = stream(seed, list(query), 'reference', draw, level, key)
                first = rng.permutation(pool)[:len(positions)]
                if len(first) < len(positions):
                    first = np.concatenate([first, rng.choice(pool, size=len(positions)-len(first), replace=True)])
                references[draw, positions] = first
        scale = stream(seed, list(query), 'scale').choice(old, size=min(scale_count, len(old)), replace=False)
        return {'eligible': True, 'new': newcomers.tolist(), 'old_pool': old.tolist(),
                'references': references.tolist(), 'scale': scale.tolist(), 'levels': levels.tolist()}
