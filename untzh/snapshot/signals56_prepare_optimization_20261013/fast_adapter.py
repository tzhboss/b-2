"""Reuse fixed TRAIN matching groups; preserve ordered reference draws exactly."""
from collections import defaultdict
import hashlib
import json
import numpy as np
from reference_adapter import PrefixStructure as ReferenceStructure, stream


class PrefixStructure(ReferenceStructure):
    def __init__(self, triples, old_ids, new_ids):
        super().__init__(triples, old_ids, new_ids)
        self._old_set = set(self.old)
        self._old_array = np.asarray(self.old, dtype=np.int64)
        self._new_array = np.asarray(self.new, dtype=np.int64)
        self._matching_groups = {}
        self._suffixes = {}

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
            self._matching_groups[role] = pools, {k: np.asarray(v) for k, v in sorted(groups.items())}
        return self._matching_groups[role]

    def draw_stream(self, prefix, draw, level, key):
        identity = draw, level, key
        suffix = self._suffixes.get(identity)
        if suffix is None:
            suffix = json.dumps(['reference', draw, level, key],sort_keys=True,separators=(',',':'))[1:].encode()
            self._suffixes[identity] = suffix
        entropy = int.from_bytes(hashlib.sha256(prefix + suffix).digest()[:16], 'big')
        return np.random.Generator(np.random.PCG64(entropy))

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
                    changed[identity] = np.asarray([e for e in pool if e not in excluded],dtype=np.int64)

        def pool_for(level, key):
            return changed.get((level, key), cells[level].get(key))

        assignments = {}
        shifted = []
        has_new_exclusions = not keep_new.all()
        for (level, key), positions in groups.items():
            if has_new_exclusions:
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
                shifted.append(((level,key),positions))
            else:
                assignments[level,key] = positions
        for identity,positions in shifted:
            if identity in assignments:
                positions = np.sort(np.concatenate([assignments[identity],positions]))
            assignments[identity] = positions
        if has_new_exclusions:
            index = np.full(len(self.new), -1, dtype=np.int64)
            index[keep_new] = np.arange(len(newcomers))
            assignments = {k:index[v] for k,v in assignments.items()}
        ordered = sorted(assignments.items()) if shifted else list(assignments.items())
        levels = np.empty(len(newcomers), dtype=np.int64)
        for (level, _), positions in assignments.items():
            levels[positions] = level
        references = np.empty((draws, len(newcomers)), dtype=np.int64)
        prefix = (json.dumps([int(seed),list(query)],sort_keys=True,separators=(',',':'))[:-1]+',').encode()
        for draw in range(draws):
            for (level, key), positions in ordered:
                pool = pool_for(level, key)
                if len(pool) == 1:
                    # A singleton permutation and every replacement draw have
                    # the same value. Streams are identity-local, so skipping
                    # this RNG cannot alter another group's draws.
                    references[draw, positions] = pool[0]
                    continue
                rng = self.draw_stream(prefix, draw, level, key)
                first = rng.permutation(pool)[:len(positions)]
                if len(first) < len(positions):
                    first = np.concatenate([first, rng.choice(pool, size=len(positions)-len(first), replace=True)])
                references[draw, positions] = first
        scale = stream(seed, list(query), 'scale').choice(old, size=min(scale_count, len(old)), replace=False)
        return {'eligible': True, 'new': newcomers.tolist(), 'old_pool': old.tolist(),
                'references': references.tolist(), 'scale': scale.tolist(), 'levels': levels.tolist()}
