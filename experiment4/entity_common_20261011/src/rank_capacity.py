"""TRAIN-label certificates for a strict, saturated old top-k list.

These certificates concern frozen old candidate scores. They do not authorize
discarding old truth constraints or prove utility for unseen new answers.
"""
import numpy as np


def saturated_mask(keys, members, old_truth):
    return np.asarray([all(int(g) in old_truth.get(tuple(q), ()) for g in row)
                       for q, row in zip(keys, members)], dtype=bool)


def score_certificate(scores, members, absolute_dot_sum, dimension):
    """Conservative FP32 multiply/dot error separates the complete top-k set.

    Old E/R are exactly represented float32 values promoted to float64. Gamma
    includes rounded query multiplication and a length-dimension dot product.
    CUDA TF32 must be disabled; overflow/subnormal behavior is not covered.
    Internal ties among members are allowed; ties across the boundary fail.
    """
    scores = np.asarray(scores, dtype=np.float64)
    members = np.asarray(members, dtype=np.int64)
    assert len(np.unique(members)) == len(members)
    assert np.isfinite(scores).all() and np.isfinite(absolute_dot_sum).all()
    other = np.ones(len(scores), dtype=bool); other[members] = False
    gap = float(scores[members].min() - scores[other].max())
    u32 = 2.0 ** -24; u64 = 2.0 ** -53
    gamma32 = dimension * u32 / (1 - dimension * u32)
    gamma64 = dimension * u64 / (1 - dimension * u64)
    # Bound both the FP32 score and the computed FP64 reference error.
    coefficient = gamma32 + u32 + gamma32 * u32 + gamma64 + u64 + gamma64 * u64
    error = float(np.max(absolute_dot_sum) * coefficient)
    error = float(np.nextafter(error * (1 + 8 * gamma64), np.inf))
    return dict(strict_gap=gap, max_score_error_bound=error,
                robust_boundary=bool(gap > 2 * error))


def selfcheck():
    q = np.array([[0, 0]])
    members = np.array([[0, 1]])
    assert saturated_mask(q, members, {(0, 0): {0, 1}}).item()
    assert not saturated_mask(q, members, {(0, 0): {0}}).item()
    assert score_certificate([4., 3., 2.], [0, 1], [4., 3., 2.], 2)['robust_boundary']
    assert not score_certificate([4., 3., 3.], [0, 1], [4., 3., 3.], 2)['robust_boundary']
    assert not score_certificate([4., 3., 3.-1e-7], [0, 1], [4., 3., 3.], 200)['robust_boundary']
    assert not score_certificate([4., 3., 5.], [0, 1], [4., 3., 5.], 2)['robust_boundary']
    return dict(controls=6, status='PASS_SATURATION_AND_BOUNDARY_CONTROLS')
