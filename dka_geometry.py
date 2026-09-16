"""Real coefficient-space geometry for the DKA occupation contrast."""

import numpy as np


def fit_bias_geometry(features, sensitive, coefficient_operator, rank=4,
                      depth=3, ridge_rel=1e-3, tolerance=1e-8):
    """Fit on training transitions only; preserve the contrast when truncating.

    The metric belongs to centered feature coefficients. Row-wise centered
    feature values are projected by right multiplication with ``projector``.
    Finite mixed-word candidates are not claimed to be an exact reducing space.
    """
    features = np.asarray(features, dtype=float)
    sensitive = np.asarray(sensitive, dtype=float).reshape(-1)
    operator = np.asarray(coefficient_operator, dtype=float)
    if features.ndim != 2 or len(features) != len(sensitive) or not len(features):
        raise ValueError('Expected nonempty aligned features and group labels.')
    dimension = features.shape[1]
    if operator.shape != (dimension, dimension):
        raise ValueError('The coefficient operator has the wrong shape.')
    if not all(np.isfinite(values).all() for values in (features, sensitive, operator)):
        raise ValueError('Geometry inputs must be finite.')
    if not np.isin(sensitive, [0, 1]).all() or np.unique(sensitive).size != 2:
        raise ValueError('Both binary sensitive groups must be present.')
    if (not isinstance(rank, int) or not 1 <= rank <= dimension
            or not isinstance(depth, int) or depth < 0
            or not np.isfinite(ridge_rel) or ridge_rel <= 0
            or not np.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError('Invalid rank, depth, ridge, or SVD tolerance.')
    mean = features.mean(axis=0)
    centered = features - mean
    probability = sensitive.mean()
    label_score = (sensitive - probability) / (probability * (1 - probability))
    contrast = np.linalg.lstsq(centered, label_score, rcond=tolerance)[0]
    covariance = centered.T @ centered / len(centered)
    ridge = ridge_rel * max(float(np.linalg.eigvalsh(covariance)[-1]), 1e-12)
    metric = covariance + ridge * np.eye(dimension)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    whitened_operator = root @ operator @ inverse_root
    seed = root @ contrast
    seed_norm = np.linalg.norm(seed)

    def orthonormalize(columns):
        left, singular, _ = np.linalg.svd(columns, full_matrices=False)
        if not singular.size or singular[0] == 0:
            return left[:, :0]
        return left[:, singular > tolerance * singular[0]]

    initial = seed[:, None] / seed_norm if seed_norm > 0 else np.empty((dimension, 0))
    closure = initial
    closure_dimensions = [closure.shape[1]]
    for _ in range(dimension):
        expanded = orthonormalize(np.column_stack((
            closure, whitened_operator @ closure, whitened_operator.T @ closure)))
        closure_dimensions.append(expanded.shape[1])
        if expanded.shape[1] == closure.shape[1]:
            closure = expanded
            break
        closure = expanded
    candidates = initial
    for _ in range(depth):
        family = np.column_stack((
            candidates, whitened_operator @ candidates,
            whitened_operator.T @ candidates))
        left, singular, _ = np.linalg.svd(family, full_matrices=False)
        retained = singular > tolerance * singular[0] if singular.size else np.zeros(0, dtype=bool)
        candidates = left[:, retained] * singular[retained]
    numerical_rank = candidates.shape[1]
    residual_singular = np.empty(0)
    if seed_norm > 0:
        residual = candidates - initial @ (initial.T @ candidates)
        left, singular, _ = np.linalg.svd(residual, full_matrices=False)
        residual_singular = singular
        cutoff = tolerance * np.linalg.norm(candidates, ord=2)
        extra_rank = min(rank - 1, int(np.sum(singular > cutoff)))
        selected = np.column_stack((initial, left[:, :extra_rank]))
        selected, _ = np.linalg.qr(selected, mode='reduced')
    else:
        selected = initial
    basis = inverse_root @ selected
    bias_projector = basis @ basis.T @ metric
    projector = np.eye(dimension) - bias_projector

    def weighted_norm(matrix):
        return float(np.linalg.norm(root @ matrix @ inverse_root, ord='fro'))

    diagnostics = {
        'candidate_family': 'seed plus mixed words in the whitened coefficient operator and its transpose',
        'candidate_depth_definition': 'maximum mixed-word length before seed-preserving rank truncation',
        'closure_dimensions': closure_dimensions,
        'candidate_depth': depth,
        'svd_tolerance': tolerance,
        'requested_rank': rank,
        'numerical_candidate_rank': numerical_rank,
        'residual_candidate_singular_values': residual_singular.tolist(),
        'applied_rank': basis.shape[1],
        'metric_ridge': ridge,
        'metric_condition': float(np.linalg.cond(metric)),
        'label_score_norm': float(np.linalg.norm(label_score) / np.sqrt(len(label_score))),
        'fitted_contrast_norm': float(np.linalg.norm(centered @ contrast) / np.sqrt(len(centered))),
        'fitted_contrast_mean': float(np.mean(centered @ contrast)),
        'seed_residual': float(np.linalg.norm(root @ projector @ contrast)),
        'relative_seed_residual': float(np.linalg.norm(root @ projector @ contrast) / seed_norm)
        if seed_norm > 0 else 0.0,
        'projector_idempotence': float(np.linalg.norm(projector @ projector - projector)),
        'weighted_self_adjointness': float(np.linalg.norm(projector.T @ metric - metric @ projector)),
        'leak_bias_to_complement': weighted_norm(projector @ operator @ bias_projector),
        'leak_complement_to_bias': weighted_norm(bias_projector @ operator @ projector),
        'operator_compression_change': weighted_norm(operator - projector @ operator @ projector),
        'bias_block_norm': weighted_norm(bias_projector @ operator @ bias_projector),
    }
    return mean, metric, basis, projector, diagnostics