"""Reproducible real-valued affine EDMD and paired MPC experiments."""

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import t as student_t


STATE_DIM = 4
D_IDX = 3
FEATURE_NAMES = ['k', 'e', 'p', 'd', 'k^2', 'e^2', 'd^2', 'k*e', 'd*e', 'k*d']
RANDOM_ROOT = 20260914
STREAM_ROLES = {
    'training': 0, 'training_controls': 1, 'evaluation': 2,
    'calibration': 3, 'test': 4, 'covariance_transform': 5, 'covariance_cohort': 6,
}
SIGN_CONVENTIONS = {
    'cohort_contrast': 'mean D_T in group 1 minus mean D_T in group 0',
    'mean_disparity': 'mean D_T over all students; not a group contrast',
    'knowledge_gap_group0_minus_group1': 'mean knowledge in group 0 minus group 1',
    'paired_projection_effects': 'MPC projected minus MPC unprojected, within the same seed and cohort',
}
PROJECTION_EFFECT_DEFINITIONS = {
    'cohort_contrast': 'Delta_DP_projected - Delta_DP_unprojected',
    'mean_disparity': 'pooled_mean_D_projected - pooled_mean_D_unprojected',
    'mean_knowledge': 'pooled_mean_knowledge_projected - pooled_mean_knowledge_unprojected',
}


def provenance():
    return {
        'revision': 'review-corrections-2026-09-14',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy': np.__version__, 'scipy': scipy.__version__,
    }


def step(states, controls, groups, noise):
    knowledge, engagement, performance, disparity = np.moveaxis(np.asarray(states), -1, 0)
    noise_k, noise_e, noise_d = np.moveaxis(np.asarray(noise), -1, 0)
    next_k = np.clip(.85 * knowledge + .15 * expit(3 * engagement - 1.5)
                     + .20 * controls - .06 * groups + noise_k, 0, 1)
    next_e = np.clip(.80 * engagement + .10 * np.tanh(performance)
                     + .15 * controls + noise_e, 0, 1)
    next_p = expit(6 * (next_k - .5))
    next_d = .90 * disparity - .05 * groups + .05 * controls + noise_d
    return np.stack([next_k, next_e, next_p, next_d], axis=-1)


def psi_raw(states):
    knowledge, engagement, performance, disparity = np.moveaxis(np.asarray(states), -1, 0)
    return np.stack([knowledge, engagement, performance, disparity, knowledge**2,
                     engagement**2, disparity**2, knowledge * engagement,
                     disparity * engagement, knowledge * disparity], axis=-1)


def stream_record(seed, role, index=0):
    if (role not in STREAM_ROLES or any(isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**32
            for value in (seed, index))):
        raise ValueError('Use a known role and unsigned 32-bit replicate/cohort indices.')
    return {'root_entropy': RANDOM_ROOT, 'replicate': int(seed), 'role': role,
            'cohort_index': int(index), 'spawn_key': [int(seed), STREAM_ROLES[role], int(index)],
            'bit_generator': 'PCG64'}


def random_stream(seed, role, index=0):
    record = stream_record(seed, role, index)
    sequence = np.random.SeedSequence(record['root_entropy'], spawn_key=tuple(record['spawn_key']))
    return np.random.Generator(np.random.PCG64(sequence))


def make_cohort(seed, students=150, horizon=25, *, role='evaluation', index=0):
    if students < 2 or students % 2 or horizon < 1:
        raise ValueError('Use an even cohort size >= 2 and a positive horizon.')
    generator = random_stream(seed, role, index)
    groups = np.repeat([0, 1], students // 2)
    initial = np.clip(np.array([.3, .3, .4, 0.])
                      + .05 * generator.standard_normal((students, STATE_DIM)), 0, 1)
    initial[:, D_IDX] = 0
    noise = generator.standard_normal((students, horizon, 3)) * [.03, .03, .02]
    return initial, noise, groups


def training_data(seed, students=150, horizon=25):
    initial, noise, groups = make_cohort(seed, students, horizon, role='training')
    controls = random_stream(seed, 'training_controls').uniform(-1, 1, (students, horizon))
    states = np.empty((students, horizon + 1, STATE_DIM))
    states[:, 0] = initial
    for time_index in range(horizon):
        states[:, time_index + 1] = step(states[:, time_index], controls[:, time_index],
                                          groups, noise[:, time_index])
    return states, controls, groups


def metric_factors(metric):
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if eigenvalues.min() <= 0:
        raise ValueError('The metric must be positive definite.')
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    return root, inverse_root


def weighted_norm(operator, metric, order='fro'):
    root, inverse_root = metric_factors(metric)
    return float(np.linalg.norm(root @ operator @ inverse_root, ord=order))


def reducing_basis(operator, metric, seed, rank=4, depth=3, tolerance=1e-8):
    dimension = len(seed)
    if not 1 <= rank <= dimension or depth < 1 or not 0 < tolerance < 1:
        raise ValueError('Invalid rank, depth, or relative singular-value tolerance.')
    root, inverse_root = metric_factors(metric)
    whitened_operator = root @ operator @ inverse_root
    whitened_seed = root @ seed
    seed_norm = np.linalg.norm(whitened_seed)
    if seed_norm <= np.finfo(float).eps:
        raise ValueError('The fitted contrast is numerically zero; no bias direction is identified.')
    seed_basis = (whitened_seed / seed_norm).reshape(-1, 1)
    closure = seed_basis
    dimensions = [1]
    for iteration in range(dimension + 1):
        candidates = np.column_stack([closure, whitened_operator @ closure,
                                      whitened_operator.T @ closure])
        left, singular, _ = np.linalg.svd(candidates, full_matrices=False)
        numeric_rank = int(np.sum(singular > tolerance * singular[0]))
        closure = left[:, :numeric_rank]
        dimensions.append(numeric_rank)
        if dimensions[-1] == dimensions[-2]:
            break
    candidates = seed_basis
    for iteration in range(depth - 1):
        candidates = np.column_stack([candidates, whitened_operator @ candidates,
                                      whitened_operator.T @ candidates])
    residual = candidates - seed_basis @ (seed_basis.T @ candidates)
    left, singular, _ = np.linalg.svd(residual, full_matrices=False)
    scale = np.linalg.norm(candidates, 2)
    numeric_rank = 1 + int(np.sum(singular > tolerance * scale))
    applied_rank = min(rank, numeric_rank)
    whitened_basis = np.column_stack([seed_basis, left[:, :applied_rank - 1]])
    basis = inverse_root @ whitened_basis
    return basis, {
        'closure_dimensions_centered_block': dimensions,
        'requested_rank': rank, 'numeric_candidate_rank': numeric_rank,
        'applied_rank': applied_rank, 'expansion_depth': depth,
        'relative_svd_tolerance': tolerance, 'residual_singular_values': singular.tolist(),
    }


@dataclass
class AffineModel:
    mean: np.ndarray
    scale: np.ndarray
    propagation: np.ndarray
    control: np.ndarray
    offset: np.ndarray
    decoder: np.ndarray
    decoder_offset: np.ndarray
    gram: np.ndarray
    metric: np.ndarray
    seed: np.ndarray
    projector: np.ndarray
    diagnostics: dict

    def encode(self, states):
        return (psi_raw(states) - self.mean) / self.scale

    def dynamics(self, projected):
        if not projected:
            return self.propagation, self.control, self.offset
        dual = self.projector.T
        return dual @ self.propagation @ dual, dual @ self.control, dual @ self.offset


def fit_model(seed=42, rank=4, depth=3, ridge_relative=1e-3, tolerance=1e-8):
    if ridge_relative <= 0:
        raise ValueError('A positive metric ridge is required for this implementation.')
    states, controls, groups = training_data(seed)
    current = states[:, :-1].reshape(-1, STATE_DIM)
    following = states[:, 1:].reshape(-1, STATE_DIM)
    raw = psi_raw(current)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale < 1e-8] = 1
    lifted = (raw - mean) / scale
    lifted_next = (psi_raw(following) - mean) / scale
    constant = np.ones((len(lifted), 1))
    regressors = np.column_stack([lifted, controls.reshape(-1), constant])
    coefficients = np.linalg.lstsq(regressors, lifted_next, rcond=None)[0]
    dimension = lifted.shape[1]
    propagation, control, offset = coefficients[:dimension].T, coefficients[dimension], coefficients[-1]
    decoder_fit = np.linalg.lstsq(np.column_stack([lifted, constant]), current, rcond=None)[0]
    decoder, decoder_offset = decoder_fit[:-1].T, decoder_fit[-1]
    gram = lifted.T @ lifted / len(lifted)
    ridge = ridge_relative * np.linalg.eigvalsh(gram).max()
    metric = gram + ridge * np.eye(dimension)
    labels = np.repeat(groups, states.shape[1] - 1)
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError('Contrast estimation requires both binary sensitive groups.')
    probability = labels.mean()
    label_score = (labels - probability) / (probability * (1 - probability))
    contrast_coefficients = np.linalg.lstsq(lifted, label_score, rcond=None)[0]
    fitted_contrast = lifted @ contrast_coefficients
    basis, diagnostics = reducing_basis(propagation.T, metric, contrast_coefficients, rank, depth, tolerance)
    projector = np.eye(dimension) - basis @ basis.T @ metric
    bias_projector = np.eye(dimension) - projector
    coefficient_operator = propagation.T
    covariance = regressors.T @ regressors / len(regressors)
    diagnostics.update({
        'identification_seed': seed,
        'training_streams': [stream_record(seed, role) for role in ('training', 'training_controls')],
        'training_trajectories': len(states), 'transitions': len(lifted),
        'contrast_reference_law': 'uniform training-trajectory/current-time occupation law under exploratory controls',
        'contrast_time_indices': list(range(states.shape[1] - 1)),
        'contrast_group_probabilities': [float(1 - probability), float(probability)],
        'centering_samples': len(current),
        'centered_feature_mean_max_abs': float(np.abs(lifted.mean(axis=0)).max()),
        'label_score_mean': float(label_score.mean()),
        'regressor_columns_including_control_and_intercept': regressors.shape[1],
        'regressor_covariance_min_eigenvalue': float(np.linalg.eigvalsh(covariance).min()),
        'gram_condition': float(np.linalg.cond(gram)), 'metric_ridge_relative': ridge_relative,
        'label_score_norm': float(np.sqrt(np.mean(label_score**2))),
        'fitted_occupation_contrast_l2': float(np.sqrt(np.mean(fitted_contrast**2))),
        'fitted_occupation_contrast_metric_norm': float(np.sqrt(contrast_coefficients @ metric @ contrast_coefficients)),
        'fitted_occupation_contrast_mean': float(fitted_contrast.mean()),
        'seed_residual': float(np.sqrt(max(0, (projector @ contrast_coefficients) @ metric @ (projector @ contrast_coefficients)))),
        'projector_idempotency': float(np.linalg.norm(projector @ projector - projector)),
        'basis_metric_orthonormality': float(np.linalg.norm(basis.T @ metric @ basis - np.eye(len(basis.T)))),
        'r1_centered_block': weighted_norm(projector @ coefficient_operator @ bias_projector, metric),
        'r2_centered_block': weighted_norm(bias_projector @ coefficient_operator @ projector, metric),
        'operator_norm_centered_block': weighted_norm(coefficient_operator, metric, 2),
        'physical_state_reconstruction_mse': float(np.mean((lifted @ decoder.T + decoder_offset - current)**2)),
        'constant_reconstruction_mse_augmented': float(np.mean((np.column_stack([lifted, constant])[:, -1] - 1)**2)),
        'removed_affine_offset_euclidean_norm': float(np.linalg.norm(offset - projector.T @ offset)),
    })
    augmented = np.block([[propagation, offset[:, None]], [np.zeros((1, dimension)), np.ones((1, 1))]])
    augmented_projector = scipy.linalg.block_diag(projector, 1.)
    augmented_metric = scipy.linalg.block_diag(metric, 1.)
    augmented_bias = np.eye(dimension + 1) - augmented_projector
    diagnostics['r1_augmented'] = weighted_norm(augmented_projector @ augmented.T @ augmented_bias, augmented_metric)
    diagnostics['r2_augmented'] = weighted_norm(augmented_bias @ augmented.T @ augmented_projector, augmented_metric)
    diagnostics['constant_fixed_error_augmented'] = float(np.linalg.norm(
        augmented_projector @ augmented.T @ augmented_projector @ np.eye(dimension + 1)[:, -1]
        - np.eye(dimension + 1)[:, -1]))
    return AffineModel(mean, scale, propagation, control, offset, decoder, decoder_offset,
                       gram, metric, contrast_coefficients, projector, diagnostics)


class MPCPolicy:
    """Exact box-constrained quadratic MPC by enumeration of active sets."""

    def __init__(self, propagation, control, offset, decoder, decoder_offset,
                 horizon=3, alpha=1., beta=8., rho=1.):
        if not 1 <= horizon <= 5 or rho <= 0 or beta < 0:
            raise ValueError('Use horizon 1--5, positive rho, and nonnegative beta.')
        dimension = len(offset)
        state_map, drift = np.eye(dimension), np.zeros(dimension)
        influence = np.zeros((dimension, horizon))
        hessian = 2 * rho * np.eye(horizon)
        gradient_map = np.zeros((horizon, dimension))
        gradient_offset = np.zeros(horizon)
        for time_index in range(horizon):
            state_map = propagation @ state_map
            drift = propagation @ drift + offset
            influence = propagation @ influence
            influence[:, time_index] += control
            disparity_influence = decoder[D_IDX] @ influence
            hessian += 2 * beta * np.outer(disparity_influence, disparity_influence)
            gradient_map += 2 * beta * np.outer(disparity_influence, decoder[D_IDX] @ state_map)
            gradient_offset += (2 * beta * disparity_influence
                                * (decoder[D_IDX] @ drift + decoder_offset[D_IDX])
                                - alpha * (decoder[0] @ influence))
        maps, offsets = [], []
        for pattern in itertools.product([-1, 0, 1], repeat=horizon):
            active = np.asarray(pattern, dtype=float)
            free = np.flatnonzero(active == 0)
            solution_map = np.zeros((horizon, horizon))
            if len(free):
                solution_map[np.ix_(free, free)] = -np.linalg.inv(hessian[np.ix_(free, free)])
            maps.append(solution_map)
            offsets.append(active + solution_map @ hessian @ active)
        self.hessian = hessian
        self.gradient_map, self.gradient_offset = gradient_map, gradient_offset
        self.solution_maps, self.solution_offsets = np.asarray(maps), np.asarray(offsets)

    def sequence(self, latent):
        gradient = np.asarray(latent) @ self.gradient_map.T + self.gradient_offset
        candidates = np.einsum('aij,...j->...ai', self.solution_maps, gradient) + self.solution_offsets
        feasible = np.all(np.abs(candidates) <= 1 + 1e-9, axis=-1)
        costs = .5 * np.einsum('...ai,ij,...aj->...a', candidates, self.hessian, candidates)
        costs += np.einsum('...ai,...i->...a', candidates, gradient)
        costs = np.where(feasible, costs, np.inf)
        best = np.argmin(costs, axis=-1)
        return np.clip(np.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :], -1, 1)

    def __call__(self, latent):
        return self.sequence(latent)[..., 0]


def make_policy(model, projected):
    return MPCPolicy(*model.dynamics(projected), model.decoder, model.decoder_offset)


def forecast_terminal(model, initial, horizon, policy, projected=False):
    """Forecast from initial observations only, without future states or controls.

    z_hat follows the surrogate dynamics and the same fixed MPC decision rule.
    It is never used as the observed state supplied to the deployed controller.
    """
    propagation, control, offset = model.dynamics(projected)
    z_hat = model.encode(np.asarray(initial))
    for _ in range(horizon):
        forecast_action = np.zeros(len(z_hat)) if policy is None else policy(z_hat)
        z_hat = z_hat @ propagation.T + forecast_action[:, None] * control + offset
    return z_hat @ model.decoder.T + model.decoder_offset


def rollout(model, initial, noise, groups, policy, projected=False):
    """Generate deployment trajectories using z_obs = Psi(x_t) at every step.

    The terminal forecast is computed independently from the initial states.
    The shared MPCPolicy is a deterministic, stateless decision rule, used
    unchanged by the main experiment, calibration, and held-out testing.
    """
    physical = np.asarray(initial).copy()
    states = [physical.copy()]
    controls = []
    for time_index in range(noise.shape[1]):
        z_obs = model.encode(physical)
        action = np.zeros(len(physical)) if policy is None else policy(z_obs)
        controls.append(action)
        physical = step(physical, action, groups, noise[:, time_index])
        states.append(physical.copy())
    decoded = forecast_terminal(model, initial, noise.shape[1], policy, projected)
    return np.stack(states, axis=1), decoded, np.stack(controls, axis=1)


def contrast(values, groups):
    groups = np.asarray(groups)
    if not np.any(groups == 0) or not np.any(groups == 1):
        raise ValueError('Both groups must have positive counts.')
    return np.asarray(values)[groups == 1].mean(axis=0) - np.asarray(values)[groups == 0].mean(axis=0)


def basis_covariance_check(model):
    dimension = len(model.seed)
    replicate = model.diagnostics['identification_seed']
    generator = random_stream(replicate, 'covariance_transform')
    rotation = np.linalg.qr(generator.normal(size=(dimension, dimension)))[0]
    transform = np.diag(np.linspace(.6, 1.8, dimension)) @ rotation
    inverse = np.linalg.inv(transform)
    metric = transform @ model.metric @ transform.T
    operator = inverse.T @ model.propagation.T @ transform.T
    seed = inverse.T @ model.seed
    basis, _ = reducing_basis(operator, metric, seed, model.diagnostics['requested_rank'],
                              model.diagnostics['expansion_depth'], model.diagnostics['relative_svd_tolerance'])
    projector = np.eye(dimension) - basis @ basis.T @ metric
    expected = inverse.T @ model.projector @ transform.T
    propagation = transform @ model.propagation @ inverse
    control, offset = transform @ model.control, transform @ model.offset
    decoder = model.decoder @ inverse
    transformed_policy = MPCPolicy(projector.T @ propagation @ projector.T,
                                    projector.T @ control, projector.T @ offset,
                                    decoder, model.decoder_offset)
    initial, noise, groups = make_cohort(replicate, role='covariance_cohort')
    original_states, _, original_controls = rollout(model, initial, noise, groups, make_policy(model, True), True)
    transported_rule = lambda latent: transformed_policy(latent @ transform.T)
    transformed_states, _, transformed_controls = rollout(model, initial, noise, groups, transported_rule, True)
    return {'transported_projector_max_error': float(np.max(np.abs(projector - expected))),
            'covariant_control_max_error': float(np.max(np.abs(original_controls - transformed_controls))),
            'covariant_physical_trajectory_max_error': float(np.max(np.abs(original_states - transformed_states))),
            'covariant_disparity_trajectory_max_error': float(np.max(np.abs(original_states[:, :, D_IDX] - transformed_states[:, :, D_IDX]))),
            'covariant_cohort_contrast_max_error': float(np.max(np.abs(
                contrast(original_states[:, :, D_IDX], groups) - contrast(transformed_states[:, :, D_IDX], groups))))}


def paired_projection_effects(projected_metrics, unprojected_metrics):
    """Subtract matched arms for each estimand, never substituting pooled D for its contrast."""
    if projected_metrics.keys() != unprojected_metrics.keys():
        raise ValueError('Both arms must report the same metrics.')
    return {name: projected_metrics[name] - unprojected_metrics[name]
            for name in projected_metrics}


def summarize_seed_values(values):
    """Descriptive seed-level statistics; paired effects must be differenced first.

    The Student-t interval describes the mean across replicate seeds under
    independent, approximately normal replicate effects. It is not a confidence
    interval for demographic parity in an external student population.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two finite seed-level values are required.')
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    standard_error = standard_deviation / np.sqrt(len(values))
    halfwidth = float(student_t.ppf(.975, len(values) - 1) * standard_error)
    return {'count': len(values), 'mean': mean, 'sample_sd': standard_deviation,
            'standard_error': float(standard_error), 't95_halfwidth': halfwidth,
            't95_interval': [mean - halfwidth, mean + halfwidth],
            'minimum': float(values.min()), 'maximum': float(values.max())}


def run_experiment(seed=42, rank=4, depth=3, tolerance=1e-8):
    model = fit_model(seed, rank, depth, tolerance=tolerance)
    initial, noise, groups = make_cohort(seed)
    trajectories, metrics = {}, {}
    for name, projected, controlled in [('zero_raw', False, False), ('zero_projected', True, False),
                                         ('mpc_projected', True, True), ('mpc_raw', False, True)]:
        policy = make_policy(model, projected) if controlled else None
        states, _, _ = rollout(model, initial, noise, groups, policy, projected)
        trajectories[name] = states
        terminal = states[:, -1]
        metrics[name] = {
            'cohort_contrast': float(contrast(terminal[:, D_IDX], groups)),
            'mean_disparity': float(terminal[:, D_IDX].mean()),
            'mean_knowledge': float(terminal[:, 0].mean()),
            'knowledge_gap_group0_minus_group1': float(-contrast(terminal[:, 0], groups)),
            'knowledge_group0': float(terminal[groups == 0, 0].mean()),
            'knowledge_group1': float(terminal[groups == 1, 0].mean()),
        }
    effects = paired_projection_effects(metrics['mpc_projected'], metrics['mpc_raw'])
    result = {'provenance': provenance(), 'seed': seed,
              'random_streams': {role: stream_record(seed, role) for role in (
                  'training', 'training_controls', 'evaluation', 'covariance_transform', 'covariance_cohort')},
              'sign_conventions': SIGN_CONVENTIONS,
              'projection_effect_definitions': PROJECTION_EFFECT_DEFINITIONS,
              'diagnostics': model.diagnostics, 'methods': metrics,
              'paired_projection_effects': effects, 'basis_covariance': basis_covariance_check(model)}
    return model, result, trajectories, groups


def run_multiseed(seeds, rank=4, depth=3, tolerance=1e-8):
    seeds = list(seeds)
    if (len(seeds) < 2 or any(isinstance(seed, (bool, np.bool_))
                            or not isinstance(seed, (int, np.integer)) or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Use at least two distinct nonnegative integer seeds.')
    seeds = [int(seed) for seed in seeds]
    results = [run_experiment(seed, rank, depth, tolerance)[1] for seed in seeds]
    metrics = results[0]['paired_projection_effects'].keys()
    effects = {name: summarize_seed_values([result['paired_projection_effects'][name]
                                           for result in results]) for name in metrics}
    return {'provenance': provenance(), 'seeds': seeds,
            'sign_conventions': SIGN_CONVENTIONS,
            'projection_effect_definitions': PROJECTION_EFFECT_DEFINITIONS,
            'uncertainty': 'Sample SD (ddof=1), SE=SD/sqrt(n), two-sided Student-t 95% interval with n-1 degrees of freedom; descriptive seed sensitivity only.',
            'paired_projection_effects': effects, 'per_seed': results}


def plot_results(model, trajectories, groups, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    angles = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(np.cos(angles), np.sin(angles), 'k--', alpha=.4)
    for projected, name in [(False, 'raw'), (True, 'projected')]:
        eigenvalues = np.linalg.eigvals(model.dynamics(projected)[0])
        axes[0].scatter(eigenvalues.real, eigenvalues.imag, label=name)
    axes[0].set(xlabel='Real part', ylabel='Imaginary part', title='Centered propagation block')
    axes[0].set_aspect('equal')
    for name, states in trajectories.items():
        axes[1].plot(contrast(states[:, :, D_IDX], groups), label=name)
        axes[2].plot(states[:, :, 0].mean(axis=0), label=name)
    axes[1].set(xlabel='Turn', ylabel='Empirical contrast: group 1 - group 0', title='True paired-cohort outcomes')
    axes[2].set(xlabel='Turn', ylabel='Mean knowledge', title='True utility')
    for axis in axes:
        axis.legend(fontsize=7)
    figure.suptitle(f"Corrected affine experiment; identification seed {model.diagnostics['identification_seed']}")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument('--seed', type=int, default=int(os.environ.get('FA_EDMD_RUN_SEED', '42')))
    seed_options.add_argument('--seeds', type=int, nargs='+', help='Distinct paired replicate seeds for a multiseed summary.')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--svd-tolerance', type=float, default=1e-8)
    parser.add_argument('--output', type=Path, default=Path('corrected-results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.seeds is not None:
        result = run_multiseed(args.seeds, args.rank, args.depth, args.svd_tolerance)
        output = args.output / 'edmd-multiseed-summary.json'
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result['paired_projection_effects'], indent=2))
        print(f'Summary and paired per-seed results saved to {output}')
        return
    model, result, trajectories, groups = run_experiment(args.seed, args.rank, args.depth, args.svd_tolerance)
    (args.output / f'edmd-seed-{args.seed}.json').write_text(json.dumps(result, indent=2) + '\n')
    plot_results(model, trajectories, groups, args.output / f'edmd-seed-{args.seed}.png')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()