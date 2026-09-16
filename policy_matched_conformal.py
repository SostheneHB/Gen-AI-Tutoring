"""Split-conformal prediction of a new cohort's empirical contrast.

Actual calibration and test trajectories use the deployment rollout: the MPC
receives z_obs = Psi(x_t) after each real observation. The separate z_hat
forecast uses only initial observations and surrogate dynamics. It neither
drives the real system nor consumes its future observations or controls.
Coverage pertains to delta_new, a realized empirical contrast under this fixed
policy and cohort construction, not to the fixed population parameter Delta_DP.
The closed interval includes tied scores; no random tie-breaking is needed for
the conservative marginal guarantee. Positive group counts are enforced by the
balanced cohort generator and contrast function.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from fa_edmd_mpc_v3_corrected import contrast, fit_model, make_cohort, make_policy, provenance, rollout, stream_record


def conformal_quantile(scores, alpha):
    """Ceiling order for closed prediction intervals, including tied scores.

    Use abs(delta_new - predicted_delta_new) <= quantile for membership.
    The infinite endpoint is required when the order exceeds calibration size.
    This quantile alone is not a population-mean confidence interval.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError('Calibration scores must be a nonempty finite nonnegative vector.')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between zero and one.')
    order = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return order, float(np.sort(scores)[order - 1]) if order <= len(scores) else np.inf


def cohort_pair(model, policy, seed, students=150, horizon=25, *, role='calibration', index=0):
    """Pair the real terminal contrast with its initial-state-only forecast.

    Reuse the main experiment's rollout rather than implementing another
    controller loop here. Its returned prediction is produced independently
    by forecast_terminal; it is not the state used for real feedback control.
    """
    initial, noise, groups = make_cohort(seed, students, horizon, role=role, index=index)
    states, prediction, _ = rollout(model, initial, noise, groups, policy, projected=True)
    return float(contrast(states[:, -1, 3], groups)), float(contrast(prediction[:, 3], groups))


def run_calibration(model=None, calibration_count=100, test_count=100, alpha=.05, seed=42):
    if any(isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
           or count < 1 for count in (calibration_count, test_count)):
        raise ValueError('Calibration and test counts must be positive integers.')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between zero and one.')
    if max(calibration_count, test_count) > 2**32:
        raise ValueError('Cohort indices must fit in unsigned 32-bit integers.')
    model = fit_model(seed) if model is None else model
    replicate = model.diagnostics['identification_seed']
    expected = [stream_record(replicate, role) for role in ('training', 'training_controls')]
    if model.diagnostics.get('training_streams') != expected:
        raise ValueError('Calibration requires a model fitted under the current stream protocol.')
    policy = make_policy(model, True)
    calibration = np.array([cohort_pair(model, policy, replicate, role='calibration', index=index)
                            for index in range(calibration_count)])
    scores = np.abs(calibration[:, 0] - calibration[:, 1])
    order, quantile = conformal_quantile(scores, alpha)
    test = np.array([cohort_pair(model, policy, replicate, role='test', index=index)
                    for index in range(test_count)])
    test_scores = np.abs(test[:, 0] - test[:, 1])
    covered = test_scores <= quantile
    test_intervals = ([[None, None] for _ in test] if np.isinf(quantile)
                      else np.column_stack((test[:, 1] - quantile, test[:, 1] + quantile)).tolist())
    coverage = float(covered.mean())
    normal_quantile = 1.959963984540054
    denominator = 1 + normal_quantile**2 / test_count
    center = (coverage + normal_quantile**2 / (2 * test_count)) / denominator
    radius = normal_quantile * np.sqrt(coverage * (1 - coverage) / test_count
                                       + normal_quantile**2 / (4 * test_count**2)) / denominator
    return {
        'provenance': provenance(),
        'calibration_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'identification_seed': model.diagnostics['identification_seed'],
        'training_streams': expected,
        'target': 'new balanced cohort empirical disparity contrast, group 1 minus group 0',
        'target_symbol': 'delta_new',
        'target_definition': 'mean observed terminal D in group 1 minus mean observed terminal D in group 0',
        'population_parameter': 'Delta_DP is fixed conditional on the policy and population law; not covered by this prediction claim',
        'coverage_scope': 'marginal over calibration and new-cohort randomness conditional on independent training; not conditional on each cohort or simultaneous over test cohorts',
        'cohort_construction': 'same balanced generator, initial/noise law, horizon and fixed observed-feedback policy for calibration and test',
        'group_counts_per_cohort': {'0': 75, '1': 75},
        'pair_columns': ['delta_observed', 'delta_predicted_from_initial_information'],
        'tie_rule': 'closed interval; score <= quantile, including equality',
        'population_sampling_error_control': None,
        'coverage_wilson95_target': 'test-cohort coverage frequency for the realized calibration interval, not Delta_DP',
        'policy': 'observed-state feedback MPC with projected affine model',
        'prediction': 'initial-state-only surrogate forecast of the same feedback decision rule; no future observed controls',
        'calibration_cohorts': calibration_count, 'test_cohorts': test_count,
        'students_per_cohort': 150, 'horizon': 25, 'alpha': alpha, 'order': order,
        'quantile': quantile if np.isfinite(quantile) else None,
        'unbounded_band': bool(np.isinf(quantile)),
        'coverage': coverage, 'coverage_wilson95': [center - radius, center + radius],
        'calibration_true_predicted': calibration.tolist(),
        'calibration_scores': scores.tolist(), 'test_true_predicted': test.tolist(),
        'test_scores': test_scores.tolist(), 'test_intervals': test_intervals,
        'test_covered': covered.tolist(),
        'calibration_streams': [stream_record(replicate, 'calibration', index)
                                for index in range(calibration_count)],
        'test_streams': [stream_record(replicate, 'test', index) for index in range(test_count)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--calibration-cohorts', type=int, default=int(os.environ.get('CONFORMAL_N_CAL', '100')))
    parser.add_argument('--test-cohorts', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=float(os.environ.get('CONFORMAL_ALPHA', '.05')))
    parser.add_argument('--output', type=Path, default=Path('corrected-results/conformal.json'))
    args = parser.parse_args()
    result = run_calibration(calibration_count=args.calibration_cohorts, test_count=args.test_cohorts,
                              alpha=args.alpha, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"Coverage: {result['coverage']:.3f}; quantile: {result['quantile']}")


if __name__ == '__main__':
    main()