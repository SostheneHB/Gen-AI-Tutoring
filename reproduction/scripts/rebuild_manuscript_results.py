"""Publish a matched EDMD/conformal run without overwriting historical artifacts."""

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np

from fa_edmd_mpc_v3_corrected import (
    D_IDX, RANDOM_ROOT, STREAM_ROLES, contrast, plot_results, run_experiment, run_multiseed,
)
from policy_matched_conformal import run_calibration


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def latex_number(value):
    if not np.isfinite(value):
        raise ValueError('Only finite values can be rendered as scientific diagnostics.')
    text = f'{value:.3g}'
    if 'e' not in text:
        return text
    mantissa, exponent = text.split('e')
    return rf'{mantissa}\times10^{{{int(exponent)}}}'


def render_text(result, output):
    """Render current manuscript numbers from the saved result, not a new run."""
    experiment = result['experiment']
    diagnostics = experiment['diagnostics']
    calibration = result['conformal']
    rows = [r'\begin{table}[htbp]', r'\centering\small',
            r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
            r'Method & $\delta$ & $\overline D_T$ & $\overline k_T$ & $k_0-k_1$ \\',
            r'\midrule']
    for name, metrics in experiment['methods'].items():
        label = name.replace('_', r'\_')
        rows.append(label + ' & ' + ' & '.join(f'{metrics[key]:+.5f}' for key in (
            'cohort_contrast', 'mean_disparity', 'mean_knowledge',
            'knowledge_gap_group0_minus_group1')) + r' \\')
    effects = experiment['paired_projection_effects']
    rows.extend([r'\bottomrule', r'\end{tabular}',
                 r'\caption{Current matched run: identification seed '
                 + str(experiment['seed']) + ', evaluation role 2, cohort index 0'
                 + r'. Terminal contrasts use group 1 minus group 0; the last column is the knowledge gap in the opposite direction.}',
                 r'\label{tab:current-edmd}', r'\end{table}',
                 'The paired projected-minus-unprojected MPC effect is '
                 + rf'\({effects["cohort_contrast"]:+.6f}\) on the group contrast and '
                 + rf'\({effects["mean_disparity"]:+.6f}\) on the pooled mean disparity. '
                 + 'These are single-cohort effects, not uncertainty estimates.'])
    (output / 'table.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    (output / 'diagnostics.tex').write_text(
        'For the current matched run, the centered-block weighted Hilbert--Schmidt '
        + rf'residuals are \(r_1={diagnostics["r1_centered_block"]:.6f}\), '
        + rf'\(r_2={diagnostics["r2_centered_block"]:.6f}\), and '
        + rf'\(\|K\|_{{G_{{\mathrm{{reg}}}},2}}={diagnostics["operator_norm_centered_block"]:.6f}\). '
        + 'The augmented residuals (including the affine constant coordinate) are '
        + rf'\({diagnostics["r1_augmented"]:.6f}\) and \({diagnostics["r2_augmented"]:.6f}\). '
        + rf'The centered seed residual is \({latex_number(diagnostics["seed_residual"])}\). '
        + 'None of these matrix diagnostics is a bound on physical MPC disparities without '
        + 'the model and reference-law error terms of Theorem~\\ref{thm:terminal_fairness}.\n',
        encoding='utf-8')
    interval = calibration['coverage_wilson95']
    (output / 'conformal.tex').write_text(
        f'The current matched run uses identification seed {experiment["seed"]}, '
        + f'{calibration["calibration_cohorts"]} calibration cohorts and '
        + f'{calibration["test_cohorts"]} disjoint test cohorts, each of '
        + f'{calibration["students_per_cohort"]} students over {calibration["horizon"]} turns. '
        + rf'At \(\alpha={calibration["alpha"]:.2f}\), the order is \(k={calibration["order"]}\) '
        + rf'and the recalculated quantile is \(\hat q={calibration["quantile"]:.6f}\). '
        + f'{sum(calibration["test_covered"])} of {calibration["test_cohorts"]} test contrasts are covered '
        + rf'(empirical coverage \({calibration["coverage"]:.3f}\); Wilson 95\% interval '
        + rf'\([{interval[0]:.3f},{interval[1]:.3f}]\)). '
        + 'The interval summarizes repeated test-cohort coverage, not uncertainty in a population '
        + 'fairness effect. Wide prediction bands do not certify small disparities.\n', encoding='utf-8')
    covariance = experiment['basis_covariance']
    (output / 'covariance.tex').write_text(
        'For this transported-metric test, the maximum absolute discrepancies are '
        + rf'\({latex_number(covariance["transported_projector_max_error"])}\) for the coefficient projector, '
        + rf'\({latex_number(covariance["covariant_control_max_error"])}\) for controls, '
        + rf'\({latex_number(covariance["covariant_physical_trajectory_max_error"])}\) for physical trajectories, '
        + rf'\({latex_number(covariance["covariant_disparity_trajectory_max_error"])}\) for disparity trajectories, and '
        + rf'\({latex_number(covariance["covariant_cohort_contrast_max_error"])}\) for group contrasts. '
        + 'This is one numerical covariance test, not a proof for arbitrary transformations or independent refits.\n',
        encoding='utf-8')


def render_multiseed(summary, output):
    statistics = summary['paired_projection_effects']['cohort_contrast']
    lower, upper = statistics['t95_interval']
    (output / 'multiseed.tex').write_text(
        'The new role-separated study uses replicate identifiers '
        + ', '.join(str(seed) for seed in summary['seeds']) + '. '
        + r'The paired contrast effect \(\tau_{\mathrm{proj},\delta}\) has '
        + rf'mean \({statistics["mean"]:+.6f}\), sample SD \({statistics["sample_sd"]:.6f}\), '
        + rf'and a Student-\(t\) 95\% interval for the replicate mean '
        + rf'\([{lower:+.6f},{upper:+.6f}]\). '
        + rf'The effects range from \({statistics["minimum"]:+.6f}\) to \({statistics["maximum"]:+.6f}\). '
        + 'The interval assumes independent, approximately normal replicate effects; '
        + 'stream separation removes the previous exact seed reuse but does not establish normality '
        + 'or a population fairness guarantee.\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('corrected-results/2026-09-14'))
    args = parser.parse_args()
    if args.output.is_absolute() or '..' in args.output.parts:
        parser.error('Use a workspace-relative output directory.')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Choose a new empty output directory; existing results are preserved.')
    sources = [Path(name) for name in (
        'fa_edmd_mpc_v3_corrected.py', 'policy_matched_conformal.py',
        'rebuild_manuscript_results.py')]
    source_hashes = {str(path): digest(path) for path in sources}
    config = {'identification_seed': 0, 'random_root_entropy': RANDOM_ROOT,
              'stream_roles': STREAM_ROLES,
              'stream_key_layout': ['replicate', 'role', 'cohort_index'],
              'multiseed_replicates': list(range(10)),
              'rank': 4, 'depth': 3, 'svd_tolerance': 1e-8, 'metric_ridge_relative': 1e-3,
              'students': 150, 'horizon': 25, 'mpc_horizon': 3,
              'mpc_alpha': 1., 'mpc_beta': 8., 'mpc_rho': 1., 'control_bounds': [-1., 1.],
              'calibration_cohorts': 100, 'test_cohorts': 100, 'alpha': .05,
              'covariance_transform_scales': [.6, 1.8], 'covariance_metric': 'T @ G_reg @ T.T'}
    model, experiment, trajectories, groups = run_experiment(
        config['identification_seed'], config['rank'], config['depth'], config['svd_tolerance'])
    calibration = run_calibration(model=model, calibration_count=config['calibration_cohorts'],
                                  test_count=config['test_cohorts'], alpha=config['alpha'])
    summary = run_multiseed(config['multiseed_replicates'], config['rank'], config['depth'],
                           config['svd_tolerance'])
    result = {'configuration': config, 'source_sha256': source_hashes,
              'experiment': experiment, 'conformal': calibration,
              'series': {name: {'cohort_contrast': contrast(states[:, :, D_IDX], groups).tolist(),
                                'mean_knowledge': states[:, :, 0].mean(axis=0).tolist()}
                         for name, states in trajectories.items()}}
    if source_hashes != {str(path): digest(path) for path in sources}:
        raise RuntimeError('Sources changed during execution; rerun before publishing.')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'result.json', result)
    write_json(args.output / 'edmd-multiseed-summary.json', summary)
    np.savez_compressed(args.output / 'trajectories.npz', groups=groups, **trajectories)
    plot_results(model, trajectories, groups, args.output / 'edmd.png')
    render_text(json.loads((args.output / 'result.json').read_text(encoding='utf-8')), args.output)
    render_multiseed(summary, args.output)
    (args.output / 'sources').mkdir()
    for source in sources:
        shutil.copyfile(source, args.output / 'sources' / source.name)
    archives = [Path(name) for name in ('fa_edmd_mpc_v3_corrected_results.txt', 'fa_edmd_seed_0.txt',
                                        'policy_matched_conformal_results.txt', 'fa_edmd_mpc_results_v3.png',
                                        'dashboard_before_fix.png', 'dashboard_after_fix.png')]
    import matplotlib
    write_json(args.output / 'manifest.json', {
        'configuration': config, 'source_sha256': source_hashes,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': experiment['provenance']['scipy'], 'matplotlib': matplotlib.__version__},
        'historical_sha256': {str(path): digest(path) for path in archives},
        'historical_main_equals_seed0': archives[0].read_bytes() == archives[1].read_bytes(),
        'artifact_sha256': {str(path.relative_to(args.output)): digest(path)
                            for path in sorted(args.output.rglob('*')) if path.is_file()},
    })
    print(f'Matched artifacts saved to {args.output}; no historical files overwritten.')


if __name__ == '__main__':
    main()