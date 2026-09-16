"""Held-out prediction diagnostics, kept independent of the training framework."""

import numpy as np


def prediction_metrics(predicted, observed, sensitive):
    """Compare aligned physical states; all signed gaps use group 1 minus 0."""
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    labels = np.asarray(sensitive)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if (predicted.ndim != 2 or predicted.shape != observed.shape
            or predicted.shape[1] < 1 or labels.ndim != 1 or len(labels) != len(predicted)):
        raise ValueError('Expected aligned sample-by-coordinate states and one label per sample.')
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError('Predicted and observed states must be finite.')
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError('Both binary sensitive groups must be present.')
    errors = predicted - observed
    squared_errors = errors ** 2
    result = {'mse': float(squared_errors.mean()),
              'mse_by_coordinate': squared_errors.mean(axis=0).tolist()}
    for group in (0, 1):
        selected = labels == group
        result[f'count_group{group}'] = int(selected.sum())
        result[f'predicted_ability_mean_group{group}'] = float(predicted[selected, 0].mean())
        result[f'real_ability_mean_group{group}'] = float(observed[selected, 0].mean())
        result[f'mse_group{group}'] = float(squared_errors[selected].mean())
        result[f'mse_by_coordinate_group{group}'] = squared_errors[selected].mean(axis=0).tolist()
        result[f'ability_mse_group{group}'] = float(squared_errors[selected, 0].mean())
        result[f'ability_bias_group{group}'] = float(errors[selected, 0].mean())
    result['predicted_ability_gap'] = (result['predicted_ability_mean_group1']
                                       - result['predicted_ability_mean_group0'])
    result['real_ability_gap'] = result['real_ability_mean_group1'] - result['real_ability_mean_group0']
    result['ability_gap_error'] = result['predicted_ability_gap'] - result['real_ability_gap']
    return result


def plot_results(history, records, output):
    """Plot every evaluated starting time, real contrasts, and group errors."""
    if not history or not records or any(row.get('split') != 'test' for row in records):
        raise ValueError('A test dashboard requires loss history and explicitly labeled test records.')
    starts = sorted({row['start'] for row in records})
    paired_rows = []
    for start in starts:
        methods = {method: sorted((row for row in records if row['start'] == start
                                   and row['method'] == method), key=lambda row: row['horizon'])
                   for method in ('unprojected', 'projected')}
        horizons = [row['horizon'] for row in methods['unprojected']]
        if (not horizons or len(set(horizons)) != len(horizons)
                or horizons != [row['horizon'] for row in methods['projected']]):
            raise ValueError('Both methods must report the same unique horizons at each start.')
        for raw, projected in zip(methods['unprojected'], methods['projected']):
            if (not np.isclose(raw['real_ability_gap'], projected['real_ability_gap'], atol=1e-12, rtol=0)
                    or any(raw[f'count_group{group}'] != projected[f'count_group{group}'] for group in (0, 1))):
                raise ValueError('The two methods must be compared against the same test cohort.')
        paired_rows.append((start, methods))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1 + len(starts), 2, figsize=(13, 3.8 * (1 + len(starts))), squeeze=False)
    for name in history[0]['losses']:
        axes[0, 0].plot([entry['epoch'] for entry in history],
                        [entry['losses'][name] for entry in history], label=name)
    axes[0, 0].set(title='Training losses', xlabel='Epoch', ylabel='Loss')
    axes[0, 1].axis('off')
    axes[0, 1].text(.02, .95,
                     'Evaluation: held-out test trajectories only\n'
                     'Contrasts: S=1 minus S=0\n'
                     'Real states are identical for both prediction methods.\n\n'
                     'A small predicted gap does not imply a small real gap.\n'
                     'Projection modifies predictions, not the real dynamics.\n'
                     'No population fairness guarantee is inferred.',
                     transform=axes[0, 1].transAxes, va='top', fontsize=11)
    for row_index, (start, methods) in enumerate(paired_rows, start=1):
        gap_axis, error_axis = axes[row_index]
        for method, color in (('unprojected', 'tab:blue'), ('projected', 'tab:orange')):
            rows = methods[method]
            horizons = [row['horizon'] for row in rows]
            gap_axis.plot(horizons, [row['predicted_ability_gap'] for row in rows],
                          color=color, label=f'predicted: {method}')
            for group, style in ((0, '-'), (1, '--')):
                error_axis.plot(horizons, [row[f'ability_mse_group{group}'] for row in rows],
                                color=color, linestyle=style, label=f'{method}, S={group}')
        rows = methods['unprojected']
        gap_axis.plot([row['horizon'] for row in rows], [row['real_ability_gap'] for row in rows],
                      'k:', linewidth=2, label='real test states')
        gap_axis.set(title=f'Test ability contrasts from t={start}', xlabel='Forecast horizon', ylabel='Contrast')
        error_axis.set(title=f'Test ability errors from t={start}', xlabel='Forecast horizon', ylabel='MSE by group')
    for axis in axes.flat:
        if axis.axison:
            axis.grid(True)
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)