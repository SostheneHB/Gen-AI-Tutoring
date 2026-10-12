"""Neural ablation grid for the Deep Koopman Autoencoder (DKA) latent
predictor described in the manuscript's Section 6.5/6.6 and left as a
pending "planned but not executed" study in earlier drafts because PyTorch
was believed unavailable. PyTorch installs cleanly from plain PyPI in this
environment (huggingface.co and download.pytorch.org are blocked, PyPI is
not), so this script actually runs the grid.

Grid (5 x 2 x 4 x 2 = 80 configurations, matching the paper's own
description in its Limitations paragraph):
  - 5 seeds: 42, 43, 44, 45, 46
  - 2 learning-rate schedules: constant 1e-3; step decay (halve every 15 epochs)
  - 4 latent weights: 0.25, 0.5, 1.0, 2.0
  - 2 orthogonality weights (lambda2): 0.05, 0.10

Scale note (disclosed honestly in the paper text): to keep the full 80-run
grid tractable on CPU in this session, each configuration trains on 2,000
trajectories (500 validation, 500 test) for 40 epochs, rather than the
8,000/2,000/2,000-trajectory, 80-epoch protocol used for the single
main-text run reported in Section 6.5. This is a smaller-scale ablation,
not a recomputation of that single run.
"""
import itertools
import json
import time
import sys
import numpy as np
import torch
from torch.nn import functional

sys.path.insert(0, '.')
from dka_notebook_v3_fixed import (PedagogicalDataset, DeepKoopmanAutoencoder,
                                    BatchIterator, evaluate)

SEEDS = [42, 43, 44, 45, 46]
LR_SCHEDULES = ['constant', 'decay']
LATENT_WEIGHTS = [0.25, 0.5, 1.0, 2.0]
LAMBDA2S = [0.05, 0.10]

TRAIN_N, VAL_N, TEST_N = 2000, 500, 500
EPOCHS = 40
RANK, DEPTH, TOLERANCE, LAMBDA3 = 4, 3, 1e-8, 0.5


def train_one(seed, lr_schedule, latent_weight, lambda2):
    torch.manual_seed(seed)
    train = PedagogicalDataset(TRAIN_N, seed=seed + 1)
    val = PedagogicalDataset(VAL_N, seed=seed + 2)
    test = PedagogicalDataset(TEST_N, seed=seed + 3)
    model = DeepKoopmanAutoencoder().double()
    loader = BatchIterator(train.states, train.sensitive, seed=seed)
    geometry_loader = BatchIterator(train.states, train.sensitive, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
                 if lr_schedule == 'decay' else None)

    model.update_bias_subspace(geometry_loader, RANK, DEPTH, TOLERANCE)
    loss_trace = []
    diverged = False
    nonfinite = False
    for epoch in range(1, EPOCHS + 1):
        model.train()
        totals, count = {}, 0
        for states, _ in loader:
            current = states[:, :-1].reshape(-1, model.state_dim)
            following = states[:, 1:].reshape(-1, model.state_dim)
            optimizer.zero_grad()
            loss, parts = model.compute_loss(current, following, lambda2=lambda2,
                                              lambda3=LAMBDA3, latent_weight=latent_weight)
            if not torch.isfinite(loss):
                nonfinite = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            count += len(current)
            totals['total'] = totals.get('total', 0.0) + len(current) * loss.item()
        if nonfinite:
            break
        if scheduler is not None:
            scheduler.step()
        if epoch % 5 == 0 or epoch == EPOCHS:
            model.update_bias_subspace(geometry_loader, RANK, DEPTH, TOLERANCE)
        epoch_loss = totals['total'] / count
        loss_trace.append(epoch_loss)

    if nonfinite:
        return dict(seed=seed, lr_schedule=lr_schedule, latent_weight=latent_weight,
                    lambda2=lambda2, status='nonfinite_loss', loss_trace=loss_trace)

    first_loss = loss_trace[0] if loss_trace else float('nan')
    last_loss = loss_trace[-1] if loss_trace else float('nan')
    best_loss = min(loss_trace) if loss_trace else float('nan')
    diverged = bool(last_loss > 2.0 * best_loss) if loss_trace else True

    records = evaluate(model, test, split='test')
    picked = [r for r in records if r['start'] == 0 and r['horizon'] == 5]
    summary = {}
    for method in ('unprojected', 'projected'):
        row = next(r for r in picked if r['method'] == method)
        summary[method] = dict(
            predicted_ability_gap=row['predicted_ability_gap'],
            real_ability_gap=row['real_ability_gap'],
            ability_gap_error=row['ability_gap_error'],
            ability_mse_group0=row['ability_mse_group0'],
            ability_mse_group1=row['ability_mse_group1'],
        )

    return dict(seed=seed, lr_schedule=lr_schedule, latent_weight=latent_weight,
                lambda2=lambda2, status='ok', first_loss=first_loss, last_loss=last_loss,
                best_loss=best_loss, diverged=diverged, test_start0_horizon5=summary)


def main():
    configs = list(itertools.product(SEEDS, LR_SCHEDULES, LATENT_WEIGHTS, LAMBDA2S))
    assert len(configs) == 80, len(configs)
    results = []
    t0 = time.time()
    for i, (seed, lr_schedule, latent_weight, lambda2) in enumerate(configs, 1):
        r = train_one(seed, lr_schedule, latent_weight, lambda2)
        results.append(r)
        if i % 10 == 0 or i == 1:
            print(f'[{i}/80] elapsed={time.time()-t0:.1f}s status={r["status"]} '
                  f'seed={seed} lr={lr_schedule} lw={latent_weight} l2={lambda2}',
                  flush=True)
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2, allow_nan=True)
    print(f'Done in {time.time()-t0:.1f}s. Wrote ablation_results.json ({len(results)} runs).')


if __name__ == '__main__':
    main()
