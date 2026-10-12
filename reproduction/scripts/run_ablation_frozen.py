"""Frozen-diagnostic re-evaluation of the full 80-configuration DKA ablation
(reviewer comment 8).

Training is identical to run_ablation.py (same grid, data seeds, optimiser,
schedule, geometry refresh every 5 epochs).  In addition, the model is saved
at epoch 0 (after the first geometry fit) and at epochs 5, 10, ..., 40, and
every checkpoint of every configuration is evaluated under one diagnostic
fixed before any result was inspected:

  common validation set  PedagogicalDataset(1000, seed=9001), shared by all
                         80 configurations and never used for training
  D1  one-step physical-state MSE, unprojected latent step
  D5  five-step open-loop physical-state MSE from x_0, unprojected
  P1, P5 the same with the projected latent step

These quantities use only the checkpoint's own encoder, latent map and
decoder and the fixed validation states; they contain no moving latent
target, so values at different epochs and in different configurations are
comparable.  Reported per configuration: first, last and best value, whether
last < first, whether the sequence is non-increasing (tolerance 1e-3
relative), and the epoch of the minimum.

    python run_ablation_frozen.py --workers 8   (one thread per worker: deterministic)
"""
import argparse
import copy
import itertools
import json
import time
from multiprocessing import Pool

import torch

from dka_notebook_v3_fixed import PedagogicalDataset, DeepKoopmanAutoencoder, BatchIterator, evaluate
from run_ablation import (SEEDS, LR_SCHEDULES, LATENT_WEIGHTS, LAMBDA2S, TRAIN_N, TEST_N,
                          EPOCHS, RANK, DEPTH, TOLERANCE, LAMBDA3)

VAL_SEED, VAL_N = 9001, 1000
CHECKPOINTS = [0] + list(range(5, EPOCHS + 1, 5))
REL_TOL = 1e-3


@torch.no_grad()
def diagnostic(model, states):
    model.eval()
    out = {}
    for fair, tag in ((False, ""), (True, "P")):
        cur = states[:, :-1].reshape(-1, model.state_dim)
        nxt = states[:, 1:].reshape(-1, model.state_dim)
        one = model.decode(model.latent_step(model.encode(cur), fair=fair))
        z = model.encode(states[:, 0])
        for _ in range(5):
            z = model.latent_step(z, fair=fair)
        five = model.decode(z)
        out[f"{tag or 'D'}1"] = torch.mean((one - nxt) ** 2).item()
        out[f"{tag or 'D'}5"] = torch.mean((five - states[:, 5]) ** 2).item()
    return out


def train_one(cfg):
    torch.set_num_threads(1)          # run_ablation.py is not reproducible across thread counts
    seed, lr_schedule, latent_weight, lambda2 = cfg
    torch.manual_seed(seed)
    train = PedagogicalDataset(TRAIN_N, seed=seed + 1)
    common = PedagogicalDataset(VAL_N, seed=VAL_SEED).states
    model = DeepKoopmanAutoencoder().double()
    loader = BatchIterator(train.states, train.sensitive, seed=seed)
    geometry_loader = BatchIterator(train.states, train.sensitive, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
                 if lr_schedule == 'decay' else None)
    model.update_bias_subspace(geometry_loader, RANK, DEPTH, TOLERANCE)
    trace = {0: diagnostic(model, common)}
    loss_trace = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, count = 0.0, 0
        for states, _ in loader:
            current = states[:, :-1].reshape(-1, model.state_dim)
            following = states[:, 1:].reshape(-1, model.state_dim)
            optimizer.zero_grad()
            loss, _ = model.compute_loss(current, following, lambda2=lambda2,
                                         lambda3=LAMBDA3, latent_weight=latent_weight)
            if not torch.isfinite(loss):
                return dict(config=cfg, status='nonfinite_loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            total += len(current) * loss.item()
            count += len(current)
        if scheduler is not None:
            scheduler.step()
        if epoch % 5 == 0 or epoch == EPOCHS:
            model.update_bias_subspace(geometry_loader, RANK, DEPTH, TOLERANCE)
            trace[epoch] = diagnostic(copy.deepcopy(model), common)
        loss_trace.append(total / count)
    summary = {}
    for key in trace[0]:
        v = [trace[e][key] for e in CHECKPOINTS]
        summary[key] = dict(first=v[0], last=v[-1], best=min(v),
                            argmin_epoch=CHECKPOINTS[v.index(min(v))],
                            last_below_first=v[-1] < v[0],
                            monotone=all(b <= a * (1 + REL_TOL) for a, b in zip(v, v[1:])),
                            last_within_5pct_of_best=v[-1] <= 1.05 * min(v),
                            trace=v)
    test = PedagogicalDataset(TEST_N, seed=seed + 3)
    picked = [r for r in evaluate(model, test, split='test') if r['start'] == 0 and r['horizon'] == 5]
    test_summary = {r['method']: {k: r[k] for k in ('predicted_ability_gap', 'real_ability_gap',
                                                     'ability_gap_error', 'ability_mse_group0',
                                                     'ability_mse_group1')} for r in picked}
    return dict(seed=seed, lr_schedule=lr_schedule, latent_weight=latent_weight, lambda2=lambda2,
                status='ok', test_start0_horizon5=test_summary, first_loss=loss_trace[0], last_loss=loss_trace[-1],
                best_loss=min(loss_trace), frozen=summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='ablation_frozen_results.json')
    args = ap.parse_args()
    configs = list(itertools.product(SEEDS, LR_SCHEDULES, LATENT_WEIGHTS, LAMBDA2S))
    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(train_one, configs)
    ok = [r for r in results if r['status'] == 'ok']
    agg = {}
    for key in ('D1', 'D5', 'P1', 'P5'):
        agg[key] = {c: int(sum(r['frozen'][key][c] for r in ok))
                    for c in ('last_below_first', 'monotone', 'last_within_5pct_of_best')}
        agg[key]['argmin_at_last_checkpoint'] = int(sum(r['frozen'][key]['argmin_epoch'] == EPOCHS for r in ok))
    agg['n_ok'] = len(ok)
    agg['raw_loss_last_above_best'] = int(sum(r['last_loss'] > r['best_loss'] for r in ok))
    agg['raw_loss_last_above_first'] = int(sum(r['last_loss'] > r['first_loss'] for r in ok))
    t = [r['test_start0_horizon5'] for r in ok]
    agg['projection_shrinks_predicted_gap'] = int(sum(
        abs(x['projected']['predicted_ability_gap']) < abs(x['unprojected']['predicted_ability_gap']) for x in t))
    agg['projection_raises_group1_error'] = int(sum(
        x['projected']['ability_mse_group1'] > x['unprojected']['ability_mse_group1'] for x in t))
    with open(args.out, 'w') as f:
        json.dump({'protocol': __doc__, 'aggregate': agg, 'runs': results}, f, indent=1)
    print(json.dumps(agg, indent=1))
    print(f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
