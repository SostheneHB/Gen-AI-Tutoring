"""Projected latent prediction with held-out, simulator-matched diagnostics.

Run this file explicitly to train; importing it never starts an experiment.
Historical dashboards and result files are not overwritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as functional
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    raise SystemExit('PyTorch is required for DKA training; no training was run.') from error

from dka_geometry import fit_bias_geometry
from dka_evaluation import prediction_metrics, plot_results


class PedagogicalDataset:
    """Independent trajectories; gamma affects ability, theta_bias helpfulness."""

    def __init__(self, num_samples=8000, seq_len=10, seed=42, noise_std=0.05,
                 alpha=0.9, beta=0.5, gamma=0.3, delta=0.8, eta=0.7,
                 zeta=0.4, theta_bias=0.2):
        if num_samples < 2 or seq_len < 2 or noise_std < 0:
            raise ValueError('Need at least two samples and states, with nonnegative noise.')
        self.num_samples, self.seq_len = num_samples, seq_len
        self.seed = seed
        self.gamma = gamma
        generator = torch.Generator().manual_seed(seed)
        ability = torch.randn(num_samples, 1, generator=generator, dtype=torch.float64) * 0.5
        helpfulness = torch.rand(num_samples, 1, generator=generator, dtype=torch.float64) * 0.5
        performance = torch.randint(0, 2, (num_samples, 1), generator=generator).double()
        self.sensitive = torch.randint(0, 2, (num_samples, 1), generator=generator).double()
        if self.sensitive.unique().numel() != 2:
            raise ValueError('This cohort contains only one group; increase its size.')
        self.states = torch.empty(num_samples, seq_len, 3, dtype=torch.float64)
        for step in range(seq_len):
            self.states[:, step] = torch.cat((ability, helpfulness, performance), dim=1)
            if step == seq_len - 1:
                break
            ability_next = (alpha * ability + beta * torch.sigmoid(helpfulness)
                            + gamma * self.sensitive + noise_std * torch.randn(
                                num_samples, 1, generator=generator, dtype=torch.float64))
            performance_next = torch.bernoulli(
                torch.sigmoid(ability_next - delta), generator=generator)
            helpfulness_next = (eta * helpfulness + zeta * torch.tanh(performance)
                               + theta_bias * self.sensitive + noise_std * torch.randn(
                                   num_samples, 1, generator=generator, dtype=torch.float64))
            ability = ability_next
            helpfulness = helpfulness_next.clamp(0, 1)
            performance = performance_next

    def __len__(self):
        return self.num_samples


class BatchIterator:
    """Tensor-only batching with a private shuffle generator."""

    def __init__(self, states, sensitive, batch_size=128, shuffle=True, seed=42):
        if batch_size < 1 or len(states) == 0 or len(states) != len(sensitive):
            raise ValueError('Invalid batch size or unaligned data.')
        self.states, self.sensitive = states, sensitive
        self.batch_size, self.shuffle = batch_size, shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        indices = (torch.randperm(len(self.states), generator=self.generator)
                   if self.shuffle else torch.arange(len(self.states)))
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            yield self.states[batch], self.sensitive[batch]

    def __len__(self):
        return (len(self.states) + self.batch_size - 1) // self.batch_size


class DeepKoopmanAutoencoder(nn.Module):
    """Real, affine latent predictor; projection is not a fairness certificate."""

    def __init__(self, state_dim=3, latent_dim=8, hidden_dim=32):
        super().__init__()
        self.state_dim, self.latent_dim = state_dim, latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))
        self.K = nn.Parameter(torch.eye(latent_dim))
        self.offset = nn.Parameter(torch.zeros(latent_dim))
        self.register_buffer('feature_mean', torch.zeros(latent_dim))
        self.register_buffer('G_reg', torch.eye(latent_dim))
        self.register_buffer('Q', torch.zeros(latent_dim, latent_dim))
        self.register_buffer('projector', torch.eye(latent_dim))
        self.geometry_diagnostics = {}

    def encode(self, states):
        return self.encoder(states)

    def decode(self, features):
        return self.decoder(features)

    def project_out_bias(self, features):
        return self.feature_mean + (features - self.feature_mean) @ self.projector

    def latent_step(self, features, fair=True):
        """Use one affine map everywhere, projecting its offset as well."""
        if fair:
            features = self.project_out_bias(features)
        prediction = features @ self.K.T + self.offset
        return self.project_out_bias(prediction) if fair else prediction

    def forward(self, states):
        features = self.encode(states)
        prediction = self.latent_step(features)
        return self.decode(prediction), features, prediction

    def forward_biased(self, states):
        features = self.encode(states)
        prediction = self.latent_step(features, fair=False)
        return self.decode(prediction), features, prediction

    def compute_loss(self, current, following, lambda1=0.5, lambda2=0.1,
                     lambda3=0.5, latent_weight=1.0):
        prediction, features, _ = self(current)
        predicted_latent = self.latent_step(features, fair=False)
        target_latent = self.encode(following).detach()
        losses = {
            'pred': functional.mse_loss(prediction, following),
            'recon': functional.mse_loss(self.decode(features), current),
            'latent': functional.mse_loss(predicted_latent, target_latent),
            'orth': torch.square(self.K.T @ self.K - torch.eye(
                self.latent_dim, dtype=self.K.dtype, device=self.K.device)).sum() / self.latent_dim,
            'fair': torch.square(self.Q.T @ self.G_reg @ self.K.T @ self.Q).sum(),
        }
        total = (losses['pred'] + lambda1 * losses['recon']
                 + latent_weight * losses['latent'] + lambda2 * losses['orth']
                 + lambda3 * losses['fair'])
        return total, {name: value.item() for name, value in losses.items()}

    @torch.no_grad()
    def update_bias_subspace(self, data_loader, rank=4, depth=3, tolerance=1e-8):
        features, labels = [], []
        for states, sensitive in data_loader:
            current = states[:, :-1].reshape(-1, self.state_dim).to(self.K.device)
            features.extend(self.encode(current).cpu().tolist())
            labels.extend(sensitive[:, None, :].expand(
                -1, states.shape[1] - 1, -1).reshape(-1).tolist())
        mean, metric, basis, projector, diagnostics = fit_bias_geometry(
            features, labels, self.K.T.detach().cpu().tolist(),
            rank=rank, depth=depth, tolerance=tolerance)
        for name, values in [('feature_mean', mean), ('G_reg', metric),
                             ('projector', projector)]:
            getattr(self, name).copy_(torch.tensor(
                values.tolist(), dtype=self.K.dtype, device=self.K.device))
        self.Q.zero_()
        self.Q[:, :basis.shape[1]].copy_(torch.tensor(
            basis.tolist(), dtype=self.K.dtype, device=self.K.device))
        centered_offset = self.feature_mean @ self.K.T + self.offset - self.feature_mean
        diagnostics['removed_centered_offset_norm'] = torch.linalg.vector_norm(
            centered_offset - centered_offset @ self.projector).item()
        self.geometry_diagnostics = diagnostics
        return diagnostics


@torch.no_grad()
def evaluate_koopman_rollout(model, initial, horizon, fair=True):
    """Open-loop latent rollout; index zero is the observed physical state."""
    model.eval()
    initial = initial.to(model.K.device)
    features = model.encode(initial)
    trajectory = [initial]
    for _ in range(horizon):
        features = model.latent_step(features, fair=fair)
        trajectory.append(model.decode(features))
    return torch.stack(trajectory, dim=1)


@torch.no_grad()
def evaluate(model, dataset, *, split):
    """Compare predictions with held-out real states, not just predicted gaps.

    At start=0, independence of x_0 and S makes population prediction gaps
    vanish for any fixed predictor that does not receive S. Real gaps and
    per-group errors are therefore always reported alongside predictions.
    """
    if split not in ('validation', 'test'):
        raise ValueError('Evaluation must identify the validation or test split.')
    model.eval()
    records = []
    starts = sorted({0, (dataset.seq_len - 1) // 2})
    for start in starts:
        available = dataset.seq_len - 1 - start
        real = dataset.states[:, start:].to(model.K.device)
        sensitive = dataset.sensitive.reshape(-1).to(model.K.device)
        labels = dataset.sensitive.reshape(-1).cpu().tolist()
        for fair in (False, True):
            predicted = evaluate_koopman_rollout(model, real[:, 0], available, fair=fair)
            latent = model.encode(real[:, 0])
            for horizon in range(1, available + 1):
                latent = model.latent_step(latent, fair=fair)
                target_latent = model.encode(real[:, horizon])
                latent_errors = torch.square(latent - target_latent)
                metrics = prediction_metrics(predicted[:, horizon].cpu().tolist(),
                                             real[:, horizon].cpu().tolist(), labels)
                records.append({
                    'split': split, 'dataset_seed': dataset.seed,
                    'start': start, 'horizon': horizon, 'target_time': start + horizon,
                    'method': 'projected' if fair else 'unprojected',
                    **metrics,
                    'latent_mse': latent_errors.mean().item(),
                    'latent_mse_by_coordinate': latent_errors.mean(dim=0).cpu().tolist(),
                    'latent_mean_error': (latent - target_latent).mean(dim=0).cpu().tolist(),
                    **{f'latent_mse_group{group}': latent_errors[sensitive == group].mean().item()
                       for group in (0, 1)},
                })
    return records


def train_dka(model, train_dataset, validation_dataset, epochs=80, rank=4,
              depth=3, lambda3=0.5, latent_weight=1.0, seed=42, tolerance=1e-8):
    if train_dataset is validation_dataset or train_dataset.seed == validation_dataset.seed:
        raise ValueError('Training and validation require different datasets and random seeds.')
    loader = BatchIterator(train_dataset.states, train_dataset.sensitive, seed=seed)
    geometry_loader = BatchIterator(train_dataset.states, train_dataset.sensitive, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, geometry_history = [], []
    geometry_history.append({'epoch': 0, **model.update_bias_subspace(
        geometry_loader, rank, depth, tolerance)})
    for epoch in range(1, epochs + 1):
        model.train()
        totals, count = {}, 0
        for states, _ in loader:
            states = states.to(model.K.device)
            current = states[:, :-1].reshape(-1, model.state_dim)
            following = states[:, 1:].reshape(-1, model.state_dim)
            optimizer.zero_grad()
            loss, parts = model.compute_loss(current, following, lambda3=lambda3,
                                             latent_weight=latent_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            parts['total'] = loss.item()
            count += len(current)
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + len(current) * value
        if epoch % 5 == 0 or epoch == epochs:
            geometry_history.append({'epoch': epoch, **model.update_bias_subspace(
                geometry_loader, rank, depth, tolerance)})
        history.append({'epoch': epoch, 'losses': {name: value / count for name, value in totals.items()},
                        'validation': evaluate(model, validation_dataset, split='validation')})
        if epoch % 10 == 0 or epoch == epochs:
            print(f'Epoch {epoch}/{epochs}: loss={totals["total"] / count:.6f}')
    return history, geometry_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=int(os.getenv('DKA_SEED', '42')))
    parser.add_argument('--epochs', type=int, default=int(os.getenv('DKA_EPOCHS', '80')))
    parser.add_argument('--rank', type=int, default=int(os.getenv('DKA_RANK', '4')))
    parser.add_argument('--depth', type=int, default=int(os.getenv('DKA_KRYLOV_DEPTH', '3')))
    parser.add_argument('--lambda3', type=float, default=float(os.getenv('DKA_LAMBDA3', '0.5')))
    parser.add_argument('--latent-weight', type=float, default=1.0)
    parser.add_argument('--svd-tolerance', type=float, default=1e-8)
    parser.add_argument('--train-samples', type=int, default=8000)
    parser.add_argument('--validation-samples', type=int, default=2000)
    parser.add_argument('--test-samples', type=int, default=2000)
    parser.add_argument('--output', type=Path, default=Path('dka-corrected-results'))
    args = parser.parse_args()
    if (args.seed < 0 or args.seed > 2**63 - 4 or args.epochs < 1
            or not 1 <= args.rank <= 8 or args.depth < 0
            or not np.isfinite([args.lambda3, args.latent_weight, args.svd_tolerance]).all()
            or min(args.lambda3, args.latent_weight) < 0 or not 0 < args.svd_tolerance < 1):
        parser.error('Invalid seed, epochs, rank, depth, weight, or tolerance.')
    if args.output.is_absolute() or '..' in args.output.parts:
        parser.error('Use a workspace-relative output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Output directory must be empty; choose a new directory for each run.')
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = PedagogicalDataset(args.train_samples, seed=args.seed + 1)
    validation = PedagogicalDataset(args.validation_samples, seed=args.seed + 2)
    test = PedagogicalDataset(args.test_samples, seed=args.seed + 3)
    model = DeepKoopmanAutoencoder().double().to(device)
    history, geometry = train_dka(
        model, train, validation, epochs=args.epochs, rank=args.rank,
        depth=args.depth, lambda3=args.lambda3, latent_weight=args.latent_weight,
        seed=args.seed, tolerance=args.svd_tolerance)
    records = evaluate(model, test, split='test')
    result = {
        'configuration': {**vars(args), 'output': str(args.output)},
        'loss_configuration': {
            'weights': {'prediction': 1.0, 'reconstruction': 0.5, 'orthogonality': 0.1,
                        'fairness': args.lambda3, 'latent': args.latent_weight},
            'state_dimension': model.state_dim, 'latent_dimension': model.latent_dim,
            'state_mse_reduction': 'mean over transitions and physical coordinates',
            'latent_mse_reduction': 'mean over transitions and latent coordinates',
            'orthogonality_reduction': 'squared Frobenius norm divided by latent dimension',
            'fairness_reduction': 'squared weighted bias-block Frobenius norm; no rank division',
            'latent_target': 'detached encoder of the observed noisy successor',
            'latent_prediction': 'unprojected affine map; not a samplewise Koopman identity',
        },
        'split_seeds': {'train': args.seed + 1, 'validation': args.seed + 2, 'test': args.seed + 3},
        'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device)},
        'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ('dka_notebook_v3_fixed.py', 'dka_geometry.py', 'dka_evaluation.py')},
        'evaluation_protocol': {
            'split': 'test', 'starting_times': sorted({row['start'] for row in records}),
            'contrast_sign': 'group 1 minus group 0',
            'error_sign': 'prediction minus real state',
            'real_trajectories_shared_between_methods': True,
            'latent_target': 'encoder of the observed noisy state at the target time',
            'latent_error_scope': 'open-loop prediction error by start, horizon and group; projected error includes the projection change',
            'population_first_step_ability_gap': test.gamma,
            'population_gap_scope': 'Default corrected simulator, independent initial states and zero-mean independent noise; not a measured test result.',
        },
        'interpretation': 'Predictive diagnostics, not intervention effects or population fairness guarantees.',
        'history': history, 'geometry_history': geometry, 'test': records,
    }
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    torch.save(model.state_dict(), args.output / 'model.pt')
    plot_results(history, records, args.output / 'dashboard.png')
    print(f'Results saved to {args.output}; historical figures were not replaced.')


if __name__ == '__main__':
    main()
