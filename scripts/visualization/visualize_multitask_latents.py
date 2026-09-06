"""Cross-task joint latent clustering for multitask world models.

Collects env state-grid latents for several tasks with ONE shared model
checkpoint (the per-task deployment exports of a multitask model share the
same encoder weights; `encode` is task-agnostic), pools them, and produces:

- joint t-SNE / PCA scatter colored by task (with per-task centroids),
- the same t-SNE layout colored by within-task normalized grid position,
- quantitative clustering metrics (silhouette, kNN task purity, centroid
  distances, cross-task mixing) saved as JSON.

Run from any directory; outputs are written to the current working dir:

    python visualize_multitask_latents.py \
        --checkpoint <task_export_dir> \
        --tasks pusht tworoom cube \
        --output-name M2_cross_task \
        --grid-size 20 --cube-grid-size 14 [--device cpu]

Requires MUJOCO_GL=egl (headless) when the cube task is included.
"""

import os

# Respect a caller-provided backend (e.g. MUJOCO_GL=egl on headless nodes);
# fall back to glfw as in visualize_env.py.
os.environ.setdefault('MUJOCO_GL', 'glfw')

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger as logging
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from stable_worldmodel import data as swm_data

from utils import get_state_grid
from visualize_env import get_env, get_world_model, prepare_info


TASK_ENVS = {
    'pusht': {
        'env_name': 'swm/PushT-v1',
        'grid_size': 20,
        'variation': ['agent.start_position'],
    },
    'tworoom': {
        'env_name': 'swm/TwoRoom-v1',
        'grid_size': 20,
        'variation': ['agent.position'],
    },
    'cube': {
        'env_name': 'swm/OGBCube-v0',
        'grid_size': 14,
        'variation': ['cube.start_position'],
    },
}

TASK_COLORS = {
    'pusht': 'tab:blue',
    'tworoom': 'tab:orange',
    'cube': 'tab:green',
}


def collect_task_latents(task: str, checkpoint: str, grid_size: int, device: str):
    """Encode one task's state grid with the shared model; return (N, D)."""
    spec = TASK_ENVS[task]
    cfg = OmegaConf.create(
        {
            'device': device,
            'seed': 42,
            'image_size': 224,
            'patch_size': 14,
            'dimensionality_reduction': 'tsne',
            'cache_dir': str(swm_data.get_cache_dir()),
            'env': {
                'env_name': spec['env_name'],
                'history_size': 1,
                'frame_skip': 1,
                'grid_size': grid_size,
                'default_variation': list(spec['variation']),
            },
            'world_model': {
                'checkpoint_path': checkpoint,
                'encoding': {},
            },
        }
    )
    env, process, transform = get_env(cfg)
    world_model = get_world_model(cfg)
    grid, state_grid = get_state_grid(
        env.unwrapped.envs[0].unwrapped, grid_size
    )

    embeddings = []
    for state in tqdm(state_grid, desc=f'encoding {task}'):
        # variation=[] suppresses random appearance resampling so every grid
        # state keeps the default appearance (cleaner cross-task comparison).
        _, infos = env.reset(options={'state': state, 'variation': []})
        infos = prepare_info(infos, process, transform)
        for key in infos:
            if isinstance(infos[key], torch.Tensor):
                infos[key] = infos[key].to(device)
        infos = world_model.encode(infos)
        embeddings.append(infos['embed'].detach().cpu().reshape(-1).numpy())
    env.close()
    return np.stack(embeddings, axis=0), grid


def knn_task_purity(latents: np.ndarray, labels: np.ndarray, k: int = 10):
    """Leave-one-out fraction of k nearest neighbors sharing the task label."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(latents)
    _, idx = nn.kneighbors(latents)
    neigh_labels = labels[idx[:, 1:]]  # drop self
    return float((neigh_labels == labels[:, None]).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='task export dir (weights.pt + config.json) of the shared model',
    )
    parser.add_argument('--tasks', nargs='+', default=['pusht', 'tworoom', 'cube'])
    parser.add_argument('--grid-size', type=int, default=20)
    parser.add_argument('--cube-grid-size', type=int, default=14)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-name', default='cross_task')
    args = parser.parse_args()

    all_latents, all_labels, all_gridnorm = [], [], []
    for task in args.tasks:
        grid_size = (
            args.cube_grid_size if task == 'cube' else args.grid_size
        )
        latents, grid = collect_task_latents(
            task, args.checkpoint, grid_size, args.device
        )
        grid_norm = (grid - grid.min(axis=0)) / (
            grid.max(axis=0) - grid.min(axis=0) + 1e-6
        )
        all_latents.append(latents)
        all_labels.extend([task] * latents.shape[0])
        all_gridnorm.append(grid_norm)
        logging.info(
            f'{task}: {latents.shape[0]} states, dim={latents.shape[1]}'
        )

    X = np.concatenate(all_latents, axis=0)
    labels = np.array(all_labels)
    pos = np.concatenate(all_gridnorm, axis=0)
    tasks = list(dict.fromkeys(all_labels))

    # ---- joint projections ----
    logging.info(f'joint t-SNE on {X.shape[0]} points...')
    X_tsne = TSNE(
        n_components=2, random_state=args.seed, perplexity=30
    ).fit_transform(X)
    X_pca = PCA(n_components=2, random_state=args.seed).fit_transform(X)

    # ---- clustering metrics (raw latent space is the meaningful one) ----
    centroid = {t: X[labels == t].mean(axis=0) for t in tasks}
    metrics = {
        'checkpoint': args.checkpoint,
        'tasks': tasks,
        'n_points': int(X.shape[0]),
        'latent_dim': int(X.shape[1]),
        'silhouette_raw': float(silhouette_score(X, labels)),
        'silhouette_pca2': float(silhouette_score(X_pca, labels)),
        'silhouette_tsne2': float(silhouette_score(X_tsne, labels)),
        'knn_task_purity_k10_raw': knn_task_purity(X, labels, k=10),
        'per_task_norm_mean': {
            t: float(np.linalg.norm(X[labels == t], axis=1).mean())
            for t in tasks
        },
        'centroid_distance': {
            f'{a}|{b}': float(np.linalg.norm(centroid[a] - centroid[b]))
            for i, a in enumerate(tasks)
            for b in tasks[i + 1 :]
        },
        # fraction of points whose 10-NN are ALL from a different task
        'cross_task_mixing_k10': {},
    }
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=11).fit(X)
    _, idx = nn.kneighbors(X)
    for t in tasks:
        mask = labels == t
        other_frac = (labels[idx[:, 1:]][mask] != t).mean(axis=1)
        metrics['cross_task_mixing_k10'][t] = {
            'mean_other_task_neighbor_fraction': float(other_frac.mean()),
            'fully_mixed_fraction': float((other_frac == 1.0).mean()),
        }

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    ax = axes[0]
    for t in tasks:
        m = labels == t
        ax.scatter(
            X_tsne[m, 0], X_tsne[m, 1],
            s=8, alpha=0.6, color=TASK_COLORS[t], label=f'{t} (n={m.sum()})',
        )
        ax.scatter(
            X_tsne[m, 0].mean(), X_tsne[m, 1].mean(),
            marker='*', s=350, color=TASK_COLORS[t],
            edgecolors='black', linewidths=1.2, zorder=5,
        )
    ax.set_title(
        f'Joint t-SNE by task  |  silhouette(raw)='
        f"{metrics['silhouette_raw']:.3f}  purity@10="
        f"{metrics['knn_task_purity_k10_raw']:.3f}"
    )
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)

    ax = axes[1]  # same layout, within-task position hue (R=x, G=y)
    colors = np.zeros((len(pos), 4))
    colors[:, 0] = pos[:, 0]
    colors[:, 1] = pos[:, 1]
    colors[:, 2] = 0.5
    colors[:, 3] = 0.85
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=8)
    for t in tasks:
        m = labels == t
        ax.text(
            X_tsne[m, 0].mean(), X_tsne[m, 1].mean(), t,
            fontsize=13, fontweight='bold',
            bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 2},
        )
    ax.set_title('Same t-SNE, colored by within-task grid position (R=x, G=y)')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.grid(True, linestyle='--', alpha=0.3)

    ax = axes[2]
    for t in tasks:
        m = labels == t
        ax.scatter(
            X_pca[m, 0], X_pca[m, 1],
            s=8, alpha=0.6, color=TASK_COLORS[t], label=t,
        )
    ax.set_title(
        f"Joint PCA by task  |  silhouette(pca2)="
        f"{metrics['silhouette_pca2']:.3f}"
    )
    ax.set_xlabel('PCA dim 1')
    ax.set_ylabel('PCA dim 2')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = f'{args.output_name}_cluster.{ext}'
        plt.savefig(out, dpi=150)
        logging.info(f'saved {out}')
    plt.close(fig)

    with open(f'{args.output_name}_cluster_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f'saved {args.output_name}_cluster_metrics.json')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
