"""Check cached-teacher/student alignment before the first optimizer step."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from scripts.train.vq_lewm_joint_distillation import (
    CachedSplit,
    image_preprocessor,
    initialize_student_from_state,
    load_cache_metadata,
    move_batch,
)
from stable_worldmodel.data import column_normalizer
from stable_worldmodel.wm.vq_lewm.distillation import (
    nearest_code_indices,
    sparse_topk_kl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    os.environ['LOCAL_DATASET_DIR'] = str(cfg.paths.dataset_cache)
    os.environ['STABLEWM_HOME'] = str(cfg.paths.dataset_cache)
    metadata, train_indices, _ = load_cache_metadata(cfg)
    base = swm.data.load_dataset(
        cfg.data.dataset,
        transform=None,
        num_steps=int(cfg.data.num_steps),
        frameskip=int(cfg.data.frameskip),
        keys_to_load=['pixels', 'action'],
    )
    action_dim = int(cfg.data.frameskip) * int(base.get_dim('action'))
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = action_dim
    base.transform = spt.data.transforms.Compose(
        image_preprocessor(int(cfg.data.img_size)),
        column_normalizer(base, 'action', 'action'),
    )
    split = CachedSplit(
        base,
        train_indices[: args.samples],
        Path(cfg.paths.cache_dir),
        'train',
        int(cfg.codebook.num_embeddings),
    )
    batch = next(iter(DataLoader(split, batch_size=args.samples)))
    device = torch.device('cuda')
    model = hydra.utils.instantiate(cfg.model)
    initialize_student_from_state(model, cfg.paths.student_init_checkpoint)
    model.to(device).eval()
    batch = move_batch(batch, device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        student = model.encode_student(batch['pixels'])
    nearest = nearest_code_indices(
        student,
        model.codebook,
        k=5,
        codebook_chunk_size=int(cfg.codebook.distance_chunk_size),
    )
    hard = batch['hard_tokens']
    report = {
        'cache_metadata_hash': metadata['metadata_sha256'],
        'latent_mse_before_step': float(
            F.mse_loss(student.float(), batch['teacher_latent'].float())
        ),
        'soft_kl_before_step': float(
            sparse_topk_kl(
                student,
                model.codebook,
                batch['topk_indices'],
                batch['topk_probs'],
                temperature=float(cfg.codebook.temperature),
                codebook_chunk_size=int(cfg.codebook.distance_chunk_size),
            )
        ),
        'token_agreement_before_step': float(
            (nearest[..., 0] == hard).float().mean()
        ),
        'top5_agreement_before_step': float(
            (nearest == hard[..., None]).any(-1).float().mean()
        ),
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
