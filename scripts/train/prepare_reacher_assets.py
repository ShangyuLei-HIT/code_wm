"""Download and validate the official Reacher LeWM model and dataset.

Mirrors scripts/train/prepare_cube_assets.py for the fourth official LeWM
task family: converts the legacy ViT state-dict layout (if present) into the
current ``vit_hf`` layout, validates the 192-dim embedding contract, and
extracts the ``reacher.h5`` dataset shipped by ``quentinll/lewm-reacher``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from stable_worldmodel.wm.utils import load_pretrained

LEGACY_VIT_REPLACEMENTS = (
    ('encoder.encoder.layer.', 'encoder.layers.'),
    ('.attention.attention.query.', '.attention.q_proj.'),
    ('.attention.attention.key.', '.attention.k_proj.'),
    ('.attention.attention.value.', '.attention.v_proj.'),
    ('.attention.output.dense.', '.attention.o_proj.'),
    ('.intermediate.dense.', '.mlp.fc1.'),
    ('.output.dense.', '.mlp.fc2.'),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def convert_legacy_vit_state_dict(state_dict):
    converted = {}
    for original, value in state_dict.items():
        key = original
        for source, target in LEGACY_VIT_REPLACEMENTS:
            key = key.replace(source, target)
        if key in converted:
            raise RuntimeError(
                f'checkpoint key collision after conversion: {key}'
            )
        converted[key] = value
    return converted


def is_legacy_vit_state_dict(state_dict) -> bool:
    return any(
        key.startswith('encoder.encoder.layer.') for key in state_dict
    )


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        default=os.environ.get('STABLEWM_HOME', './.stablewm'),
    )
    parser.add_argument(
        '--skip-dataset',
        action='store_true',
        help='Only prepare the model checkpoint.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    raw_checkpoint = root / 'checkpoints' / 'official_lewm_reacher_raw'
    checkpoint = root / 'checkpoints' / 'official_lewm_reacher_compat'
    dataset_root = root / 'datasets' / 'quentinll--lewm-reacher'
    raw_checkpoint.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    raw_weights = raw_checkpoint / 'weights.pt'
    compat_weights = checkpoint / 'weights.pt'

    required_raw = ('config.json', 'weights.pt')
    if not all((raw_checkpoint / name).exists() for name in required_raw):
        snapshot_download(
            repo_id='quentinll/lewm-reacher',
            repo_type='model',
            local_dir=raw_checkpoint,
            allow_patterns=['config.json', 'weights.pt', 'README.md'],
        )
    for filename in ('config.json', 'README.md'):
        source = raw_checkpoint / filename
        destination = checkpoint / filename
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)

    raw_state = torch.load(
        raw_weights, map_location='cpu', weights_only=True
    )
    converted_state = (
        convert_legacy_vit_state_dict(raw_state)
        if is_legacy_vit_state_dict(raw_state)
        else raw_state
    )
    write_compat = not compat_weights.exists()
    if compat_weights.exists():
        existing_state = torch.load(
            compat_weights, map_location='cpu', weights_only=True
        )
        write_compat = is_legacy_vit_state_dict(existing_state)
        if not write_compat and set(existing_state) != set(converted_state):
            raise RuntimeError(
                'existing Reacher compat checkpoint has unexpected keys'
            )
    if write_compat:
        atomic_torch_save(converted_state, compat_weights)
        print(
            f'Converted official Reacher checkpoint: {checkpoint}',
            flush=True,
        )

    conversion = {
        'source_repo': 'quentinll/lewm-reacher',
        'raw_checkpoint': str(raw_checkpoint),
        'raw_weights_sha256': sha256_file(raw_weights),
        'compatible_weights_sha256': sha256_file(compat_weights),
        'num_tensors': len(converted_state),
        'legacy_vit_replacements': [
            list(pair) for pair in LEGACY_VIT_REPLACEMENTS
        ],
    }
    conversion_path = checkpoint / 'conversion.json'
    if conversion_path.exists():
        if json.loads(conversion_path.read_text()) != conversion:
            raise RuntimeError('Reacher conversion metadata mismatch')
    else:
        conversion_path.write_text(
            json.dumps(conversion, indent=2, ensure_ascii=False) + '\n'
        )

    config = json.loads((checkpoint / 'config.json').read_text())
    dimensions = {
        int(config['predictor']['input_dim']),
        int(config['predictor']['hidden_dim']),
        int(config['predictor']['output_dim']),
        int(config['action_encoder']['emb_dim']),
        int(config['projector']['input_dim']),
        int(config['projector']['output_dim']),
        int(config['pred_proj']['input_dim']),
        int(config['pred_proj']['output_dim']),
    }
    if dimensions != {192}:
        raise RuntimeError(
            f'official Reacher embedding dimensions are {dimensions}'
        )
    if int(config['action_encoder']['input_dim']) != 10:
        raise RuntimeError('official Reacher action block dimension is not 10')
    model = load_pretrained(checkpoint)
    model.load_state_dict(model.state_dict(), strict=True)
    del model
    print(f'Validated official Reacher checkpoint: {checkpoint}', flush=True)

    if args.skip_dataset:
        return
    target = dataset_root / 'reacher.h5'
    if target.exists() and target.stat().st_size > 0:
        print(f'Reusing existing Reacher dataset: {target}', flush=True)
        return
    archive = dataset_root / 'reacher.tar.zst'
    if not archive.exists():
        hf_hub_download(
            repo_id='quentinll/lewm-reacher',
            repo_type='dataset',
            filename='reacher.tar.zst',
            local_dir=dataset_root,
        )
    subprocess.run(
        ['tar', '--zstd', '-xf', str(archive), '-C', str(dataset_root)],
        check=True,
    )
    candidates = list(dataset_root.rglob('reacher.h5'))
    if not candidates:
        raise FileNotFoundError(
            f'archive did not contain reacher.h5: {archive}'
        )
    extracted = candidates[0]
    if extracted != target:
        os.replace(extracted, target)
    print(f'Prepared Reacher dataset: {target}', flush=True)


if __name__ == '__main__':
    main()
