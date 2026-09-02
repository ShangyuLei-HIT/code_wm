"""Download and validate the official Cube LeWM model and dataset."""

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
        help='Only prepare the 72 MB model checkpoint.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    raw_checkpoint = root / 'checkpoints' / 'official_lewm_cube_raw'
    checkpoint = root / 'checkpoints' / 'official_lewm_cube_compat'
    dataset_root = root / 'datasets' / 'quentinll--lewm-cube'
    raw_checkpoint.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    raw_weights = raw_checkpoint / 'weights.pt'
    compat_weights = checkpoint / 'weights.pt'
    if not raw_weights.exists() and compat_weights.exists():
        downloaded = torch.load(
            compat_weights, map_location='cpu', weights_only=True
        )
        if is_legacy_vit_state_dict(downloaded):
            for filename in ('config.json', 'weights.pt', 'README.md'):
                source = checkpoint / filename
                destination = raw_checkpoint / filename
                if source.exists() and not destination.exists():
                    shutil.copy2(source, destination)
            if sha256_file(raw_weights) != sha256_file(compat_weights):
                raise RuntimeError('failed to preserve official Cube weights')
            print(
                f'Preserved official Cube checkpoint: {raw_checkpoint}',
                flush=True,
            )

    required_raw = ('config.json', 'weights.pt')
    if not all((raw_checkpoint / name).exists() for name in required_raw):
        snapshot_download(
            repo_id='quentinll/lewm-cube',
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
                'existing Cube compat checkpoint has unexpected keys'
            )
    if write_compat:
        atomic_torch_save(converted_state, compat_weights)
        print(
            f'Converted official Cube checkpoint: {checkpoint}',
            flush=True,
        )

    conversion = {
        'source_repo': 'quentinll/lewm-cube',
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
            raise RuntimeError('Cube conversion metadata mismatch')
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
        raise RuntimeError(f'official Cube embedding dimensions are {dimensions}')
    if int(config['action_encoder']['input_dim']) != 25:
        raise RuntimeError('official Cube action block dimension is not 25')
    model = load_pretrained(checkpoint)
    model.load_state_dict(model.state_dict(), strict=True)
    del model
    print(f'Validated official Cube checkpoint: {checkpoint}', flush=True)

    if args.skip_dataset:
        return
    target = dataset_root / 'cube_single_expert.h5'
    if target.exists():
        print(f'Reusing existing Cube dataset: {target}', flush=True)
        return
    archive = hf_hub_download(
        repo_id='quentinll/lewm-cube',
        repo_type='dataset',
        filename='cube_single_expert.tar.zst',
        local_dir=dataset_root,
    )
    subprocess.run(
        ['tar', '--zstd', '-xf', archive, '-C', str(dataset_root)],
        check=True,
    )
    candidates = list(dataset_root.rglob('cube_single_expert.h5'))
    if not candidates:
        raise FileNotFoundError(
            f'archive did not contain cube_single_expert.h5: {archive}'
        )
    extracted = candidates[0]
    if extracted != target:
        os.replace(extracted, target)
    print(f'Prepared Cube dataset: {target}', flush=True)


if __name__ == '__main__':
    main()
