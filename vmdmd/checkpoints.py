"""Audited checkpoint imports. Load only trusted local checkpoint files."""
import hashlib
from pathlib import Path
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_state(path):
    # Original mmseg checkpoints contain Python metadata and optimizer state.
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint
    for key in ('state_dict', 'model_state_dict', 'model'):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise ValueError('Checkpoint has no state dictionary')
    state = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in state.items()
             if isinstance(v, torch.Tensor) and not k.endswith(('total_ops', 'total_params'))}
    return state, checkpoint.get('meta', {}) if isinstance(checkpoint, dict) else {}


def load_teacher(model, path):
    state, _ = read_state(path)
    if any(k.startswith('backbone.') for k in state):
        state = {k[len('backbone.'):]: v for k, v in state.items() if k.startswith('backbone.')}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = [k for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape]
    if missing or unexpected or mismatched:
        raise ValueError(f'Teacher checkpoint mismatch: missing={missing}, unexpected={unexpected}, shapes={mismatched}')
    model.load_state_dict(state, strict=True)
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path),
            'matched_tensors': len(state), 'coverage': 1.0}


def load_student_pretrained(model, path):
    state, _ = read_state(path)
    state = {(k[len('vmunet.'):] if k.startswith('vmunet.') else k): v for k, v in state.items()}
    target = model.state_dict()
    selected = {k: v for k, v in state.items() if k in target}
    for key, value in list(state.items()):
        if key.startswith('layers.'):
            stage = int(key.split('.')[1])
            decoder_key = key.replace(f'layers.{stage}.', f'layers_up.{3-stage}.', 1)
            if decoder_key in target:
                selected[decoder_key] = value
    mismatch = [k for k, v in selected.items() if v.shape != target[k].shape]
    if mismatch:
        raise ValueError(f'Student pretrained shape mismatch: {mismatch}')
    encoder_keys = [k for k in target if k.startswith(('patch_embed.', 'layers.'))]
    missing_encoder = [k for k in encoder_keys if k not in selected]
    if missing_encoder:
        raise ValueError(f'Incomplete student encoder initialization: {missing_encoder}')
    if not selected:
        raise ValueError('No student pretrained weights matched')
    model.load_state_dict(selected, strict=False)
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path),
            'matched_tensors': len(selected), 'missing': sorted(set(target)-set(selected)),
            'encoder_coverage': 1.0}


