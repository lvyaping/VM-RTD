"""Explicit slow reference backend for tests, never an automatic CUDA fallback."""
import os
import torch
import torch.nn.functional as F


def selective_scan_reference(u, delta, A, B, C, D=None, z=None,
                             delta_bias=None, delta_softplus=False,
                             return_last_state=False):
    if A.is_complex() or B.ndim != 4 or C.ndim != 4:
        raise ValueError('Reference backend supports real, grouped SS2D tensors only')
    u, delta, A, B, C = (x.float() for x in (u, delta, A, B, C))
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, channels, length = u.shape
    groups = B.shape[1]
    if channels % groups:
        raise ValueError('Channels must be divisible by scan groups')
    B = B.repeat_interleave(channels // groups, dim=1)
    C = C.repeat_interleave(channels // groups, dim=1)
    state = u.new_zeros(batch, channels, A.shape[1])
    ys = []
    for t in range(length):
        dt = delta[:, :, t, None]
        state = torch.exp(dt * A[None]) * state + dt * B[:, :, :, t] * u[:, :, t, None]
        ys.append((state * C[:, :, :, t]).sum(-1))
    y = torch.stack(ys, -1)
    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z)
    return (y, state) if return_last_state else y


def select_backend():
    backend = os.environ.get('VMDMD_SCAN_BACKEND', 'cuda')
    if backend == 'reference':
        return selective_scan_reference
    if backend != 'cuda':
        raise ValueError('VMDMD_SCAN_BACKEND must be cuda or reference')
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except ImportError as exc:
        raise RuntimeError('CUDA training requires mamba-ssm. Use --scan-backend reference only for small tests.') from exc
    return selective_scan_fn
