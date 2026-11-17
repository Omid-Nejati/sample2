#!/usr/bin/env python3
"""Post-hoc spectral drift / overlap audit for sample2 SpectralMoELinear checkpoints.

Usage:
    python measure_sear_spectral.py --checkpoint path/to/checkpoint.pt \
        --output-prefix spectral_audit

Reads state_dict entries .A/.B, .anchor_A/.anchor_B, .basis_U/.basis_V.
Does not require model weights, training data, or importing the MedQwenV2 repo.
All numerical results are measurements of the GIVEN checkpoint, not validation scores.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from itertools import combinations
import torch


def orthonormal_span(x: torch.Tensor) -> torch.Tensor:
    # Thin SVD is robust to rank-deficient factors, unlike unpivoted QR.
    u, s, _ = torch.linalg.svd(x.float(), full_matrices=False)
    if s.numel() == 0:
        return u[:, :0]
    tol = max(x.shape) * torch.finfo(s.dtype).eps * s.max()
    return u[:, s > tol]


def drift(reference: torch.Tensor, current: torch.Tensor):
    """Mean squared sine of principal angles of current vs fixed reference span."""
    if current.shape[1] == 0:
        return None
    captured = (reference.T @ current).square().sum().item()
    return float(max(0., min(1., 1 - captured / current.shape[1])))


def overlap(q1: torch.Tensor, q2: torch.Tensor):
    """Mean squared cosine of principal angles for two spans (higher = more overlap)."""
    denom = min(q1.shape[1], q2.shape[1])
    if denom == 0:
        return None
    return float(max(0., min(1., (q1.T @ q2).square().sum().item() / denom)))


def trace_correction_norm_sq(B, A, B0, A0):
    # Avoid materializing the potentially enormous output-by-input correction.
    # Use a difference-of-factors identity to avoid catastrophic cancellation
    # when BA and B0A0 are close: C=(B-B0)A0 + B(A-A0).
    left = torch.cat((B-B0, B), dim=1)
    right = torch.cat((A0, A-A0), dim=0)
    norm_sq = ((left.T @ left) * (right @ right.T).T).sum()
    # Equivalent to tr((left.T @ left) @ (right @ right.T)).
    return float(max(0., norm_sq.item())), left, right


def projection_capture(U, V, left, right, corr_norm_sq, relative_update, threshold):
    if corr_norm_sq <= 0 or relative_update < threshold:
        return None
    # U and V have orthonormal columns: ||P_U C P_V||_F = ||U.T C V||_F
    local = (U.T @ left) @ (right @ V)
    frac = float(local.square().sum().item() / corr_norm_sq)
    return float(max(0., min(1., frac)))


def load_state_dict(path: Path):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as e:
        raise RuntimeError(f'Cannot safely read {path}: {e}') from e
    if isinstance(checkpoint, dict):
        for key in ('trainable', 'state_dict', 'model_state_dict', 'model', 'adapter_state_dict'):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a state_dict or a checkpoint containing a state_dict')
    return checkpoint


def analyse(state, threshold=1e-4):
    prefixes = sorted(k[:-2] for k, v in state.items()
                      if k.endswith('.A') and isinstance(v, torch.Tensor) and v.ndim == 3)
    if not prefixes:
        raise ValueError('No .A tensors of shape [experts, rank, d_in] found. '
                         'This script expects sample2 SpectralMoELinear checkpoints.')
    rows, pairs = [], []
    required = ('A', 'B', 'anchor_A', 'anchor_B', 'basis_U', 'basis_V')
    for prefix in prefixes:
        missing = [k for k in required if prefix + '.' + k not in state]
        if missing:
            raise ValueError(f'{prefix}: missing {missing}. A trainable-only checkpoint '
                             'needs frozen anchors and bases saved as well.')
        a, b, a0, b0, u0, v0 = (state[prefix + '.' + x].detach().float().cpu()
                                    for x in required)
        if not (a.ndim == b.ndim == u0.ndim == v0.ndim == 3):
            raise ValueError(f'{prefix}: expected [expert,...] tensors')
        if not (a.shape[0] == b.shape[0] == u0.shape[0] == v0.shape[0]):
            raise ValueError(f'{prefix}: expert counts mismatch')
        spansB, spansA = [], []
        corrections = []
        correction_ratios = []
        for e in range(a.shape[0]):
            qb, qa = orthonormal_span(b[e]), orthonormal_span(a[e].T)
            spansB.append(qb)
            spansA.append(qa)
            c_norm_sq, left, right = trace_correction_norm_sq(b[e], a[e], b0[e], a0[e])
            corrections.append((left, right, c_norm_sq))
            # Small Gram matrix formula avoids a full output-by-input product.
            anchor_norm = float(torch.sqrt(((b0[e].T @ b0[e]) * (a0[e] @ a0[e].T).T).sum().clamp_min(0)))
            relative_update = c_norm_sq**0.5 / max(anchor_norm, 1e-12)
            correction_ratios.append(relative_update)
            captured = projection_capture(u0[e], v0[e], left, right,
                                          c_norm_sq, relative_update, threshold)
            rows.append(dict(layer=prefix, expert=e,
                             left_drift=drift(u0[e], qb),
                             right_drift=drift(v0[e], qa),
                             correction_energy=c_norm_sq,
                             correction_to_anchor_ratio=relative_update,
                             paired_basis_capture=captured,
                             paired_basis_leakage=None if captured is None else 1 - captured))
        for e, f in combinations(range(a.shape[0]), 2):
            le, re, ne = corrections[e]
            lf, rf, nf = corrections[f]
            correction_overlap = None
            if ne > 0 and nf > 0 and correction_ratios[e] >= threshold and correction_ratios[f] >= threshold:
                # <Le Re, Lf Rf>_F = <Le^T Lf, Re Rf^T>_F.
                corr_dot = ((le.T @ lf) * (re @ rf.T)).sum().item()
                correction_overlap = float(max(0., min(1., abs(corr_dot) / (ne * nf)**0.5)))
            pairs.append(dict(layer=prefix, expert_e=e, expert_f=f,
                              left_overlap=overlap(spansB[e], spansB[f]),
                              right_overlap=overlap(spansA[e], spansA[f]),
                              correction_overlap_abs_cosine=correction_overlap))
    return rows, pairs


def summarize(rows, pairs):
    out = {'layers': len({r['layer'] for r in rows}),
           'experts': len(rows), 'expert_pairs': len(pairs)}
    for key, data in [('left_drift', rows), ('right_drift', rows),
                      ('paired_basis_leakage', rows),
                      ('left_overlap', pairs), ('right_overlap', pairs),
                      ('correction_overlap_abs_cosine', pairs)]:
        values = [x[key] for x in data if x[key] is not None]
        out[key] = ({'mean': sum(values)/len(values), 'max': max(values), 'count': len(values)}
                    if values else {'mean': None, 'max': None, 'count': 0})
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-prefix', type=Path, default=Path('sear_spectral'))
    p.add_argument('--minimum-relative-correction', type=float, default=1e-4,
                   help='Skip paired-basis leakage when correction/anchor norm below this')
    args = p.parse_args()
    rows, pairs = analyse(load_state_dict(args.checkpoint),
                          threshold=args.minimum_relative_correction)
    summary = summarize(rows, pairs)
    prefix = str(args.output_prefix)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, data in [('_experts.csv', rows), ('_pairs.csv', pairs)]:
        write_csv(Path(prefix + suffix), data)
    Path(prefix + '_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Wrote {prefix}_experts.csv, {prefix}_pairs.csv, {prefix}_summary.json')


if __name__ == '__main__':
    main()
