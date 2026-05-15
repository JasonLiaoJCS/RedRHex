#!/usr/bin/env python3
"""Inspect and smoke-test a RedRhex policy.onnx."""

from __future__ import annotations

import argparse

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path")
    parser.add_argument("--expected-obs-dim", type=int, default=56)
    parser.add_argument("--expected-action-dim", type=int, default=12)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except Exception as exc:
        raise SystemExit("onnxruntime is not installed. Try: pip install onnxruntime") from exc

    sess = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"providers: {sess.get_providers()}")
    print(f"input name:  {inp.name}")
    print(f"input shape: {inp.shape}")
    print(f"input type:  {inp.type}")
    print(f"output name:  {out.name}")
    print(f"output shape: {out.shape}")
    print(f"output type:  {out.type}")

    obs_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else args.expected_obs_dim
    if obs_dim not in (args.expected_obs_dim, args.expected_obs_dim * 5):
        raise SystemExit(
            f"Unexpected ONNX input dim {obs_dim}. Expected {args.expected_obs_dim} "
            f"or {args.expected_obs_dim * 5} for policy+history."
        )

    obs = np.zeros((1, obs_dim), dtype=np.float32)
    action = np.asarray(sess.run([out.name], {inp.name: obs})[0], dtype=np.float32)
    print(f"zero-observation action shape: {action.shape}")
    print(f"zero-observation action finite: {np.isfinite(action).all()}")
    print(f"zero-observation action min/max: {float(np.min(action)):.6f} / {float(np.max(action)):.6f}")
    if action.reshape(-1).shape[0] != args.expected_action_dim:
        raise SystemExit(f"Unexpected action dim {action.reshape(-1).shape[0]}")
    if not np.isfinite(action).all():
        raise SystemExit("ONNX output contains NaN/Inf")
    print("ONNX I/O check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
