#!/usr/bin/env python3
"""Compare ONNX policy output with an exported TorchScript policy.pt.

Training checkpoints (model_*.pt) require IsaacLab/RSL-RL runner recreation.
For deployment, the most useful offline check is policy.pt vs policy.onnx,
both exported by scripts/rsl_rl/play.py with the same normalizer.
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to exported policy.onnx")
    parser.add_argument("--torchscript", required=True, help="Path to exported policy.pt")
    parser.add_argument("--obs-npy", default="", help="Optional .npy observation vector or batch")
    parser.add_argument("--obs-dim", type=int, default=56)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
        import torch
    except Exception as exc:
        raise SystemExit("Need torch and onnxruntime: pip install torch onnxruntime") from exc

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    obs_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else args.obs_dim

    if args.obs_npy:
        obs = np.load(args.obs_npy).astype(np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
    else:
        obs = np.zeros((1, obs_dim), dtype=np.float32)

    if obs.shape[-1] != obs_dim:
        raise SystemExit(f"Observation dim {obs.shape[-1]} does not match ONNX input {obs_dim}")

    onnx_action = np.asarray(sess.run([out.name], {inp.name: obs})[0], dtype=np.float32)
    module = torch.jit.load(args.torchscript, map_location="cpu")
    module.eval()
    with torch.no_grad():
        torch_action = module(torch.from_numpy(obs)).detach().cpu().numpy().astype(np.float32)

    diff = onnx_action - torch_action
    print(f"onnx action shape:  {onnx_action.shape}")
    print(f"torch action shape: {torch_action.shape}")
    print(f"max abs diff: {float(np.max(np.abs(diff))):.8f}")
    print(f"mean abs diff: {float(np.mean(np.abs(diff))):.8f}")
    if not np.allclose(onnx_action, torch_action, rtol=args.rtol, atol=args.atol):
        raise SystemExit("Mismatch too large. Check normalizer, export path, eval mode, and input order.")
    print("ONNX/TorchScript consistency OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
