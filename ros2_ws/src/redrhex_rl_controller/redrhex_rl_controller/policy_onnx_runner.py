"""ONNX Runtime wrapper for RedRhex policy inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ONNXIOInfo:
    input_name: str
    input_shape: list[Any]
    input_type: str
    output_name: str
    output_shape: list[Any]
    output_type: str
    obs_dim: int | None
    action_dim: int | None
    providers: list[str]


class PolicyONNXRunner:
    """Small, defensive ONNX Runtime runner.

    The exported RSL-RL ONNX in this repo includes the actor observation
    normalizer when one exists because scripts/rsl_rl/play.py calls
    export_policy_as_onnx(policy_nn, normalizer=normalizer, ...).
    """

    def __init__(
        self,
        onnx_path: str,
        expected_obs_dim: int = 56,
        expected_action_dim: int = 12,
        use_cuda: bool = False,
        use_tensorrt: bool = False,
        allow_history_dim: bool = True,
    ) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - depends on Jetson install
            raise RuntimeError(
                "onnxruntime is required. Install on Jetson with onnxruntime or onnxruntime-gpu."
            ) from exc

        self._ort = ort
        self.onnx_path = str(Path(onnx_path).expanduser())
        if not Path(self.onnx_path).exists():
            raise FileNotFoundError(f"policy.onnx not found: {self.onnx_path}")

        available = ort.get_available_providers()
        providers: list[str] = []
        if use_tensorrt and "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if use_cuda and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(self.onnx_path, sess_options=session_options, providers=providers)
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        self.input_name = self.input.name
        self.output_name = self.output.name
        self.expected_obs_dim = int(expected_obs_dim)
        self.expected_action_dim = int(expected_action_dim)
        self.allow_history_dim = bool(allow_history_dim)

        self.obs_dim = self._last_static_dim(self.input.shape)
        self.action_dim = self._last_static_dim(self.output.shape)

        allowed_obs_dims = {self.expected_obs_dim}
        if self.allow_history_dim:
            allowed_obs_dims.add(self.expected_obs_dim * 5)
        if self.obs_dim is not None and self.obs_dim not in allowed_obs_dims:
            raise ValueError(
                f"ONNX input dim {self.obs_dim} is not compatible with expected "
                f"{sorted(allowed_obs_dims)}. Inspect policy export/obs history."
            )
        if self.action_dim is not None and self.action_dim != self.expected_action_dim:
            raise ValueError(
                f"ONNX output dim {self.action_dim} != expected action dim {self.expected_action_dim}."
            )

    @staticmethod
    def _last_static_dim(shape: list[Any]) -> int | None:
        if not shape:
            return None
        dim = shape[-1]
        if isinstance(dim, int) and dim > 0:
            return dim
        return None

    @property
    def io_info(self) -> ONNXIOInfo:
        return ONNXIOInfo(
            input_name=self.input_name,
            input_shape=list(self.input.shape),
            input_type=self.input.type,
            output_name=self.output_name,
            output_shape=list(self.output.shape),
            output_type=self.output.type,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            providers=list(self.session.get_providers()),
        )

    def run(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if obs.ndim != 2 or obs.shape[0] != 1:
            raise ValueError(f"Observation must have shape [1, N] or [N], got {obs.shape}.")
        if self.obs_dim is not None and obs.shape[1] != self.obs_dim:
            raise ValueError(f"Observation dim {obs.shape[1]} does not match ONNX input dim {self.obs_dim}.")
        if not np.isfinite(obs).all():
            raise ValueError("Observation contains NaN or Inf.")

        outputs = self.session.run([self.output_name], {self.input_name: obs})
        action = np.asarray(outputs[0], dtype=np.float32)
        if action.ndim == 2 and action.shape[0] == 1:
            action = action.reshape(-1)
        if action.shape != (self.expected_action_dim,):
            raise ValueError(f"Policy action shape {action.shape} != ({self.expected_action_dim},).")
        if not np.isfinite(action).all():
            raise ValueError("Policy action contains NaN or Inf.")
        return action
