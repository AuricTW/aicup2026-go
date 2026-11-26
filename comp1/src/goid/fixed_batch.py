"""Versioned, fixed-shape encoder; the frozen v1 encoder remains unchanged."""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np

from goid import inference as v1

POLICY = {
    "name": "fixed1024_postpreprocess_zero_pad_v2",
    "batch_rows": 1024,
    "tail": "Zero-pad the preprocessed MLP input to1024 rows; discard padded outputs",
    "feature_order": "lexically sorted unique content keys",
    "precision": "fp32",
    "autocast": False,
    "tf32": False,
    "deterministic_algorithms": True,
    "cpu_threads": 4,
    "cublas_workspace_config": ":4096:8",
    "statistical_transform": "unchanged v1, real rows only",
    "aggregation_and_rrf": "unchanged v1",
}


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def frozen_model(bundle, device):
    import torch
    from goid.feature_metric import FeatureMetric, FrozenFeaturePreprocessor
    synthetic = bundle["manifest"].get("synthetic_control", False)
    if device != "cuda" and not (synthetic and device == "cpu"):
        raise ValueError("Real inference requires local RTX3060 CUDA")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG conflicts with pinned policy")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda" and (not torch.cuda.is_available() or "RTX 3060" not in torch.cuda.get_device_name()):
        raise RuntimeError("Local RTX3060 CUDA unavailable")
    config = bundle["manifest"]["model_config"]
    pre = FrozenFeaturePreprocessor(bundle["path"], "standardized", 10)
    model = FeatureMetric(len(pre.indices), config["embedding_dim"], config["hidden_dim"], config["dropout"]).to(device)
    model.load_state_dict(torch.load(bundle["path"]/"model_state.pt", map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    return model, pre


def padded_input(preprocessed):
    """Padding happens after preprocessing and cannot enter player prototypes."""
    if preprocessed.dtype != np.float32 or preprocessed.ndim != 2 or not 1 <= len(preprocessed) <= 1024:
        raise ValueError("Expected1..1024 real preprocessed FP32 rows")
    value = np.zeros((1024, preprocessed.shape[1]), dtype=np.float32)
    value[:len(preprocessed)] = preprocessed
    return value


def encode_features(features, bundle, device="cuda"):
    import torch
    if features.ndim != 2 or features.shape[1] != v1.FEATURE_DIM or features.dtype != np.float32 or len(features) < 1:
        raise ValueError("Invalid feature matrix")
    model, pre = frozen_model(bundle, device)
    before = model_digest(model)
    statistical = np.empty((len(features), len(pre.indices)), np.float32)
    learned = np.empty((len(features), bundle["manifest"]["model_config"]["embedding_dim"]), np.float32)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
        for i in range(0, len(features), 1024):
            real = features[i:i+1024]
            statistical[i:i+len(real)] = v1.statistical_vectors(real, bundle)
            matrix = padded_input(pre(real))
            output = model(torch.from_numpy(matrix).to(device)).float().cpu().numpy()
            learned[i:i+len(real)] = output[:len(real)]
    after = model_digest(model)
    if before != after or any(p.grad is not None for p in model.parameters()) or not np.isfinite(learned).all():
        raise ValueError("Model mutated, gradient or nonfinite embedding")
    return (statistical, learned), {
        "policy": POLICY, "device": device,
        "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "model_before": before, "model_after": after, "unchanged": True,
        "optimizer_constructed": False, "backward_calls": 0, "fit_calls": 0,
        "real_rows": len(features), "batches": (len(features)+1023)//1024,
        "mlp_forward_rows_each_batch": 1024,
        "padding_rows_discarded": (-len(features)) % 1024,
        "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
        "elapsed_seconds": time.perf_counter()-start,
    }
