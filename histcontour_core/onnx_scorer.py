"""Optional ONNX Runtime CPU inference for synthetic patch classifiers."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .patch_context import PATCH_CONTEXT_SCHEMA, patch_metadata, segment_context_tensor


class OnnxScorerUnavailable(RuntimeError):
    """Raised when the optional ONNX Runtime CPU dependency is absent."""


class OnnxScorer:
    def __init__(self, model_path: str | Path):
        path = Path(model_path)
        metadata_path = path.with_suffix(path.suffix + ".json")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise OnnxScorerUnavailable("ONNX contour scoring requires onnxruntime CPU; install it explicitly") from error
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("ONNX model metadata sidecar is missing or malformed") from error
        expected = patch_metadata()
        if metadata.get("schema") != PATCH_CONTEXT_SCHEMA or any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("ONNX patch model metadata does not match the current segment context schema")
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self.metadata = metadata

    def probability(self, image, points) -> float:
        tensor = segment_context_tensor(image, points)[None, ...]
        result = float(self._session.run(None, {self._input_name: tensor})[0].reshape(-1)[0])
        if result >= 0:
            return 1.0 / (1.0 + math.exp(-result))
        value = math.exp(result)
        return value / (1.0 + value)


__all__ = ["OnnxScorer", "OnnxScorerUnavailable"]
