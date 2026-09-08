"""Optional PyTorch/ONNX image-context classifier trained only on synthetic Ink."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from histcontour_core.patch_context import patch_metadata, segment_context_tensor
from histcontour_core.synthetic_ink import iter_synthetic_cases
from scripts.run_synthetic_ink_experiment import records_for_case


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/derived/synthetic_ink_experiment/ink_patch_classifier.onnx"))
    parser.add_argument("--terrain-count", type=int, default=120)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--max-per-class", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=4)
    return parser.parse_args()


def _resolve(path):
    return path if path.is_absolute() else REPOSITORY / path


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise SystemExit("This optional image classifier needs PyTorch: python -m pip install torch onnx") from error
    return torch, nn


def _network(nn, channels):
    class TinyPatchClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(channels, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(32, 1)
        def forward(self, values):
            return self.head(self.features(values).flatten(1))
    return TinyPatchClassifier()


def _collect(terrain_count, variants, size, maximum):
    import numpy as np
    grouped = {"train": {0: [], 1: []}, "validation": {0: [], 1: []}, "test": {0: [], 1: []}}
    for case in iter_synthetic_cases(terrain_count=terrain_count, variants=variants, size=size):
        for record in records_for_case(case):
            if record["review_status"] not in {"contour", "text"}:
                continue
            label = int(record["review_status"] == "contour")
            bucket = grouped[case.split][label]
            if len(bucket) < maximum:
                bucket.append(segment_context_tensor(case.image, record["pixel_points"]))
    packed = {}
    for split, classes in grouped.items():
        values, labels = [], []
        for label, tensors in classes.items():
            values.extend(tensors)
            labels.extend([label] * len(tensors))
        if not values:
            raise ValueError(f"no eligible {split} patch examples")
        packed[split] = (np.stack(values).astype(np.float32), np.asarray(labels, dtype=np.float32))
    return packed


def main():
    args = parse_args()
    if args.epochs < 1 or args.max_per_class < 10:
        raise SystemExit("epochs must be positive and max-per-class must be at least 10")
    torch, nn = _torch()
    torch.manual_seed(20260908)
    data = _collect(args.terrain_count, args.variants, args.size, args.max_per_class)
    model = _network(nn, patch_metadata()["channels"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    train_x, train_y = (torch.from_numpy(item) for item in data["train"])
    for _epoch in range(args.epochs):
        order = torch.randperm(train_x.shape[0])
        for batch_start in range(0, len(order), 64):
            batch = order[batch_start:batch_start + 64]
            optimizer.zero_grad()
            loss = loss_fn(model(train_x[batch]).squeeze(1), train_y[batch])
            loss.backward()
            optimizer.step()
    model.eval()
    metrics = {}
    for split in ("validation", "test"):
        values, labels = (torch.from_numpy(item) for item in data[split])
        with torch.no_grad():
            probabilities = torch.sigmoid(model(values).squeeze(1))
        predicted = probabilities >= 0.5
        metrics[split] = {
            "count": int(labels.numel()),
            "accuracy": float((predicted == labels.bool()).float().mean()),
            "contour_recall": float(predicted[labels.bool()].float().mean()),
            "non_contour_reduction": float((~predicted[~labels.bool()]).float().mean()),
        }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, patch_metadata()["channels"], patch_metadata()["patch_size"], patch_metadata()["patch_size"]), dtype=torch.float32)
    try:
        torch.onnx.export(model, sample, str(output), input_names=["segment_context"], output_names=["contour_logit"], dynamic_axes={"segment_context": {0: "batch"}, "contour_logit": {0: "batch"}}, opset_version=17)
    except Exception as error:
        raise SystemExit(f"ONNX export failed; install a compatible onnx package: {error}") from error
    metadata = {"model_type": "synthetic_tiny_patch_classifier", "training_scope": "synthetic only", **patch_metadata(), "metrics": metrics}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
