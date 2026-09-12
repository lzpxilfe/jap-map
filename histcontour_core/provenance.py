"""Stable identities and provenance for reproducible Ink review outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


INK_ADAPTER_VERSION = "jap-map-ink-adapter/3"


def canonical_json(value: Mapping | Sequence | str | int | float | bool | None) -> str:
    """Encode small metadata deterministically for IDs and sidecars."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the digest of an input raster without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execution_id(*, raster_sha256: str, backend: str, upstream_commit: str, settings: Mapping, vectorization: Mapping) -> str:
    """Identify one exact Ink execution, independent of proposal ordering."""

    payload = {
        "adapter": INK_ADAPTER_VERSION,
        "backend": str(backend),
        "raster_sha256": str(raster_sha256),
        "settings": dict(settings),
        "upstream_commit": str(upstream_commit),
        "vectorization": dict(vectorization),
    }
    return "ink-run-" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:20]


def polyline_geometry_id(points: Iterable[Sequence[float]], *, precision: int = 4) -> str:
    """Hash a simplified pixel line after normalising its traversal direction."""

    if precision < 0:
        raise ValueError("precision must be non-negative")
    materialized = tuple((round(float(point[0]), precision), round(float(point[1]), precision)) for point in points)
    if len(materialized) < 2:
        raise ValueError("a segment geometry needs at least two points")
    canonical = min(materialized, tuple(reversed(materialized)))
    return "ink-segment-" + hashlib.sha256(canonical_json(canonical).encode("utf-8")).hexdigest()[:20]


__all__ = ["INK_ADAPTER_VERSION", "canonical_json", "execution_id", "polyline_geometry_id", "sha256_file"]
