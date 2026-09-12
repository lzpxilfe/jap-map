"""Optional PaddleOCR 3.7 adapter for preselected local margin crops.

No Paddle import occurs until a locally pinned model bundle is checked. No
model downloader or remote inference interface is exposed by this adapter.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from importlib import metadata
import os
from pathlib import Path
import socket
from unittest.mock import patch
from urllib.parse import urlparse

from .margin_ocr import MarginOcrError, digest_file, digest_value, required_text


PADDLEOCR_VERSION = "3.7.0"
PADDLEOCR_COMMIT = "b03f46425e8ff4442b268ce449e3eef758146cd4"
LOCK_SCHEMA = "jap-map-margin-ocr-models/1"
PACKAGES = ("paddleocr", "paddlex", "paddlepaddle")


def _inventory(directory: Path) -> dict:
    if directory.is_symlink() or not directory.is_dir():
        raise MarginOcrError(f"model directory is absent or a symlink: {directory}")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise MarginOcrError(f"model bundle contains a symlink: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest_file(path)}
    if not files or not any(name.endswith(".pdiparams") for name in files):
        raise MarginOcrError("local model bundle needs Paddle inference weights (.pdiparams); ONNX is not supported by this adapter")
    return files


def _source(spec: dict) -> None:
    for field in ("model_name", "source_url", "source_revision", "weight_license"):
        value = required_text(spec.get(field), field)
        if value.lower() in {"unknown", "unverified", "todo"} or "REPLACE" in value:
            raise MarginOcrError(f"fill verified model provenance first: {field}")
    source_url = urlparse(spec["source_url"])
    if source_url.scheme != "https" or not source_url.netloc:
        raise MarginOcrError("model source_url must describe an HTTPS source")


def _expected_files(entry: dict, files: dict) -> None:
    expected = entry.get("expected_files")
    if expected is not None and expected != {name: item["sha256"] for name, item in files.items()}:
        raise MarginOcrError("model files do not match the predeclared source fingerprints")


def pin_local_models(spec: dict, base: Path) -> dict:
    """Record a local inventory after the researcher supplies provenance.

    This fingerprints the supplied files; it does not prove their origin or
    grant a license. No URLs in the spec are fetched.
    """
    try:
        versions = {name: metadata.version(name) for name in PACKAGES}
    except metadata.PackageNotFoundError as error:
        raise MarginOcrError("pinning needs an isolated PaddleOCR/PaddleX/PaddlePaddle environment") from error
    if versions["paddleocr"] != PADDLEOCR_VERSION:
        raise MarginOcrError(f"this pilot adapter targets paddleocr=={PADDLEOCR_VERSION}")
    models = {}
    for role in ("detection", "recognition"):
        entry = spec.get("models", {}).get(role)
        if not isinstance(entry, dict):
            raise MarginOcrError(f"model spec needs {role}")
        _source(entry)
        directory = Path(required_text(entry.get("directory"), "model directory"))
        directory = directory if directory.is_absolute() else base / directory
        files = _inventory(directory)
        _expected_files(entry, files)
        models[role] = {**{key: entry[key] for key in ("model_name", "source_url", "source_revision", "weight_license")},
                        "directory": str(directory.resolve()), "files": files}
        if "expected_files" in entry:
            models[role]["expected_files"] = dict(entry["expected_files"])
    return {"schema": LOCK_SCHEMA, "paddleocr_commit": PADDLEOCR_COMMIT, "packages": versions,
            "models": models, "source_spec_sha256": digest_value(spec),
            "origin_verification": "supplied_provenance_with_optional_predeclared_fingerprints"}


def verify_models(lock: dict, base: Path, *, check_runtime: bool = True) -> dict:
    if lock.get("schema") != LOCK_SCHEMA or lock.get("paddleocr_commit") != PADDLEOCR_COMMIT:
        raise MarginOcrError("unknown PaddleOCR model lock schema or source revision")
    versions = lock.get("packages", {})
    if versions.get("paddleocr") != PADDLEOCR_VERSION or any(not isinstance(versions.get(name), str) or not versions[name] for name in PACKAGES):
        raise MarginOcrError("model lock must pin PaddleOCR, PaddleX, and PaddlePaddle versions")
    directories = {}
    for role in ("detection", "recognition"):
        entry = lock.get("models", {}).get(role)
        if not isinstance(entry, dict):
            raise MarginOcrError(f"model lock needs {role}")
        _source(entry)
        directory = Path(required_text(entry.get("directory"), "model directory"))
        directory = directory if directory.is_absolute() else base / directory
        files = _inventory(directory)
        if files != entry.get("files"):
            raise MarginOcrError(f"{role} model files differ from source-lock; refusing inference")
        _expected_files(entry, files)
        directories[role] = str(directory.resolve())
    if check_runtime:
        for name in PACKAGES:
            try:
                version = metadata.version(name)
            except metadata.PackageNotFoundError as error:
                raise MarginOcrError(f"optional runtime not installed: {name}") from error
            if version != versions[name]:
                raise MarginOcrError(f"{name} differs from model lock: {version} != {versions[name]}")
    return directories


@contextmanager
def local_inference_only(*, cache_directory: Path | None = None):
    """Block Python socket connections during this standalone CPU worker.

    Use only in the isolated CLI process; patching sockets in QGIS's host
    process would affect unrelated plugins. Local model directories are
    still mandatory even with this guard.
    """
    def denied(*_args, **_kwargs):
        raise MarginOcrError("network connection attempted during crop-only local OCR")
    with ExitStack() as stack:
        environment = {"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"}
        if cache_directory is not None:
            environment["PADDLE_PDX_CACHE_HOME"] = str(cache_directory.resolve())
        # PaddleX 3.7.2 reads these before its first import. The worker owns
        # this process; do not mutate environment flags in a running QGIS host.
        stack.enter_context(patch.dict(os.environ, environment))
        stack.enter_context(patch.object(socket.socket, "connect", denied))
        stack.enter_context(patch.object(socket.socket, "connect_ex", denied))
        stack.enter_context(patch.object(socket, "create_connection", denied))
        yield


def detection_options(det_max_side: int | None = None) -> dict:
    """Expose one frozen detector setting without altering archived crops."""
    if det_max_side is None:
        return {}
    if type(det_max_side) is not int or not 128 <= det_max_side <= 4096:
        raise MarginOcrError("det_max_side must be an integer in [128, 4096]")
    return {"text_det_limit_type": "max", "text_det_limit_side_len": det_max_side}


def create_engine(lock: dict, base: Path, *, det_max_side: int | None = None):
    options = detection_options(det_max_side)
    directories = verify_models(lock, base)
    # Optional dependency must never run on plugin import or project open.
    from paddleocr import PaddleOCR
    return PaddleOCR(
        text_detection_model_name=lock["models"]["detection"]["model_name"],
        text_detection_model_dir=directories["detection"],
        text_recognition_model_name=lock["models"]["recognition"]["model_name"],
        text_recognition_model_dir=directories["recognition"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=0.0,
        device="cpu",
        **options,
    )
