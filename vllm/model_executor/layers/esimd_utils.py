# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility loader for the optional XPU ESIMD extensions."""

import importlib.metadata
import sys
from pathlib import Path

import torch


_ESIMD_EXTENSION_PREFIXES = (
    "custom_esimd_kernels",
    "custom_esimd_kernels_gemm",
    "custom_esimd_kernels_moe",
    "custom_esimd_kernels_lgrf",
)
_ESIMD_LOADED = False


def _find_esimd_package_dir() -> Path:
    try:
        distribution = importlib.metadata.distribution("custom-esimd-kernels-vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("custom-esimd-kernels-vllm is not installed") from exc

    package_dir = Path(distribution.locate_file("custom_esimd_kernels_vllm"))
    if package_dir.is_dir():
        return package_dir
    raise ImportError(f"custom ESIMD package directory is missing: {package_dir}")


def _load_esimd_extensions() -> None:
    global _ESIMD_LOADED
    if _ESIMD_LOADED:
        return

    package_dir = _find_esimd_package_dir()
    loaded = False
    for prefix in _ESIMD_EXTENSION_PREFIXES:
        candidates = sorted(package_dir.glob(f"{prefix}*.so"))
        if not candidates:
            continue
        torch.ops.load_library(str(candidates[0]))
        loaded = True

    if not loaded:
        search_path = ":".join(sys.path)
        raise ImportError(
            "custom ESIMD shared libraries are missing from "
            f"{package_dir}; Python search path: {search_path}"
        )
    _ESIMD_LOADED = True


def load_esimd_library(prefix: str) -> None:
    """Load an optional extension without importing the package namespace."""
    package_dir = _find_esimd_package_dir()
    candidates = sorted(package_dir.glob(f"{prefix}*.so"))
    if not candidates:
        raise ImportError(
            f"custom ESIMD shared library is missing from {package_dir}: {prefix}"
        )

    errors = []
    for candidate in candidates:
        try:
            torch.ops.load_library(str(candidate))
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
        else:
            return
    raise ImportError(
        f"custom ESIMD shared library could not be loaded: {'; '.join(errors)}"
    )


def get_esimd_op(name: str):
    """Return a registered ESIMD operator without importing optional modules.

    The wheel currently contains a mixture of Torch ABIs in its optional
    Python modules. The core shared libraries still register the operators
    used by the Gemma dense path, so load those libraries directly and avoid
    importing the package's all-or-nothing ``__init__``.
    """

    _load_esimd_extensions()
    try:
        return getattr(torch.ops.custom_esimd_kernels_vllm, name)
    except AttributeError as exc:
        raise ImportError(f"ESIMD operator is unavailable: {name}") from exc
