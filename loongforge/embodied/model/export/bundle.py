# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Writing a bundle directory.

The on-disk layout is fixed by ``legacy_phyai_bundle.zh-CN.md`` §2.1
and is identical for both tiers, which is the whole point of the
design: one format, two backends. This module owns the layout, the
weight file naming, and the sharding, so that a per-model exporter
never builds a path by hand.

Two constraints come from the receiving side and are honoured here
rather than documented and hoped for:

* ``find_safetensors`` resolves ``model.safetensors.index.json``
  first, then ``model.safetensors``, then a glob over
  ``*.safetensors`` (``utils/checkpoint.py:115-157``). The glob branch
  would sweep up the arch tier's ``layout.safetensors``, so one of
  the first two must always exist -- enforced below.
* Only ``safetensors`` is used. ``torch.save`` would carry Python
  types across the repository boundary, which is exactly what the
  boundary is for.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Mapping

import torch
from safetensors.torch import save_file

from .manifest import (CAPABILITIES_FILENAME, MANIFEST_FILENAME,
                       Capabilities, NativeManifest, write_json)
from .report import REPORT_FILENAME, ExportError, ExportReport

WEIGHTS_FILENAME = "model.safetensors"
WEIGHTS_INDEX_FILENAME = "model.safetensors.index.json"
CONFIG_FILENAME = "config.json"

_BUNDLE_FILES = frozenset({
    MANIFEST_FILENAME, CAPABILITIES_FILENAME, REPORT_FILENAME,
    CONFIG_FILENAME, WEIGHTS_FILENAME, WEIGHTS_INDEX_FILENAME,
})


def _materialise(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Detach tensors to CPU and break any shared storage.

    ``save_file`` refuses two entries that share memory, and a
    :class:`~.keymap.Fanout` deliberately produces exactly that: one
    live tensor under several canonical names. A view into a larger
    buffer is refused for the same reason, so anything whose storage
    is bigger than itself is copied.

    Args:
        tensors: Canonical-name tensors from the live model.

    Returns:
        Independent CPU tensors, safe to hand to ``save_file``.

    Raises:
        ExportError: A value is not a tensor, or the mapping is empty.
    """
    if not tensors:
        raise ExportError("refusing to write a bundle with no tensors")
    seen: set[int] = set()
    out: dict[str, torch.Tensor] = {}
    for key in sorted(tensors):
        value = tensors[key]
        if not isinstance(value, torch.Tensor):
            raise ExportError(
                f"state dict entry {key!r} is "
                f"{type(value).__name__}, not a tensor"
            )
        cpu = value.detach().to("cpu")
        shared = cpu.data_ptr() in seen
        viewed = cpu.untyped_storage().nbytes() != cpu.nbytes
        out[key] = cpu.clone() if (shared or viewed
                                   or not cpu.is_contiguous()) else cpu
        seen.add(out[key].data_ptr())
    return out


def _shard(tensors: Mapping[str, torch.Tensor],
           max_shard_bytes: int) -> list[list[str]]:
    """Split keys into shards, greedily, in sorted key order.

    Args:
        tensors: Materialised tensors.
        max_shard_bytes: Soft cap per shard. A single tensor larger
            than the cap gets its own shard rather than an error.

    Returns:
        One list of keys per shard, never empty.
    """
    shards: list[list[str]] = [[]]
    used = 0
    for key in sorted(tensors):
        size = tensors[key].nbytes
        if shards[-1] and used + size > max_shard_bytes:
            shards.append([])
            used = 0
        shards[-1].append(key)
        used += size
    return shards


def _write_weights(dest: pathlib.Path,
                   tensors: Mapping[str, torch.Tensor],
                   max_shard_bytes: int | None) -> tuple[str, ...]:
    """Write the weight file or files, plus an index when sharded.

    Args:
        dest: Bundle directory, already created.
        tensors: Materialised tensors.
        max_shard_bytes: ``None`` for a single file.

    Returns:
        The relative names written, index first when there is one.
    """
    total = sum(t.nbytes for t in tensors.values())
    if max_shard_bytes is None or total <= max_shard_bytes:
        save_file(dict(tensors), str(dest / WEIGHTS_FILENAME),
                  metadata={"format": "pt"})
        return (WEIGHTS_FILENAME,)
    shards = _shard(tensors, max_shard_bytes)
    count = len(shards)
    weight_map: dict[str, str] = {}
    names: list[str] = []
    for index, keys in enumerate(shards, start=1):
        name = f"model-{index:05d}-of-{count:05d}.safetensors"
        save_file({k: tensors[k] for k in keys}, str(dest / name),
                  metadata={"format": "pt"})
        weight_map.update({k: name for k in keys})
        names.append(name)
    payload = {"metadata": {"total_size": total},
               "weight_map": dict(sorted(weight_map.items()))}
    (dest / WEIGHTS_INDEX_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return (WEIGHTS_INDEX_FILENAME, *names)


def _prepare_dest(dest: str | pathlib.Path,
                  overwrite: bool) -> pathlib.Path:
    """Resolve and create the bundle directory.

    Args:
        dest: Target directory.
        overwrite: Whether an existing bundle may be replaced.

    Returns:
        The resolved absolute path.

    Raises:
        ExportError: The directory already holds bundle files and
            ``overwrite`` is false, or the path is not a directory.
    """
    path = pathlib.Path(dest).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ExportError(f"bundle destination {path} is not a directory")
    if path.is_dir() and not overwrite:
        existing = sorted(p.name for p in path.iterdir()
                          if p.name in _BUNDLE_FILES)
        if existing:
            raise ExportError(
                f"bundle destination {path} already holds {existing}; "
                f"pass overwrite=True to replace it"
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_bundle(
    dest: str | pathlib.Path,
    *,
    tensors: Mapping[str, torch.Tensor],
    manifest: NativeManifest,
    capabilities: Capabilities,
    config: Mapping[str, object] | None = None,
    report: ExportReport | None = None,
    max_shard_bytes: int | None = None,
    overwrite: bool = False,
) -> pathlib.Path:
    """Write one bundle directory.

    Args:
        dest: Target directory. It is resolved to an absolute path,
            because that is what the plugin's ``checkpoint_dir`` has
            to be: ``bundle_dir`` is not on the engine's
            relative-path whitelist (``engine_builder.py:202-207``).
        tensors: Canonical-name tensors.
        manifest: Manifest to write. Its ``weight_files`` field is
            filled in from what actually gets written.
        capabilities: Support claim to write.
        config: Native-tier ``config.json`` payload. Required for the
            native backend, absent for the arch backend.
        report: Export-time self-check to write alongside.
        max_shard_bytes: Soft per-shard cap; ``None`` writes one file.
        overwrite: Whether an existing bundle may be replaced.

    Returns:
        The resolved bundle directory.

    Raises:
        ExportError: The destination is unusable, the tensors are
            empty or not tensors, a native bundle has no config, or a
            blocking gate recorded in ``report`` failed. Advisory
            gates are written down, not enforced.
    """
    if manifest.backend == "native" and config is None:
        raise ExportError(
            "a native-tier bundle needs config.json; the plugin "
            "resolves its config from checkpoint_dir/config.json"
        )
    if report is not None and report.failed:
        raise ExportError(
            f"refusing to write a bundle with failed blocking gates: "
            f"{list(report.failed)}"
        )
    path = _prepare_dest(dest, overwrite)
    written = _materialise(tensors)
    names = _write_weights(path, written, max_shard_bytes)
    write_json(path / MANIFEST_FILENAME,
               dataclasses.replace(manifest,
                                   weight_files=names).to_dict())
    write_json(path / CAPABILITIES_FILENAME, capabilities.to_dict())
    if config is not None:
        write_json(path / CONFIG_FILENAME, config)
    if report is not None:
        report.dest = str(path)
        report.written_keys = len(written)
        write_json(report.path(path), report.to_dict())
    return path


