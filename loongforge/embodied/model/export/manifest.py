# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``arch_manifest.json`` and ``capabilities.json`` for the native tier.

The bundle is the only cross-repository interface (J1,
``legacy_phyai_bundle.zh-CN.md`` §2). Its manifest is written here and
read on the PhyAI side by a shared preflight; neither side imports the
other, so the field set has to be narrow and typed, and the schema
version has to be checked against a whitelist rather than compared
with ``>=``: a reader that silently accepts a newer schema is the
failure mode the version field exists to prevent.

The native tier adds four fields the arch tier does not need
(§2.3):

* ``plugin`` -- which PhyAI plugin assembles this bundle. The arch
  tier always uses ``"arch"``, so the field is only interesting here.
* ``variant`` -- fingerprints of the fields whose *defaults* differ
  between the two sides. A field both sides default differently is
  the one way a correct ``config.json`` still yields wrong numbers.
* ``upstream`` -- the pretrained artefacts pinned by id and revision.
* ``entry_args`` -- scalars the plugin's ``Args`` needs that are not
  geometry.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass, field

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
BACKENDS = frozenset({"arch", "native"})

MANIFEST_FILENAME = "arch_manifest.json"
CAPABILITIES_FILENAME = "capabilities.json"

_SCALARS = (bool, int, float, str, type(None))


class ManifestError(RuntimeError):
    """A manifest field is absent, mistyped, or out of its whitelist."""


def _check_scalars(values: Mapping[str, object], where: str) -> None:
    """Reject any non-scalar value in a mapping bound for JSON.

    Args:
        values: Mapping to check.
        where: Field name, used in the message.

    Raises:
        ManifestError: A value is not a bool, int, float, str, or
            None. Tensors and callables are the two that matter:
            ``build_engine`` refuses them on the PhyAI side
            (``engine_builder.py:116-119``), so letting one into the
            manifest only moves the failure later.
    """
    for name, value in sorted(values.items()):
        if not isinstance(value, _SCALARS):
            raise ManifestError(
                f"{where}[{name!r}] is {type(value).__name__}; only "
                f"scalars cross the bundle boundary"
            )


@dataclass(frozen=True)
class Upstream:
    """One pretrained artefact the bundle depends on.

    Attributes:
        kind: ``"huggingface"`` or ``"pip"``.
        name: Repository id or package name.
        revision: Commit sha or version. Must not be a moving ref
            such as ``"main"``, which is what G8 pins.
    """

    kind: str
    name: str
    revision: str

    def to_dict(self) -> dict[str, str]:
        """Render as JSON-ready data."""
        return {"kind": self.kind, "name": self.name,
                "revision": self.revision}


@dataclass(frozen=True)
class NativeManifest:
    """The native-tier ``arch_manifest.json``.

    Attributes:
        model_type: LoongForge registry key, e.g. ``"pi05"``.
        plugin: PhyAI plugin name the bundle is meant for.
        variant: Fingerprints of fields the two sides default
            differently (§2.3). Scalars only.
        upstream: Pinned pretrained artefacts.
        entry_args: Non-geometry scalars for the plugin's ``Args``.
        weight_files: Relative names of the weight files written.
        backend: Always ``"native"`` for this class.
        schema_version: Bundle schema version.
    """

    model_type: str
    plugin: str
    variant: Mapping[str, object] = field(default_factory=dict)
    upstream: tuple[Upstream, ...] = ()
    entry_args: Mapping[str, object] = field(default_factory=dict)
    weight_files: tuple[str, ...] = ()
    backend: str = "native"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the whitelisted and scalar-only fields.

        Raises:
            ManifestError: Backend or schema version is outside its
                whitelist, a required string is empty, or a variant
                or entry-arg value is not a scalar.
        """
        if self.backend not in BACKENDS:
            raise ManifestError(
                f"backend {self.backend!r} is not one of "
                f"{sorted(BACKENDS)}"
            )
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ManifestError(
                f"schema_version {self.schema_version} is not one of "
                f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        for name in ("model_type", "plugin"):
            if not getattr(self, name):
                raise ManifestError(f"{name} must be a non-empty string")
        _check_scalars(self.variant, "variant")
        _check_scalars(self.entry_args, "entry_args")

    def to_dict(self) -> dict[str, object]:
        """Render as JSON-ready data, with keys in a fixed order."""
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "model_type": self.model_type,
            "plugin": self.plugin,
            "variant": dict(sorted(self.variant.items())),
            "upstream": [item.to_dict() for item in self.upstream],
            "entry_args": dict(sorted(self.entry_args.items())),
            "weight_files": list(self.weight_files),
        }


SUPPORT_LEVELS = frozenset({"EXACT", "DEGRADED", "UNSUPPORTED"})


@dataclass(frozen=True)
class Capabilities:
    """``capabilities.json``: what the receiving side may assume.

    Kept apart from the manifest because the manifest describes the
    artefact while this describes the *claim* about it, and because a
    DEGRADED entry is the one thing a reviewer should be able to read
    without scrolling past geometry.

    Attributes:
        level: ``"EXACT"``, ``"DEGRADED"``, or ``"UNSUPPORTED"``.
        notes: Why, one entry per deviation. Required unless EXACT.
        parallel_domains: Parallel domains the bundle may be loaded
            under, as the plugin's ``validate_parallel`` expects.
    """

    level: str
    notes: tuple[str, ...] = ()
    parallel_domains: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the level and demand a reason for a deviation.

        Raises:
            ManifestError: Level is outside the whitelist, or a
                non-EXACT level carries no note.
        """
        if self.level not in SUPPORT_LEVELS:
            raise ManifestError(
                f"level {self.level!r} is not one of "
                f"{sorted(SUPPORT_LEVELS)}"
            )
        if self.level != "EXACT" and not self.notes:
            raise ManifestError(
                f"level {self.level!r} needs at least one note saying "
                f"what deviates"
            )
        _check_scalars(self.parallel_domains, "parallel_domains")

    def to_dict(self) -> dict[str, object]:
        """Render as JSON-ready data."""
        return {
            "level": self.level,
            "notes": list(self.notes),
            "parallel_domains": dict(sorted(
                self.parallel_domains.items())),
        }


def write_json(path: pathlib.Path,
               payload: Mapping[str, object]) -> None:
    """Write JSON with a stable layout.

    Args:
        path: Destination file. Its parent must exist.
        payload: JSON-ready data.
    """
    text = json.dumps(payload, indent=2, sort_keys=False,
                      ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


