# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``export_report.json``: the export-time self-check, on disk.

The gates in ``legacy_phyai_bundle.zh-CN.md`` §6 split in two: some
can only run on the receiving side (G2's ``strict=True`` load, G5's
cosine on a velocity field), and some can only run while the live
model is still in hand. The second group is what this file records --
which key transform ran, what it dropped, which config fields were
written, what was fingerprinted, and what was pinned.

Writing it down matters because a bundle outlives the process that
produced it. When a load fails three weeks later, the first question
is whether the exporter or the plugin changed, and that question is
only answerable if the exporter left a record.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

REPORT_FILENAME = "export_report.json"


class ExportError(RuntimeError):
    """An export-time blocking gate failed; no bundle was written."""


@dataclass(frozen=True)
class Check:
    """One export-time gate result.

    Attributes:
        name: Gate id, e.g. ``"G3_config_fields_explicit"``.
        passed: Whether the gate held.
        detail: Evidence, short enough to read in a list.
        blocking: Whether failing this gate must stop the write. A
            blocking gate guards correctness of the bundle itself; an
            advisory one records something a reader should know about
            but that does not make the bundle wrong. G8 (pinned
            upstream artefacts) is the advisory case: an unpinned
            export is less reproducible, not unloadable.
    """

    name: str
    passed: bool
    detail: str = ""
    blocking: bool = True

    def to_dict(self) -> dict[str, object]:
        """Render as JSON-ready data."""
        return {"name": self.name, "passed": self.passed,
                "detail": self.detail, "blocking": self.blocking}


@dataclass
class ExportReport:
    """Mutable accumulator for one export.

    Attributes:
        model_type: LoongForge registry key being exported.
        plugin: PhyAI plugin the bundle targets.
        dest: Bundle directory, as a string.
        checks: Gate results in the order they ran.
        key_ops: Rendered key-transform operator list.
        source_keys: Number of keys in the live state dict.
        written_keys: Number of keys written to the bundle.
        dropped_keys: Source keys no operator kept.
        fanned_out: Source key to its several canonical names.
        config_fields: Number of config fields written explicitly.
        variant: Variant fingerprint recorded in the manifest.
    """

    model_type: str
    plugin: str
    dest: str = ""
    checks: list[Check] = field(default_factory=list)
    key_ops: tuple[str, ...] = ()
    source_keys: int = 0
    written_keys: int = 0
    dropped_keys: tuple[str, ...] = ()
    fanned_out: Mapping[str, Sequence[str]] = field(default_factory=dict)
    config_fields: int = 0
    variant: Mapping[str, object] = field(default_factory=dict)

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        """Append a blocking gate result without raising yet.

        The write refuses any bundle whose report carries a failed
        blocking gate (``bundle.py``), so this defers the refusal
        rather than waiving it. Use :meth:`advise` for a gate that
        should be written down but should not stop the export.

        Args:
            name: Gate id.
            passed: Whether the gate held.
            detail: Evidence.
        """
        self.checks.append(Check(name=name, passed=passed,
                                 detail=detail))

    def advise(self, name: str, passed: bool, detail: str = "") -> None:
        """Append an advisory gate result, which never blocks.

        Args:
            name: Gate id.
            passed: Whether the gate held.
            detail: Evidence.
        """
        self.checks.append(Check(name=name, passed=passed,
                                 detail=detail, blocking=False))

    def require(self, name: str, passed: bool, detail: str = "") -> None:
        """Append a blocking gate result and raise when it failed.

        Args:
            name: Gate id.
            passed: Whether the gate held.
            detail: Evidence, also used as the message.

        Raises:
            ExportError: ``passed`` is false.
        """
        self.record(name, passed, detail)
        if not passed:
            raise ExportError(f"{name} failed: {detail}")

    @property
    def failed(self) -> tuple[str, ...]:
        """Return the ids of the blocking gates that did not hold."""
        return tuple(c.name for c in self.checks
                     if not c.passed and c.blocking)

    @property
    def unmet(self) -> tuple[str, ...]:
        """Return the ids of the advisory gates that did not hold."""
        return tuple(c.name for c in self.checks
                     if not c.passed and not c.blocking)

    def to_dict(self) -> dict[str, object]:
        """Render as JSON-ready data, with keys in a fixed order."""
        return {
            "model_type": self.model_type,
            "plugin": self.plugin,
            "dest": self.dest,
            "passed": not self.failed,
            "failed": list(self.failed),
            "unmet": list(self.unmet),
            "checks": [c.to_dict() for c in self.checks],
            "key_ops": list(self.key_ops),
            "source_keys": self.source_keys,
            "written_keys": self.written_keys,
            "dropped_keys": list(self.dropped_keys),
            "fanned_out": {k: list(v) for k, v
                           in sorted(self.fanned_out.items())},
            "config_fields": self.config_fields,
            "variant": dict(sorted(self.variant.items())),
        }

    def path(self, dest: pathlib.Path) -> pathlib.Path:
        """Return where the report belongs inside a bundle.

        Args:
            dest: Bundle directory.

        Returns:
            The report path.
        """
        return dest / REPORT_FILENAME


