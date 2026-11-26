# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Declarative key-name transforms for bundle export.

An exporter has to turn the key namespace of a *live* LoongForge module
into the canonical namespace that a PhyAI plugin's built-in remap
accepts (``legacy_phyai_bundle.zh-CN.md`` §3.3). The transform is
written here as an ordered tuple of small operators rather than as a
loop of ``if`` branches inside each per-model exporter, for two
reasons:

* it can be unit-tested as a pure function of one key, with no model
  and no checkpoint;
* a reviewer can diff the operator list against the plugin's remap
  without reading control flow.

Four operators cover the five A-class correspondences:
:class:`StripPrefix`, :class:`RenameSegment`, :class:`Drop`, and
:class:`Fanout`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch


class KeyMapError(RuntimeError):
    """A key transform produced a namespace the plugin cannot load."""


@runtime_checkable
class KeyOp(Protocol):
    """One step of a key-name transform."""

    def apply(self, key: str) -> tuple[str, ...]:
        """Map one key to zero or more output keys.

        Args:
            key: Key as it stands after the preceding operators.

        Returns:
            The output keys. An empty tuple drops the key.
        """


@dataclass(frozen=True)
class StripPrefix:
    """Remove a leading prefix, leaving non-matching keys alone.

    Attributes:
        prefix: Prefix to remove, including its trailing dot.
    """

    prefix: str

    def apply(self, key: str) -> tuple[str, ...]:
        """Strip :attr:`prefix` from ``key`` when it is present."""
        if key.startswith(self.prefix):
            return (key[len(self.prefix):],)
        return (key,)


@dataclass(frozen=True)
class RenameSegment:
    """Rewrite one dotted segment wherever it occurs.

    The match is a substring match on purpose: that is what PhyAI's
    own dict-shaped ``weight_remap`` does (``main_pi05.py:73-101``),
    so an operator with the same semantics can be read against it
    directly.

    Attributes:
        old: Substring to replace, conventionally dot-delimited.
        new: Replacement substring.
    """

    old: str
    new: str

    def apply(self, key: str) -> tuple[str, ...]:
        """Replace every occurrence of :attr:`old` in ``key``."""
        return (key.replace(self.old, self.new),)


@dataclass(frozen=True)
class Drop:
    """Remove keys the plugin's own remap would refuse.

    Both forms exist because both appear in the A-class plugins: pi05
    names two individual tied weights
    (``main_pi05.py:63-69``), while a duplicated module alias is a
    whole subtree (``modeling_pi05.py:982``).

    Attributes:
        keys: Keys dropped on an exact match.
        prefixes: Keys dropped when they start with one of these.
        reason: Why the drop is correct, quoted in the export report.
    """

    keys: frozenset[str] = frozenset()
    prefixes: tuple[str, ...] = ()
    reason: str = ""

    def apply(self, key: str) -> tuple[str, ...]:
        """Return ``()`` when ``key`` is dropped, else ``(key,)``."""
        if key in self.keys:
            return ()
        if any(key.startswith(p) for p in self.prefixes):
            return ()
        return (key,)


@dataclass(frozen=True)
class Fanout:
    """Write one source tensor out under several canonical keys.

    Needed where the two sides disagree on tying: a weight the live
    module holds once may be loaded by two PhyAI submodules.

    Attributes:
        source: Key to match exactly.
        targets: Keys to emit instead. Must not be empty.
    """

    source: str
    targets: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject a fanout with no target, which is a silent drop."""
        if not self.targets:
            raise KeyMapError(
                f"fanout from {self.source!r} has no target; use Drop "
                f"to remove a key"
            )

    def apply(self, key: str) -> tuple[str, ...]:
        """Expand ``key`` into :attr:`targets` on an exact match."""
        if key == self.source:
            return tuple(self.targets)
        return (key,)


@dataclass(frozen=True)
class KeyMapResult:
    """Outcome of running a :class:`KeyMap` over a state dict.

    Attributes:
        tensors: Canonical-name tensors, ready to be written.
        dropped: Source keys no operator kept, sorted.
        renamed: Source keys that survived under a different name.
        fanned_out: Source key to its several canonical names.
    """

    tensors: dict[str, torch.Tensor]
    dropped: tuple[str, ...]
    renamed: tuple[str, ...]
    fanned_out: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class KeyMap:
    """An ordered key-name transform.

    Attributes:
        ops: Operators, applied left to right. Each operator sees the
            keys the previous one produced, so a :class:`Drop` placed
            after a :class:`StripPrefix` must name the stripped form.
    """

    ops: tuple[KeyOp, ...] = field(default_factory=tuple)

    def apply(self, key: str) -> tuple[str, ...]:
        """Run every operator over one key.

        Args:
            key: Key as it appears in the live ``state_dict()``.

        Returns:
            The canonical keys, possibly empty.
        """
        current = (key,)
        for op in self.ops:
            nxt: list[str] = []
            for item in current:
                nxt.extend(op.apply(item))
            current = tuple(nxt)
            if not current:
                return ()
        return current

    def apply_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> KeyMapResult:
        """Rewrite a whole state dict into the canonical namespace.

        Args:
            state_dict: Live ``state_dict()`` of the module to export.

        Returns:
            A :class:`KeyMapResult`.

        Raises:
            KeyMapError: Two source keys collide on one canonical
                name. The collision is refused rather than resolved,
                because the plugin's loader refuses it too
                (``weights/loader.py:460-465``), and because silently
                keeping one of the two would export whichever tensor
                happened to come last.
        """
        tensors: dict[str, torch.Tensor] = {}
        origin: dict[str, str] = {}
        dropped: list[str] = []
        renamed: list[str] = []
        fanned_out: dict[str, tuple[str, ...]] = {}
        for src, value in state_dict.items():
            targets = self.apply(src)
            if not targets:
                dropped.append(src)
                continue
            if len(targets) > 1:
                fanned_out[src] = targets
            elif targets[0] != src:
                renamed.append(src)
            for dst in targets:
                if dst in tensors:
                    raise KeyMapError(
                        f"keys {origin[dst]!r} and {src!r} both map to "
                        f"{dst!r}; drop one of them explicitly"
                    )
                tensors[dst] = value
                origin[dst] = src
        return KeyMapResult(
            tensors=tensors,
            dropped=tuple(sorted(dropped)),
            renamed=tuple(sorted(renamed)),
            fanned_out=dict(fanned_out),
        )

    def describe(self) -> tuple[str, ...]:
        """Render the operator list for ``export_report.json``.

        Returns:
            One human-readable line per operator, in order.
        """
        lines: list[str] = []
        for op in self.ops:
            if isinstance(op, StripPrefix):
                lines.append(f"strip_prefix {op.prefix!r}")
            elif isinstance(op, RenameSegment):
                lines.append(f"rename_segment {op.old!r} -> {op.new!r}")
            elif isinstance(op, Drop):
                lines.append(
                    f"drop keys={sorted(op.keys)} "
                    f"prefixes={list(op.prefixes)}"
                    + (f" ({op.reason})" if op.reason else "")
                )
            elif isinstance(op, Fanout):
                lines.append(
                    f"fanout {op.source!r} -> {list(op.targets)}"
                )
            else:
                lines.append(type(op).__name__)
        return tuple(lines)


def identity() -> KeyMap:
    """Return the empty transform.

    Used by models whose live module tree already carries the
    canonical names, such as dreamzero (§4.4).

    Returns:
        A :class:`KeyMap` with no operators.
    """
    return KeyMap(ops=())


