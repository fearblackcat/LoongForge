# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``config.json`` translation and its reflection assertion.

PhyAI's ``PretrainedConfig.from_dict`` filters an incoming dict down
to the declared field names and **silently drops** everything else,
and every field carries a default
(``models/configuration.py:96-132``). So a misspelled or forgotten
geometry key does not raise: the config falls back to its default and
the model merely produces wrong numbers. That is the failure class
§3.4 exists to close, and it is closed in two directions at once:

* every declared field must be written *explicitly*, so a field the
  translator forgot is a hard error rather than a silent default;
* every written key must be declared, so a key the translator
  misspelled is a hard error rather than a silent drop.

The declared field set is an inlined constant in each
``export_<name>.py`` rather than a reflection over an imported PhyAI
class, because the bundle is the only cross-repository interface and
importing PhyAI here would break it. The constant is kept honest by a
test that parses PhyAI's ``configuration_*.py`` with :mod:`ast` (G9).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


class ConfigBridgeError(RuntimeError):
    """A translated config does not match the declared field set."""


@dataclass(frozen=True)
class ConfigSchema:
    """The field names of one PhyAI config dataclass.

    Attributes:
        name: Dotted path of this level, ``""`` at the top.
        fields: Every declared field name at this level.
        children: Sub-config fields, by field name. Each key must
            also appear in :attr:`fields`.
    """

    name: str
    fields: frozenset[str]
    children: Mapping[str, "ConfigSchema"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that every child is also a declared field.

        Raises:
            ConfigBridgeError: A child names a field this level does
                not declare, which would make the child unreachable.
        """
        unknown = sorted(set(self.children) - set(self.fields))
        if unknown:
            raise ConfigBridgeError(
                f"schema {self.name or '<top>'!r} declares children "
                f"{unknown} that are not fields"
            )

    def label(self) -> str:
        """Return a readable name for messages."""
        return self.name or "<top>"


def validate_config(schema: ConfigSchema,
                    written: Mapping[str, object]) -> None:
    """Assert a translated config covers exactly its declared fields.

    Args:
        schema: Declared field set, nested like the config itself.
        written: What the translator produced for this level.

    Raises:
        ConfigBridgeError: A declared field was never written, a
            written key is not declared, or a sub-config level was
            written as something other than a mapping.
    """
    missing = sorted(set(schema.fields) - set(written))
    if missing:
        raise ConfigBridgeError(
            f"config fields never written explicitly at "
            f"{schema.label()}: {missing}"
        )
    extra = sorted(set(written) - set(schema.fields))
    if extra:
        raise ConfigBridgeError(
            f"config keys not declared by PhyAI at {schema.label()}: "
            f"{extra}; from_dict would drop them silently"
        )
    for name, child in sorted(schema.children.items()):
        value = written[name]
        if not isinstance(value, Mapping):
            raise ConfigBridgeError(
                f"sub-config {child.label()!r} was written as "
                f"{type(value).__name__}, expected a mapping"
            )
        validate_config(child, value)

_JSON_SCALARS = (bool, int, float, str, type(None))


def _normalise(value: object, where: str) -> object:
    """Convert one config value into JSON-ready data.

    Tuples become lists because that is what JSON has; the receiving
    dataclass field may be declared ``tuple[int, ...]``, and
    ``from_dict`` does not coerce, so the only thing this guarantees
    is that the written file and the re-read value agree.

    Args:
        value: Value produced by a translator.
        where: Dotted path, used in the message.

    Returns:
        The JSON-ready value.

    Raises:
        ConfigBridgeError: The value is of a type JSON cannot carry,
            such as a tensor, an enum, or a model object.
    """
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, Mapping):
        return {str(k): _normalise(v, f"{where}.{k}")
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v, f"{where}[{i}]")
                for i, v in enumerate(value)]
    raise ConfigBridgeError(
        f"config value at {where} is {type(value).__name__}; only "
        f"JSON types cross the bundle boundary"
    )


def finalise_config(schema: ConfigSchema,
                    written: Mapping[str, object]) -> dict[str, object]:
    """Validate a translated config and return it JSON-ready.

    Args:
        schema: Declared field set.
        written: Translator output.

    Returns:
        The same data with tuples flattened to lists.

    Raises:
        ConfigBridgeError: Validation failed, per
            :func:`validate_config` and :func:`_normalise`.
    """
    validate_config(schema, written)
    result = _normalise(written, schema.label())
    assert isinstance(result, dict)
    return result


