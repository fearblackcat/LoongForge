# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Export a pi0.5 policy as a native-tier PhyAI bundle.

Design: ``docs/design/legacy_phyai_bundle.zh-CN.md`` §4.3. pi0.5 is
the mirror image of dreamzero (§4.4): the config is nearly free and
the keys need real work.

* **Keys: drop an alias branch, strip one prefix, drop two tied
  weights.** ``PI05Policy`` holds the *same* ``PI05Pytorch`` twice, at
  ``self.model`` and at ``self.pi05``
  (``modeling_pi05.py:977``, ``:982``), so ``state_dict()`` emits
  1626 entries for 813 parameters. The ``pi05.`` branch goes first,
  then ``model.`` is stripped, which lands exactly on the namespace
  the plugin's remap reads: ``paligemma_with_expert.*`` plus the four
  action/time projections (``modeling_pi05.py:335``, ``:528``,
  ``:778``, ``:921-942`` on the PhyAI side).
* **Config: 4 dataclasses, 40 fields, 3 of them nested.** The nesting
  is not optional here. ``vision``/``text``/``expert`` carry
  ``default_factory``, so ``from_dict`` rebuilds them recursively from
  nested dicts and a flat payload would leave all three at their
  defaults (§3.4).

The geometry has to be probed rather than read off the config:
``Pi05ModelConfig`` carries no backbone fields at all
(``model_configuration_pi05.py:55-76``). ``PI05Pytorch`` pins it from
two variant names via ``_get_gemma_config``
(``modeling_pi05.py:255-264``) and from HuggingFace's ``paligemma``
config defaults, and the live HF config objects are the only place all
40 numbers exist together.

One version-dependent read is isolated in :func:`_rope_theta`:
transformers 5.3 keeps ``rope_theta`` inside ``rope_parameters``,
while the plugin declares it flat.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..export import (Capabilities, ConfigSchema, Drop, ExportReport,
                      KeyMap, NativeManifest, StripPrefix, Upstream,
                      finalise_config, write_bundle)
from ..export.report import ExportError

PLUGIN_NAME = "pi05"
MODEL_TYPE = "pi05"

# The duplicated module alias and the one-dot module prefix, in the
# order :func:`key_map` applies them.
ALIAS_PREFIX = "pi05."
MODULE_PREFIX = "model."

# ``_PI05_CANONICAL_DROP_KEYS`` verbatim (``main_pi05.py:63-69``). The
# expert's ``lm_head`` has no inference-side parameter to land in, and
# the tied PaliGemma embedding is loaded through
# ``paligemma.lm_head.weight`` instead. Dropping them here rather than
# leaving them to the plugin's own filter saves writing a 257152x2048
# tensor twice.
CANONICAL_DROP_KEYS = frozenset({
    "paligemma_with_expert.gemma_expert.lm_head.weight",
    "paligemma_with_expert.paligemma.model.language_model."
    "embed_tokens.weight",
})

# The key that must survive: the plugin loads the tied embedding from
# it (``modeling_pi05.py:546``), so an export that dropped both tied
# aliases would fail ``strict=True`` on the embedding instead.
TIED_EMBEDDING_KEY = "paligemma_with_expert.paligemma.lm_head.weight"

# ---------------------------------------------------------------------
# G9: the field names PhyAI declares, inlined so that nothing here
# imports PhyAI. ``test_arch_export.py`` parses
# ``models/pi05/configuration_pi05.py`` with ``ast`` and compares these
# sets field by field.
# ---------------------------------------------------------------------
_VISION_FIELDS = frozenset({
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "intermediate_size", "image_size", "patch_size", "num_channels",
    "layer_norm_eps", "projection_dim",
})
_TEXT_FIELDS = frozenset({
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "intermediate_size",
    "vocab_size", "rms_norm_eps", "rope_theta",
    "max_position_embeddings",
})
_EXPERT_FIELDS = frozenset({
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "intermediate_size",
    "rms_norm_eps", "rope_theta", "max_position_embeddings",
    "use_adarms", "adarms_cond_dim",
})
_TOP_FIELDS = frozenset({
    "vision", "text", "expert", "chunk_size", "max_action_dim",
    "num_inference_steps", "min_period", "max_period",
    "tokenizer_max_length", "add_value_head", "value_after_vlm",
    "value_head_hidden_sizes", "critic_action_chunk",
})


CONFIG_SCHEMA = ConfigSchema(
    name="",
    fields=_TOP_FIELDS,
    children={
        "vision": ConfigSchema(name="vision", fields=_VISION_FIELDS),
        "text": ConfigSchema(name="text", fields=_TEXT_FIELDS),
        "expert": ConfigSchema(name="expert", fields=_EXPERT_FIELDS),
    },
)

# Flow-matching and tokenizer knobs. These live on
# ``_PI05InternalConfig`` (``modeling_pi05.py:73-100``) with no
# ``Pi05ModelConfig`` counterpart, so no run can change them, and both
# sides already agree on all four values.
_SAMPLER_DEFAULTS: Mapping[str, Any] = {
    "num_inference_steps": 10,
    "min_period": 4e-3,
    "max_period": 4.0,
    "tokenizer_max_length": 200,
}

# The RL critic exists on the PhyAI side only. Writing the defaults
# makes the absence of a critic explicit in ``config.json`` instead of
# leaving it to ``from_dict``.
_CRITIC_DEFAULTS: Mapping[str, Any] = {
    "add_value_head": False,
    "value_after_vlm": True,
    "value_head_hidden_sizes": (1024, 512, 256),
    "critic_action_chunk": None,
}

# The geometry ``PI05Pytorch`` pins when nothing overrides it: the
# ``gemma_2b`` / ``gemma_300m`` rows of ``_get_gemma_config``
# (``modeling_pi05.py:255-264``), the four SigLIP values
# ``PaliGemmaWithExpertModel`` sets explicitly
# (``modeling_pi05.py:401-419``), and HuggingFace's own ``paligemma``
# defaults for the rest. Used when no probe is supplied; a probe that
# disagrees is an error, not an override.
_DEFAULT_VISION: Mapping[str, Any] = {
    "hidden_size": 1152,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "intermediate_size": 4304,
    "image_size": 224,
    "patch_size": 14,
    "num_channels": 3,
    "layer_norm_eps": 1e-6,
    "projection_dim": 2048,
}
_DEFAULT_TEXT: Mapping[str, Any] = {
    "hidden_size": 2048,
    "num_hidden_layers": 18,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "intermediate_size": 16384,
    "vocab_size": 257152,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 8192,
}
_DEFAULT_EXPERT: Mapping[str, Any] = {
    "hidden_size": 1024,
    "num_hidden_layers": 18,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "intermediate_size": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 8192,
    "use_adarms": True,
    "adarms_cond_dim": 1024,
}


DEFAULT_GEOMETRY: Mapping[str, Mapping[str, Any]] = {
    "vision": _DEFAULT_VISION,
    "text": _DEFAULT_TEXT,
    "expert": _DEFAULT_EXPERT,
}


def key_map() -> KeyMap:
    """Return the three-operator transform pi0.5 needs.

    Order matters. The alias :class:`~..export.Drop` names the
    un-stripped ``pi05.`` form because it runs before the
    :class:`~..export.StripPrefix`, while the canonical drop names the
    stripped form because it runs after.

    Returns:
        A :class:`~..export.KeyMap` with three operators.
    """
    return KeyMap(ops=(
        Drop(prefixes=(ALIAS_PREFIX,),
             reason="PI05Policy.pi05 is the same module as "
                    "PI05Policy.model (modeling_pi05.py:982); the two "
                    "branches are byte-identical"),
        StripPrefix(MODULE_PREFIX),
        Drop(keys=CANONICAL_DROP_KEYS,
             reason="the plugin's own filter refuses both "
                    "(main_pi05.py:63-69)"),
    ))


def _rope_theta(hf_config: Any) -> float:
    """Read RoPE theta across the two transformers layouts.

    Args:
        hf_config: A HuggingFace text or Gemma config object.

    Returns:
        The theta value.

    Raises:
        ExportError: Neither layout carries it. Guessing 10000.0 here
            would be the silent-default failure the config bridge
            exists to prevent.
    """
    params = getattr(hf_config, "rope_parameters", None)
    if isinstance(params, Mapping) and "rope_theta" in params:
        return float(params["rope_theta"])
    theta = getattr(hf_config, "rope_theta", None)
    if theta is not None:
        return float(theta)
    raise ExportError(
        f"{type(hf_config).__name__} exposes rope_theta neither "
        f"directly nor under rope_parameters"
    )


def _read(hf_config: Any, fields: Mapping[str, Any],
          where: str) -> dict[str, Any]:
    """Copy one sub-config's declared fields off a live HF config.

    Args:
        hf_config: The HuggingFace config object to read.
        fields: The field set to copy, used only for its keys.
        where: Sub-config name, used in the message.

    Returns:
        One entry per key in ``fields``.

    Raises:
        ExportError: A field is absent. ``rope_theta`` is the one
            field read through :func:`_rope_theta` instead.
    """
    out: dict[str, Any] = {}
    for name in sorted(fields):
        if name == "rope_theta":
            out[name] = _rope_theta(hf_config)
            continue
        if not hasattr(hf_config, name):
            raise ExportError(
                f"{where} config {type(hf_config).__name__} has no "
                f"{name!r}, which the plugin declares"
            )
        out[name] = getattr(hf_config, name)
    return out


def probe_model(model: torch.nn.Module) -> dict[str, Any]:
    """Read the backbone geometry off a live pi0.5 policy.

    Args:
        model: A live ``PI05Policy``.

    Returns:
        ``{"vision": ..., "text": ..., "expert": ...}``, each a flat
        mapping over the plugin's declared fields.

    Raises:
        ExportError: The model does not expose the pi0.5 backbone, or
            a declared field is missing from a live HF config.
    """
    inner = getattr(model, "model", None)
    backbone = getattr(inner, "paligemma_with_expert", None)
    if backbone is None:
        raise ExportError(
            f"{type(model).__name__} has no "
            f"model.paligemma_with_expert; export expects a PI05Policy"
        )
    paligemma = backbone.paligemma.config
    return {
        "vision": _read(paligemma.vision_config, _VISION_FIELDS,
                        "vision"),
        "text": _read(paligemma.text_config, _TEXT_FIELDS, "text"),
        "expert": _read(backbone.gemma_expert.config, _EXPERT_FIELDS,
                        "expert"),
    }


def _check_probe(probe: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse a probe that departs from the pinned pi0.5 geometry.

    ``PI05Pytorch`` has no geometry knobs, so a probe that differs
    from :data:`DEFAULT_GEOMETRY` means the model was built by
    something this exporter has not been read against. Writing its
    numbers would still load on the plugin side -- and might even be
    right -- but it would do so unreviewed.

    Args:
        probe: :func:`probe_model` output.

    Raises:
        ExportError: Any field differs from the pinned value.
    """
    bad: list[str] = []
    for section, expected in DEFAULT_GEOMETRY.items():
        found = probe.get(section, {})
        bad.extend(
            f"{section}.{name}: expected={expected[name]!r} "
            f"found={found.get(name)!r}"
            for name in sorted(expected)
            if found.get(name) != expected[name]
        )
    if bad:
        raise ExportError(
            f"live pi0.5 geometry departs from the pinned pi05_base "
            f"values: {bad}"
        )


def translate_config(model_cfg: Any, *,
                     probe: Mapping[str, Any] | None = None,
                     ) -> dict[str, object]:
    """Render the plugin's nested ``config.json`` payload.

    Args:
        model_cfg: A ``Pi05ModelConfig``.
        probe: Optional :func:`probe_model` output. When absent, the
            pinned :data:`DEFAULT_GEOMETRY` is used.

    Returns:
        The validated, JSON-ready payload.

    Raises:
        ExportError: The probe departs from the pinned geometry, or
            the action width exceeds the padded width.
        ConfigBridgeError: A declared field is missing or an
            undeclared one was written (``config_bridge.py``).
    """
    if probe is not None:
        _check_probe(probe)
    geometry = probe if probe is not None else DEFAULT_GEOMETRY
    if model_cfg.action_dim > model_cfg.max_action_dim:
        raise ExportError(
            f"action_dim {model_cfg.action_dim} exceeds max_action_dim "
            f"{model_cfg.max_action_dim}; the plugin's action_in_proj "
            f"is built at max_action_dim"
        )
    return finalise_config(CONFIG_SCHEMA,
                           _payload(model_cfg, geometry))


def _payload(model_cfg: Any,
             geometry: Mapping[str, Any]) -> dict[str, object]:
    """Assemble the nested ``config.json`` body.

    ``chunk_size`` is the one non-obvious mapping: the plugin spends
    it as the suffix block length, and LoongForge calls the same
    number ``action_horizon``
    (``modeling_pi05.py:963-964`` passes it straight through as
    ``chunk_size``).

    Args:
        model_cfg: A ``Pi05ModelConfig``.
        geometry: Probed or pinned geometry.

    Returns:
        The payload, before validation.
    """
    return {
        "vision": dict(geometry["vision"]),
        "text": dict(geometry["text"]),
        "expert": dict(geometry["expert"]),
        "chunk_size": model_cfg.action_horizon,
        "max_action_dim": model_cfg.max_action_dim,
        **_SAMPLER_DEFAULTS,
        **_CRITIC_DEFAULTS,
    }

# Build-time choices ``PI05Pytorch`` makes that the plugin's ``Args``
# and config both default differently (§2.3).
# ``PaliGemmaWithExpertModel._FP32_PARAM_SELECTORS``
# (``modeling_pi05.py:384-386``) keeps the vision tower, the
# projector, and the norms in fp32 while the rest goes to bf16, which
# is what the plugin's ``vision_params_dtype`` reproduces.
_VARIANT_PINS: Mapping[str, Any] = {
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "use_adarms_text": False,
    "use_adarms_expert": True,
    "tied_embedding_source": TIED_EMBEDDING_KEY,
}


def _variant(model_cfg: Any) -> dict[str, object]:
    """Fingerprint the fields the two sides default differently.

    The last two entries have no plugin counterpart at all: pi0.5
    feeds the robot state through the language stream, so the plugin
    config has no state width and nothing would catch a mismatch.

    Args:
        model_cfg: A ``Pi05ModelConfig``.

    Returns:
        Scalars only, as :class:`~..export.NativeManifest` requires.
    """
    return {
        **_VARIANT_PINS,
        "loongforge_action_dim": model_cfg.action_dim,
        "loongforge_max_state_dim": model_cfg.max_state_dim,
    }


def export(model: torch.nn.Module,
           model_cfg: Any,
           dest: str,
           *,
           variant_probe: Mapping[str, Any] | None = None,
           upstream: tuple[Upstream, ...] = (),
           max_shard_bytes: int | None = None,
           overwrite: bool = False) -> ExportReport:
    """Write a native-tier bundle for a live pi0.5 policy.

    Args:
        model: A live ``PI05Policy``.
        model_cfg: Its ``Pi05ModelConfig``.
        dest: Bundle directory. Created if absent.
        variant_probe: Override for :func:`probe_model`.
        upstream: Pinned pretrained artefacts (G8).
        max_shard_bytes: Soft per-shard cap. pi0.5 is ~3.3 B
            parameters, so a caller that wants the HuggingFace
            convention passes 5 GiB here.
        overwrite: Whether an existing bundle may be replaced.

    Returns:
        The :class:`~..export.ExportReport` that was written.

    Raises:
        ExportError: A required gate failed, in which case nothing is
            written.
    """
    probe = (dict(variant_probe) if variant_probe is not None
             else probe_model(model))
    transform = key_map()
    report = ExportReport(model_type=MODEL_TYPE, plugin=PLUGIN_NAME,
                          key_ops=transform.describe())
    config = translate_config(model_cfg, probe=probe)
    report.config_fields = len(config)
    result = transform.apply_state_dict(model.state_dict())
    report.source_keys = len(result.tensors) + len(result.dropped)
    report.dropped_keys = result.dropped
    report.fanned_out = dict(result.fanned_out)
    leaked = sorted(key for key in result.tensors
                    if key.startswith((ALIAS_PREFIX, MODULE_PREFIX)))
    report.require(
        "no_module_prefix_survives", not leaked,
        f"{len(leaked)} key(s) still carry the LoongForge module "
        f"prefix: {leaked[:3]}",
    )
    report.require(
        "tied_embedding_survives", TIED_EMBEDDING_KEY in result.tensors,
        f"{TIED_EMBEDDING_KEY} is the only remaining copy of the tied "
        f"PaliGemma embedding",
    )
    report.require(
        "canonical_drops_applied",
        not (CANONICAL_DROP_KEYS & set(result.tensors)),
        f"the plugin refuses {sorted(CANONICAL_DROP_KEYS)}",
    )
    alias_drops = sum(1 for key in result.dropped
                      if key.startswith(ALIAS_PREFIX))
    report.record(
        "alias_branch_dropped", alias_drops == len(result.tensors) + 2,
        f"{alias_drops} alias key(s) dropped against "
        f"{len(result.tensors)} written plus 2 canonical drops",
    )
    report.advise("G8_upstream_pinned", bool(upstream),
                  f"{len(upstream)} artefact(s) pinned")
    variant = _variant(model_cfg)
    report.variant = variant
    manifest = NativeManifest(
        model_type=MODEL_TYPE, plugin=PLUGIN_NAME, variant=variant,
        upstream=tuple(upstream),
        entry_args={"weight_strict": True,
                    "vision_params_dtype": "float32"},
    )
    write_bundle(dest, tensors=result.tensors, manifest=manifest,
                 capabilities=Capabilities(level="EXACT"),
                 config=config, report=report,
                 max_shard_bytes=max_shard_bytes, overwrite=overwrite)
    return report
