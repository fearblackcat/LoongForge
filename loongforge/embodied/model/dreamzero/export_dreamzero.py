# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Export a DreamZero policy as a native-tier PhyAI bundle.

Design: ``docs/design/legacy_phyai_bundle.zh-CN.md`` §4.4. DreamZero is
the cheapest of the five A-class correspondences on the key side and
the most expensive on the config side:

* **Keys: no transform at all.** ``WANPolicyHead`` holds its four
  components at ``action_head.{model,text_encoder,image_encoder,vae}``
  (``modules/action_head_tf.py:240-282``), and that is exactly the
  namespace the plugin's four remaps accept
  (``modeling_dreamzero.py:1236-1244``, ``text_encoder_wan.py:25-33``,
  ``image_encoder_wan.py:26-49``, ``vae_wan.py:679-685``). All four
  remaps run against the *same* ``checkpoint_dir``
  (``builder_dreamzero.py:100-181``), so one weight file carries all
  four components and each remap selects its own subtree.
* **Config: 5 dataclasses, 67 fields, 4 of them nested.** The plugin
  declares 21 ``nested_sources`` aliases rooted at
  ``action_head_cfg.config.*``, but the exporter writes the flat
  canonical names instead (§3.4), which also sidesteps
  ``DreamZeroConfig.from_dict``'s ``hidden_size`` special case: that
  branch only fires when ``action_head_cfg`` is present, and it never
  is here.

One hard limit, enforced rather than documented: the plugin's VAE
supports Wan2.1 only. ``DreamZeroVAEConfig.__post_init__`` raises
unless ``z_dim == 16``, and the decoder reshapes 16 hardcoded latent
statistics with ``view(1, z_dim, 1, 1, 1)``
(``vae_wan.py:460-510``). LoongForge's *default* backbone variant is
``wan22_5b`` with ``z_dim = 48``, so the common case is the
unsupported one and has to fail loudly at export time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..export import (Capabilities, ConfigSchema, ExportReport, KeyMap,
                      NativeManifest, Upstream, finalise_config,
                      identity, write_bundle)
from ..export.report import ExportError

PLUGIN_NAME = "dreamzero"
MODEL_TYPE = "dreamzero"

# The plugin's VAE is Wan2.1-only; see the module docstring.
SUPPORTED_BACKBONE_VARIANTS = frozenset({"wan21_14b"})

# ---------------------------------------------------------------------
# G9: the field names PhyAI declares, inlined so that nothing here
# imports PhyAI. ``test_arch_export.py`` parses
# ``models/dreamzero/configuration_dreamzero.py`` with ``ast`` and
# compares these sets field by field.
# ---------------------------------------------------------------------
_DIT_FIELDS = frozenset({
    "dim", "eps", "ffn_dim", "frame_seqlen", "freq_dim", "in_dim",
    "max_chunk_size", "num_action_per_block", "num_frame_per_block",
    "num_heads", "num_layers", "num_state_per_block", "out_dim",
})
_TEXT_ENCODER_FIELDS = frozenset({
    "vocab", "dim", "dim_attn", "dim_ffn", "num_heads", "num_layers",
    "num_buckets", "shared_pos", "max_length",
})
_IMAGE_ENCODER_FIELDS = frozenset({
    "image_size", "patch_size", "vision_dim", "vision_mlp_ratio",
    "vision_heads", "vision_layers", "vision_pool", "vision_pre_norm",
    "vision_post_norm", "activation", "norm_eps", "num_channels",
    "feature_layer",
})
_VAE_FIELDS = frozenset({
    "z_dim", "base_dim", "decoder_base_dim", "dim_mult",
    "num_res_blocks", "temporal_downsample", "out_channels",
    "patch_size", "scale_factor_temporal", "scale_factor_spatial",
})
_TOP_FIELDS = frozenset({
    "dit", "text_encoder", "image_encoder", "vae", "action_dim",
    "action_horizon", "max_action_dim", "max_state_dim", "hidden_size",
    "num_frames", "num_inference_timesteps", "num_timestep_buckets",
    "num_frame_per_block", "cfg_scale", "sigma_shift",
    "decouple_inference_noise", "video_inference_final_noise", "tiled",
    "tile_size_height", "tile_size_width", "tile_stride_height",
    "tile_stride_width",
})

CONFIG_SCHEMA = ConfigSchema(
    name="",
    fields=_TOP_FIELDS,
    children={
        "dit": ConfigSchema(name="dit", fields=_DIT_FIELDS),
        "text_encoder": ConfigSchema(name="text_encoder",
                                     fields=_TEXT_ENCODER_FIELDS),
        "image_encoder": ConfigSchema(name="image_encoder",
                                      fields=_IMAGE_ENCODER_FIELDS),
        "vae": ConfigSchema(name="vae", fields=_VAE_FIELDS),
    },
)

# ``dreamzero_provider._build_text_encoder`` and
# ``_build_image_encoder`` pass **no** geometry arguments
# (``dreamzero_provider.py:665-695``), so these three components are
# pinned by the LoongForge constructor defaults rather than by the
# model config. The numbers are repeated here instead of read off the
# live module because the modules do not expose all of them as
# attributes; ``test_arch_export.py`` compares each entry against the
# constructor default via ``inspect.signature``.
_TEXT_ENCODER_GEOMETRY: Mapping[str, Any] = {
    "vocab": 256384,
    "dim": 4096,
    "dim_attn": 4096,
    "dim_ffn": 10240,
    "num_heads": 64,
    "num_layers": 24,
    "num_buckets": 32,
    "shared_pos": False,
    "max_length": 512,
}

# ``clip_xlm_roberta_vit_h_14``'s cfg dict
# (``modules/wan_video_image_encoder.py:829-856``) plus the
# ``VisionTransformer`` norm defaults (``:410-433``).
# ``feature_layer = 31`` is the ``use_31_block=True`` call at
# ``:886``: blocks 0..30 are used, block 31 is not, and the plugin's
# remap drops exactly that range.
_IMAGE_ENCODER_GEOMETRY: Mapping[str, Any] = {
    "image_size": 224,
    "patch_size": 14,
    "vision_dim": 1280,
    "vision_mlp_ratio": 4,
    "vision_heads": 16,
    "vision_layers": 32,
    "vision_pool": "token",
    "vision_pre_norm": True,
    "vision_post_norm": False,
    "activation": "gelu",
    "norm_eps": 1e-5,
    "num_channels": 3,
    "feature_layer": 31,
}

# ``VideoVAE``'s own constructor defaults
# (``modules/wan_video_vae.py:1080-1105``), reached because
# ``WanVideoVAE(z_dim=16)`` forwards only ``z_dim``. ``patch_size = 1``
# and ``out_channels = 3`` are the two values
# ``DreamZeroVAEConfig.__post_init__`` insists on; the two scale
# factors are the Wan2.1 compression ratios, matching
# ``WanVideoVAE.upsampling_factor = 8`` and the three-stage temporal
# downsample below.
_VAE_GEOMETRY: Mapping[str, Any] = {
    "z_dim": 16,
    "base_dim": 96,
    "decoder_base_dim": 96,
    "dim_mult": (1, 2, 4, 4),
    "num_res_blocks": 2,
    "temporal_downsample": (False, True, True),
    "out_channels": 3,
    "patch_size": 1,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 8,
}

# Sampler knobs that exist on the PhyAI side only; LoongForge's
# config has no counterpart, so the PhyAI defaults are restated here
# rather than left out. Leaving them out would be the silent-drop
# failure mode of §3.4: ``from_dict`` fills a missing field from its
# own default, so an omitted field and a field written with the same
# value are indistinguishable at load time but not at review time.
_INFERENCE_DEFAULTS: Mapping[str, Any] = {
    "cfg_scale": 5.0,
    "sigma_shift": 5.0,
    "decouple_inference_noise": False,
    "video_inference_final_noise": 0.8,
}


def key_map() -> KeyMap:
    """Return the key transform, which is empty for DreamZero.

    ``WANPolicyHead`` already holds its four components under the
    ``action_head.{model,text_encoder,image_encoder,vae}`` names the
    plugin's four remaps strip, so there is nothing to rewrite. The
    two key groups a transform might have had to remove are removed
    by the plugin itself: ``log_scale`` and ``visual.head*`` by
    ``dreamzero_image_encoder_weight_remap``
    (``image_encoder_wan.py:26-49``), and the unused upper vision
    blocks by the same remap's ``feature_layer`` range. The textual
    tower never appears at all -- ``XLMRobertaCLIP`` sets
    ``self.textual = None`` (``wan_video_image_encoder.py:709``).

    Returns:
        The identity :class:`~..export.KeyMap`.
    """
    return identity()

# Geometry the live ``CausalWanModel`` stores as attributes
# (``modules/causal_wan_model.py:2259-2287``). ``max_chunk_size`` is
# deliberately absent: the module keeps only the derived
# ``local_attn_size``, so that one field has to come from the config.
_PROBE_FIELDS: tuple[str, ...] = (
    "action_dim", "dim", "eps", "ffn_dim", "frame_seqlen", "freq_dim",
    "hidden_size", "in_dim", "max_state_dim", "num_action_per_block",
    "num_frame_per_block", "num_heads", "num_layers",
    "num_state_per_block", "out_dim",
)

# Which LoongForge config field each probe field is also derivable
# from, so that ``translate_config`` stays a pure function of the
# config and the probe can be used to cross-check it. The mapping is
# not the identity anywhere, which is the reason it is written out.
_CONFIG_SOURCES: Mapping[str, str] = {
    "action_dim": "max_action_dim",
    "dim": "backbone_hidden_size",
    "ffn_dim": "ffn_hidden_size",
    "frame_seqlen": "backbone_frame_seqlen",
    "freq_dim": "backbone_freq_dim",
    "hidden_size": "dit_action_state_hidden_size",
    "in_dim": "backbone_in_dim",
    "max_state_dim": "max_state_dim",
    "num_action_per_block": "num_action_per_block",
    "num_frame_per_block": "num_frame_per_block",
    "num_heads": "num_attention_heads",
    "num_layers": "num_layers",
    "num_state_per_block": "num_state_per_block",
    "out_dim": "backbone_out_dim",
}

# ``dreamzero_provider._build_diffusion_model`` passes no ``eps``
# (``dreamzero_provider.py:753-776``), so the DiT's own constructor
# default is the only source. ``DreamZeroDiTConfig.eps`` defaults to
# the same value, which is why this is a constant and not a mapping
# onto ``DreamZeroConfig.norm_epsilon``: that field feeds Megatron
# utilities, not the DiT.
_DIT_EPS = 1e-6


def probe_model(model: torch.nn.Module) -> dict[str, Any]:
    """Read the geometry the constructed DiT knows.

    Args:
        model: A live ``DreamZeroPolicy``. Its DiT is reached at
            ``action_head.model``
            (``modeling_dreamzero.py:167-173``,
            ``modules/action_head_tf.py:277-280``).

    Returns:
        One entry per :data:`_PROBE_FIELDS`, plus ``num_embodiments``
        read off the real parameter shape rather than off
        ``max_num_embodiments``: the module stores the *argument*
        (32) but builds its three category-specific submodules with a
        hardcoded 1 (``causal_wan_model.py:2292-2312``), and it is
        the shape the plugin has to agree with.

    Raises:
        ExportError: The model does not expose the DreamZero DiT.
    """
    dit = getattr(getattr(model, "action_head", None), "model", None)
    if dit is None:
        raise ExportError(
            f"{type(model).__name__} has no action_head.model; "
            f"export expects a DreamZeroPolicy"
        )
    missing = [name for name in _PROBE_FIELDS if not hasattr(dit, name)]
    if missing:
        raise ExportError(
            f"{type(dit).__name__} is missing {missing}; it does not "
            f"look like a CausalWanModel"
        )
    probe: dict[str, Any] = {
        name: getattr(dit, name) for name in _PROBE_FIELDS
    }
    probe["num_embodiments"] = int(
        dit.state_encoder.layer1.W.shape[0]
    )
    return probe


def _geometry(model_cfg: Any) -> dict[str, Any]:
    """Derive the DiT geometry from the LoongForge config alone.

    ``DreamZeroConfig.__post_init__`` copies the backbone preset onto
    the config (``model_configuration_dreamzero.py:299-311``), so the
    preset table does not have to be read a second time here.

    Args:
        model_cfg: A ``DreamZeroConfig``.

    Returns:
        The same keys :func:`probe_model` returns.

    Raises:
        ExportError: A mapped config field is absent.
    """
    out: dict[str, Any] = {"eps": _DIT_EPS}
    for name, source in _CONFIG_SOURCES.items():
        if not hasattr(model_cfg, source):
            raise ExportError(
                f"{type(model_cfg).__name__} has no {source!r}, needed "
                f"for the plugin's dit.{name}"
            )
        out[name] = getattr(model_cfg, source)
    return out


def _check_probe(geometry: Mapping[str, Any],
                 probe: Mapping[str, Any]) -> None:
    """Refuse a config that disagrees with the live model.

    Args:
        geometry: Values derived from the config.
        probe: Values read off the live DiT.

    Raises:
        ExportError: Any shared field differs. Exporting the config
            value while shipping the probed weights is the one failure
            the plugin cannot detect: ``from_dict`` would accept the
            config and the loader would accept the weights, and only
            the numbers would be wrong.
    """
    bad = sorted(
        name for name in _PROBE_FIELDS
        if name in probe and probe[name] != geometry[name]
    )
    if bad:
        detail = ", ".join(
            f"{name}: config={geometry[name]!r} model={probe[name]!r}"
            for name in bad
        )
        raise ExportError(
            f"config and live model disagree on {bad}; {detail}"
        )


def translate_config(model_cfg: Any, *,
                     probe: Mapping[str, Any] | None = None,
                     ) -> dict[str, object]:
    """Render the plugin's ``config.json`` payload.

    Args:
        model_cfg: A ``DreamZeroConfig``.
        probe: Optional :func:`probe_model` output, cross-checked
            against the config.

    Returns:
        The validated, JSON-ready payload.

    Raises:
        ExportError: The backbone variant is unsupported, the text
            length disagrees with the frozen text encoder, or the
            config and the probe disagree.
        ConfigBridgeError: A declared field is missing or an
            undeclared one was written (``config_bridge.py``).
    """
    variant = getattr(model_cfg, "backbone_variant", "")
    if variant not in SUPPORTED_BACKBONE_VARIANTS:
        raise ExportError(
            f"backbone_variant {variant!r} cannot be exported: the "
            f"plugin's DreamZeroVAEConfig accepts Wan2.1 only "
            f"(z_dim=16), so only "
            f"{sorted(SUPPORTED_BACKBONE_VARIANTS)} map across"
        )
    geometry = _geometry(model_cfg)
    if probe is not None:
        _check_probe(geometry, probe)
    text_len = int(getattr(model_cfg, "text_len"))
    if text_len != _TEXT_ENCODER_GEOMETRY["max_length"]:
        raise ExportError(
            f"text_len {text_len} differs from the frozen text "
            f"encoder's max_length "
            f"{_TEXT_ENCODER_GEOMETRY['max_length']}; the plugin "
            f"builds one T5 from text_encoder.max_length and feeds "
            f"the DiT from it"
        )
    return finalise_config(CONFIG_SCHEMA,
                           _payload(model_cfg, geometry))


def _payload(model_cfg: Any,
             geometry: Mapping[str, Any]) -> dict[str, object]:
    """Assemble the flat-plus-nested ``config.json`` body.

    Two mappings here are not the identity and are the reason this is
    a function rather than a dict comprehension:

    * ``action_horizon`` comes from LoongForge's
      ``num_action_per_block`` (32), **not** from its own
      ``action_horizon`` (48). ``DreamZeroConfig.__post_init__``
      requires ``action_horizon == dit.num_action_per_block``, and the
      48 is the policy's output chunk, not the DiT's block size.
    * ``hidden_size`` is the DiT's state/action MLP width
      (``dit_action_state_hidden_size``, 1024), not
      ``action_head_hidden_size`` (64). The plugin spends it on
      ``DreamZeroCategorySpecificMLP``
      (``modeling_dreamzero.py:960-1030``).

    Args:
        model_cfg: A ``DreamZeroConfig``.
        geometry: :func:`_geometry` output.

    Returns:
        The payload, before validation.
    """
    dit = {
        "dim": geometry["dim"],
        "eps": geometry["eps"],
        "ffn_dim": geometry["ffn_dim"],
        "frame_seqlen": geometry["frame_seqlen"],
        "freq_dim": geometry["freq_dim"],
        "in_dim": geometry["in_dim"],
        "max_chunk_size": model_cfg.max_chunk_size,
        "num_action_per_block": geometry["num_action_per_block"],
        "num_frame_per_block": geometry["num_frame_per_block"],
        "num_heads": geometry["num_heads"],
        "num_layers": geometry["num_layers"],
        "num_state_per_block": geometry["num_state_per_block"],
        "out_dim": geometry["out_dim"],
    }
    return {
        "dit": dit,
        "text_encoder": dict(_TEXT_ENCODER_GEOMETRY),
        "image_encoder": dict(_IMAGE_ENCODER_GEOMETRY),
        "vae": dict(_VAE_GEOMETRY),
        "action_dim": geometry["action_dim"],
        "action_horizon": geometry["num_action_per_block"],
        "max_action_dim": model_cfg.max_action_dim,
        "max_state_dim": geometry["max_state_dim"],
        "hidden_size": geometry["hidden_size"],
        "num_frames": model_cfg.num_frames,
        "num_inference_timesteps": model_cfg.num_inference_timesteps,
        "num_timestep_buckets": model_cfg.num_timestep_buckets,
        "num_frame_per_block": geometry["num_frame_per_block"],
        "tiled": bool(model_cfg.vae_tiled),
        "tile_size_height": model_cfg.vae_tile_size_height,
        "tile_size_width": model_cfg.vae_tile_size_width,
        "tile_stride_height": model_cfg.vae_tile_stride_height,
        "tile_stride_width": model_cfg.vae_tile_stride_width,
        **_INFERENCE_DEFAULTS,
    }


def _variant(model_cfg: Any,
             probe: Mapping[str, Any]) -> dict[str, object]:
    """Fingerprint the fields the two sides default differently.

    §2.3: a field both sides declare, with different defaults, is the
    one way a correct ``config.json`` still produces wrong numbers.
    All five entries below are such fields.

    Args:
        model_cfg: A ``DreamZeroConfig``.
        probe: :func:`probe_model` output.

    Returns:
        Scalars only, as :class:`~..export.NativeManifest` requires.
    """
    return {
        "backbone_variant": model_cfg.backbone_variant,
        "dit_eps": _DIT_EPS,
        "num_embodiments": probe["num_embodiments"],
        "image_encoder_feature_layer":
            _IMAGE_ENCODER_GEOMETRY["feature_layer"],
        "loongforge_action_horizon": model_cfg.action_horizon,
    }


def export(model: torch.nn.Module,
           model_cfg: Any,
           dest: str,
           *,
           variant_probe: Mapping[str, Any] | None = None,
           upstream: tuple[Upstream, ...] = (),
           max_shard_bytes: int | None = None,
           overwrite: bool = False) -> ExportReport:
    """Write a native-tier bundle for a live DreamZero policy.

    Args:
        model: A live ``DreamZeroPolicy``.
        model_cfg: Its ``DreamZeroConfig``.
        dest: Bundle directory. Created if absent.
        variant_probe: Override for :func:`probe_model`, so that a
            test can exercise the config path without a real DiT.
        upstream: Pinned pretrained artefacts (G8). Empty is allowed
            but recorded as an unmet gate rather than ignored,
            because the revisions are not knowable from LoongForge's
            path-only config.
        max_shard_bytes: Soft per-shard cap; ``None`` writes one file.
        overwrite: Whether an existing bundle may be replaced.

    Returns:
        The :class:`~..export.ExportReport` that was written.

    Raises:
        ExportError: A required gate failed. The report is not
            written in that case, by design: a bundle on disk without
            a passing report would be indistinguishable from a good
            one at load time.
    """
    probe = (dict(variant_probe) if variant_probe is not None
             else probe_model(model))
    transform = key_map()
    report = ExportReport(model_type=MODEL_TYPE, plugin=PLUGIN_NAME,
                          key_ops=transform.describe())
    report.require(
        "G3_backbone_variant_supported",
        model_cfg.backbone_variant in SUPPORTED_BACKBONE_VARIANTS,
        f"backbone_variant={model_cfg.backbone_variant!r}, supported="
        f"{sorted(SUPPORTED_BACKBONE_VARIANTS)}",
    )
    report.require(
        "single_embodiment",
        probe.get("num_embodiments") == 1,
        f"the plugin hardcodes num_embodiments=1; this model holds "
        f"{probe.get('num_embodiments')!r}",
    )
    config = translate_config(model_cfg, probe=probe)
    report.config_fields = len(config)
    result = transform.apply_state_dict(model.state_dict())
    report.source_keys = len(result.tensors) + len(result.dropped)
    report.dropped_keys = result.dropped
    report.fanned_out = dict(result.fanned_out)
    report.record(
        "G3_no_key_transform",
        not result.dropped and not result.renamed
        and not result.fanned_out,
        "DreamZero's live names are already canonical; the plugin's "
        "four remaps each select their own subtree",
    )
    report.advise("G8_upstream_pinned", bool(upstream),
                  f"{len(upstream)} artefact(s) pinned")
    variant = _variant(model_cfg, probe)
    report.variant = variant
    manifest = NativeManifest(
        model_type=MODEL_TYPE, plugin=PLUGIN_NAME, variant=variant,
        upstream=tuple(upstream),
        entry_args={"weight_strict": True,
                    "dtype": str(model_cfg.model_dtype)},
    )
    write_bundle(dest, tensors=result.tensors, manifest=manifest,
                 capabilities=Capabilities(level="EXACT"),
                 config=config, report=report,
                 max_shard_bytes=max_shard_bytes, overwrite=overwrite)
    return report
