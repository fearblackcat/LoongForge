# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared machinery for exporting a legacy model as a PhyAI bundle.

Design: ``docs/design/legacy_phyai_bundle.zh-CN.md``. The bundle is
the only interface between this repository and PhyAI (J1); nothing
here imports PhyAI, and nothing in PhyAI imports this.

A per-model exporter lives next to its ``modeling_<name>.py``, not
here, so that a change to the model and a change to its export rules
land in the same directory and the same review. Each one exposes
three functions (§3.2)::

    def key_map() -> KeyMap: ...
    def translate_config(model_cfg) -> dict: ...
    def export(model, model_cfg, dest, *, variant_probe=None): ...
"""

from __future__ import annotations

from .bundle import (CONFIG_FILENAME, WEIGHTS_FILENAME,
                     WEIGHTS_INDEX_FILENAME, write_bundle)
from .config_bridge import (ConfigBridgeError, ConfigSchema,
                            finalise_config, validate_config)
from .keymap import (Drop, Fanout, KeyMap, KeyMapError, KeyMapResult,
                     RenameSegment, StripPrefix, identity)
from .manifest import (BACKENDS, CAPABILITIES_FILENAME,
                       MANIFEST_FILENAME, SCHEMA_VERSION,
                       SUPPORTED_SCHEMA_VERSIONS, Capabilities,
                       ManifestError, NativeManifest, Upstream,
                       write_json)
from .report import (REPORT_FILENAME, Check, ExportError, ExportReport)

__all__ = [
    "BACKENDS", "CAPABILITIES_FILENAME", "CONFIG_FILENAME",
    "Capabilities", "Check", "ConfigBridgeError", "ConfigSchema",
    "Drop", "ExportError", "ExportReport", "Fanout", "KeyMap",
    "KeyMapError", "KeyMapResult", "MANIFEST_FILENAME",
    "ManifestError", "NativeManifest", "REPORT_FILENAME",
    "RenameSegment", "SCHEMA_VERSION", "SUPPORTED_SCHEMA_VERSIONS",
    "StripPrefix", "Upstream", "WEIGHTS_FILENAME",
    "WEIGHTS_INDEX_FILENAME", "finalise_config", "identity",
    "validate_config", "write_bundle", "write_json",
]
