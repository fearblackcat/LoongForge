# Exporting a PhyAI Bundle

A *bundle* is a self-contained directory that PhyAI can load for
inference without importing any LoongForge code. Files on disk are the
only interface between the two repositories: PhyAI never imports
`loongforge`, and `loongforge` never imports `phyai`.

This page covers the **native tier**, in which the bundle is consumed by
the model's own PhyAI plugin and the bundle directory doubles as the
plugin's `checkpoint_dir`.

## Supported models

| `model_type` | Exporter | Status |
| --- | --- | --- |
| `pi05` | `loongforge.embodied.model.pi05.export_pi05` | implemented |
| `dreamzero` | `loongforge.embodied.model.dreamzero.export_dreamzero` | implemented |
| `gr00t_n1_7` | — | not written |
| `lingbot_vla_v2` | — | not written |
| `cosmos3` | — | not written |

There is no CLI entry point. An export is a Python call, normally made
from a short script that first builds the model the same way training
does.

## Exporting pi0.5

```python
from loongforge.embodied.model.pi05 import export_pi05
from loongforge.embodied.model.export import Upstream

report = export_pi05.export(
    model,                       # a live PI05Policy
    model_cfg,                   # the config the model was built from
    "/abs/path/to/bundle",
    upstream=(Upstream(kind="huggingface",
                       name="physical-intelligence/pi05-base",
                       revision="<commit sha>"),),
    max_shard_bytes=5 * 2**30,
    overwrite=True,
)
print(report.written_keys, report.failed, report.unmet)
```

`export` returns an `ExportReport` and writes the same data to
`export_report.json` inside the bundle.

## Exporting DreamZero

```python
from loongforge.embodied.model.dreamzero import export_dreamzero

report = export_dreamzero.export(model, model_cfg, "/abs/path/to/bundle",
                                 overwrite=True)
```

`export` probes the live model itself. `probe_model` reaches the DiT
through `model.action_head.model` and reads the embodiment count off
`dit.state_encoder.layer1.W.shape[0]`, so an export needs the assembled
model rather than the config alone.

## Arguments common to both exporters

Both exporters expose the same four public names — `key_map()`,
`probe_model(model)`, `translate_config(model_cfg, *, probe=None)`, and
`export(...)` — with this signature:

```python
export(model, model_cfg, dest, *, variant_probe=None, upstream=(),
       max_shard_bytes=None, overwrite=False) -> ExportReport
```

| Argument | Meaning |
| --- | --- |
| `dest` | Target directory. Resolved to an absolute path, because that is what PhyAI's `checkpoint_dir` has to be. |
| `variant_probe` | Override for `probe_model`, so that a test can exercise the config path without a real backbone. Leave it unset in a real export: `export` probes the live model itself and verifies the geometry against the supported variants. |
| `upstream` | `Upstream(kind, name, revision)` records of the artefacts this checkpoint descends from. `kind` is `"huggingface"` or `"pip"`; `revision` must be a commit sha or a released version, never a moving ref such as `main`. |
| `max_shard_bytes` | Soft per-shard cap. `None` writes a single `model.safetensors`, and so does any cap the total fits under. Pass `5 * 2**30` to reproduce the HuggingFace 5 GiB convention; pi0.5 is about 3.3 B parameters, so bf16 weights land in two shards. |
| `overwrite` | Required to replace a directory that already holds bundle files. Without it the export refuses rather than mixing two exports in one directory. |

Leaving `upstream` empty is allowed. It is recorded as an unmet advisory
gate, not treated as an error — see the next two sections.

## What ends up on disk

```text
bundle/
  arch_manifest.json     # model_type, plugin, variant, upstream, entry_args, weight_files
  capabilities.json      # support level and any caveats
  export_report.json     # the export's own self-check
  config.json            # native tier only: the payload PhyAI's config dataclass reads
  model.safetensors      # or model-00001-of-0000N.safetensors plus
                         # model.safetensors.index.json when sharded
```

`config.json` is written for the native tier only, and it is mandatory
there: the plugin resolves its config from `checkpoint_dir/config.json`.
Weights are always `safetensors`; `torch.save` would carry Python types
across the repository boundary, which is what the boundary exists to
prevent.

## Reading `export_report.json`

The report is how an export says what it checked. Four fields matter:

| Field | Meaning |
| --- | --- |
| `passed` | True when no blocking gate failed. |
| `failed` | Ids of **blocking** gates that did not hold. A non-empty list means no bundle was written at all. |
| `unmet` | Ids of **advisory** gates that did not hold. The bundle is complete and loadable; something a reader should know about is recorded. |
| `checks[].blocking` | Per-gate severity, so a reader can tell the two apart without knowing the gate ids. |

A failed blocking gate leaves nothing behind — `write_bundle` refuses
before creating the directory, so there is no half-written bundle to
clean up or mistake for a good one.

Gates currently emitted:

| Gate | Severity | Model |
| --- | --- | --- |
| `G3_backbone_variant_supported` | blocking, raises at the call site | dreamzero |
| `single_embodiment` | blocking, raises at the call site | dreamzero |
| `G3_no_key_transform` | blocking, deferred to the write | dreamzero |
| `no_module_prefix_survives` | blocking, raises at the call site | pi05 |
| `tied_embedding_survives` | blocking, raises at the call site | pi05 |
| `canonical_drops_applied` | blocking, raises at the call site | pi05 |
| `alias_branch_dropped` | blocking, deferred to the write | pi05 |
| `G8_upstream_pinned` | advisory | both |

## Loading the bundle from PhyAI

Point the plugin's `checkpoint_dir` at the bundle directory and leave
the rest empty:

```yaml
plugin: pi05
plugin_args:
  checkpoint_dir: /abs/path/to/bundle   # must be absolute
  # config: omitted      -> resolved from checkpoint_dir/config.json
  # weight_remap: omitted -> key names were normalised at export time
```

Three constraints come from the engine side:

* `build_engine` refuses config fields holding a `Callable` or a
  `Tensor` (`engine_builder.py:116-119`), so everything in
  `plugin_args` has to be plain data.
* `checkpoint_dir` is on the engine's relative-path whitelist but
  `bundle_dir` is not (`engine_builder.py:202-207`). Use an absolute
  path, which is also what `export` returns.
* Under a managed worker, the `build_engine` call must sit inside
  `if __name__ == "__main__":` (`engine_builder.py:278`).

The manifest's `entry_args` carries the loader flags the export vouches
for: `weight_strict: true` in both models, plus
`vision_params_dtype: "float32"` for pi0.5 and `dtype` for DreamZero.

## Per-model constraints

**DreamZero only supports `backbone_variant == "wan21_14b"`.** PhyAI's
`DreamZeroVAEConfig.__post_init__` requires `z_dim == 16`, while the
LoongForge default `wan22_5b` has `z_dim = 48`. The common case is
therefore the unsupported one, and the export raises instead of writing
a bundle that would load with the wrong latent geometry.

**DreamZero requires a single embodiment.** The probed
`num_embodiments` must be 1; PhyAI's config has no place to put more.

**pi0.5 pins its geometry.** Any probed value departing from
`export_pi05.DEFAULT_GEOMETRY` is refused, because the PhyAI plugin was
written against `pi05_base`'s dimensions. `translate_config` also
refuses `action_dim > max_action_dim`.

**pi0.5 drops keys on purpose.** `PI05Policy` exposes every parameter
twice, once under each of its two module aliases, and the plugin's own
filter refuses two further keys. A minimal export therefore reports more
dropped keys than written ones, which is expected rather than a warning
sign.

## Why `config.json` is written field by field

`PretrainedConfig.from_dict` filters its input down to the field names
the dataclass declares, and every field carries a default
(`models/configuration.py:97-134`). A geometry key that was forgotten or
misspelled on the export side therefore does not fail the load — it
silently takes the default, and the policy runs on numbers nobody chose.

The exporters answer that by writing every field explicitly from an
inlined copy of the names PhyAI declares (`CONFIG_SCHEMA` in each
exporter). That only works while the copy stays in sync, which is what
`tests/embodied/unit/test_arch_export.py` checks: it reads the nine
upstream dataclasses with `ast` — not by importing them, which the
boundary rule forbids — and compares the field lists in both
directions.

Run it against a PhyAI checkout:

```bash
cd LoongForge
PHYAI_SOURCE_ROOT=/path/to/phyai/phyai/src \
  PYTHONPATH=. python -m pytest tests/embodied/unit/test_arch_export.py -q
```

Without `$PHYAI_SOURCE_ROOT` the test looks for a sibling
`phyai/phyai/src` checkout, and skips at module level when there is
none. A skip means the comparison did not run, so treat it as a missing
check rather than a pass.

## Writing a bundle by hand

An exporter for a new model composes the same four pieces the two
existing ones use, all from `loongforge.embodied.model.export`:

```python
from loongforge.embodied.model.export import (Capabilities, NativeManifest,
                                              ExportReport, write_bundle)

report = ExportReport(model_type="mymodel", plugin="mymodel")
report.require("gate_that_must_hold", condition, "evidence")   # raises here
report.record("gate_checked_later", condition, "evidence")     # blocks the write
report.advise("gate_worth_noting", condition, "evidence")      # never blocks

write_bundle(dest, tensors=tensors,
             manifest=NativeManifest(model_type="mymodel", plugin="mymodel"),
             capabilities=Capabilities(level="EXACT"),
             config=config, report=report)
```

`Capabilities.level` is one of `EXACT`, `DEGRADED`, `UNSUPPORTED`; any
level other than `EXACT` must come with at least one note explaining
what is degraded.

Key renaming belongs in that exporter's `key_map()`, built from the
declarative operators in `export/keymap.py` (`StripPrefix`,
`RenameSegment`, `Drop`, `Fanout`), so that the renaming is data the
report can describe rather than code nobody can audit.
