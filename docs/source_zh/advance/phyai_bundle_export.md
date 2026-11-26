# 导出 PhyAI Bundle

**bundle** 是一个自包含的目录，PhyAI 可以直接加载它做推理，不需要 import 任何
LoongForge 代码。两个仓库之间唯一的接口就是磁盘上的文件：PhyAI 不 import
`loongforge`，`loongforge` 也不 import `phyai`。

本文覆盖 **native 档**：bundle 由模型自己的 PhyAI 插件消费，bundle 目录同时就是
插件的 `checkpoint_dir`。

## 已支持的模型

| `model_type` | 导出器 | 状态 |
| --- | --- | --- |
| `pi05` | `loongforge.embodied.model.pi05.export_pi05` | 已实现 |
| `dreamzero` | `loongforge.embodied.model.dreamzero.export_dreamzero` | 已实现 |
| `gr00t_n1_7` | — | 未实现 |
| `lingbot_vla_v2` | — | 未实现 |
| `cosmos3` | — | 未实现 |

没有 CLI 入口。导出是一次 Python 调用，通常写在一个短脚本里，脚本先按训练时
同样的方式把模型搭起来。

## 导出 pi0.5

```python
from loongforge.embodied.model.pi05 import export_pi05
from loongforge.embodied.model.export import Upstream

report = export_pi05.export(
    model,                       # 活的 PI05Policy
    model_cfg,                   # 搭这个模型用的 config
    "/abs/path/to/bundle",
    upstream=(Upstream(kind="huggingface",
                       name="physical-intelligence/pi05-base",
                       revision="<commit sha>"),),
    max_shard_bytes=5 * 2**30,
    overwrite=True,
)
print(report.written_keys, report.failed, report.unmet)
```

`export` 返回一个 `ExportReport`，同一份数据也写进 bundle 里的
`export_report.json`。

## 导出 DreamZero

```python
from loongforge.embodied.model.dreamzero import export_dreamzero

report = export_dreamzero.export(model, model_cfg, "/abs/path/to/bundle",
                                 overwrite=True)
```

`export` 自己会去探活模型。`probe_model` 通过 `model.action_head.model` 拿到
DiT，并从 `dit.state_encoder.layer1.W.shape[0]` 读出 embodiment 数量，所以导出
需要已经装配好的模型，光有 config 不行。

## 两个导出器共有的参数

两个导出器暴露同样四个公开名字 —— `key_map()`、`probe_model(model)`、
`translate_config(model_cfg, *, probe=None)`、`export(...)`，签名如下：

```python
export(model, model_cfg, dest, *, variant_probe=None, upstream=(),
       max_shard_bytes=None, overwrite=False) -> ExportReport
```

| 参数 | 含义 |
| --- | --- |
| `dest` | 目标目录。会被解析成绝对路径，因为 PhyAI 的 `checkpoint_dir` 必须是绝对路径。 |
| `variant_probe` | `probe_model` 的覆盖入口，便于测试在没有真主干的情况下走通 config 路径。真实导出不要传：`export` 会自己探活模型，并把几何与支持的变体核对一遍。 |
| `upstream` | 本 checkpoint 的上游制品记录 `Upstream(kind, name, revision)`。`kind` 取 `"huggingface"` 或 `"pip"`；`revision` 必须是 commit sha 或已发布版本号，不能是 `main` 这类会动的引用。 |
| `max_shard_bytes` | 每个分片的软上限。`None` 表示只写一个 `model.safetensors`；总量没超过上限时同样只写一个。传 `5 * 2**30` 即复刻 HuggingFace 的 5 GiB 惯例；pi0.5 约 3.3 B 参数，bf16 权重会落成两个分片。 |
| `overwrite` | 要替换一个已经有 bundle 文件的目录时必须传。不传则直接报错，而不是把两次导出混在一个目录里。 |

`upstream` 留空是允许的。它会被记成一条**未满足的 advisory 闸门**，不算错误 ——
见下两节。

## 磁盘上落下什么

```text
bundle/
  arch_manifest.json     # model_type、plugin、variant、upstream、entry_args、weight_files
  capabilities.json      # 支持等级与注意事项
  export_report.json     # 本次导出的自检结果
  config.json            # 仅 native 档：PhyAI 的 config dataclass 读的那份数据
  model.safetensors      # 分片时改为 model-00001-of-0000N.safetensors
                         # 加一份 model.safetensors.index.json
```

`config.json` 只有 native 档写，而且在 native 档是必需的：插件从
`checkpoint_dir/config.json` 解析自己的 config。权重一律用 `safetensors`；
`torch.save` 会把 Python 类型带过仓库边界，而这正是设边界要防的事。

## 怎么读 `export_report.json`

报告是导出过程对「我核对了什么」的交代。四个字段要紧：

| 字段 | 含义 |
| --- | --- |
| `passed` | 没有 blocking 闸门失败时为 true。 |
| `failed` | 失败的 **blocking** 闸门 id。非空就意味着**一个 bundle 都没写出来**。 |
| `unmet` | 未满足的 **advisory** 闸门 id。bundle 是完整可加载的，只是记下了一件读者应当知道的事。 |
| `checks[].blocking` | 每条闸门的严重级别，读者不必记住闸门 id 也能区分两类。 |

blocking 闸门失败时什么都不留 —— `write_bundle` 在建目录之前就拒绝，所以不会有
写了一半的 bundle 要你清理，也不会有人把它当成好的那份。

当前会发出的闸门：

| 闸门 | 级别 | 模型 |
| --- | --- | --- |
| `G3_backbone_variant_supported` | blocking，在调用处当场 raise | dreamzero |
| `single_embodiment` | blocking，在调用处当场 raise | dreamzero |
| `G3_no_key_transform` | blocking，推迟到写入时拦 | dreamzero |
| `no_module_prefix_survives` | blocking，在调用处当场 raise | pi05 |
| `tied_embedding_survives` | blocking，在调用处当场 raise | pi05 |
| `canonical_drops_applied` | blocking，在调用处当场 raise | pi05 |
| `alias_branch_dropped` | blocking，推迟到写入时拦 | pi05 |
| `G8_upstream_pinned` | advisory | 两者 |

## 在 PhyAI 侧加载

把插件的 `checkpoint_dir` 指向 bundle 目录，其余留空：

```yaml
plugin: pi05
plugin_args:
  checkpoint_dir: /abs/path/to/bundle   # 必须绝对路径
  # config 不填       -> 插件按 checkpoint_dir/config.json 解析
  # weight_remap 不填 -> 键名已在导出期归一
```

引擎侧有三条硬约束：

* `build_engine` 拒绝取值为 `Callable` 或 `Tensor` 的 config 字段
  （`engine_builder.py:116-119`），所以 `plugin_args` 里只能放纯数据。
* `checkpoint_dir` 在引擎的相对路径白名单里，`bundle_dir` 不在
  （`engine_builder.py:202-207`）。用绝对路径，这也正是 `export` 的返回值。
* 在 managed worker 下，`build_engine` 调用必须写在
  `if __name__ == "__main__":` 里面（`engine_builder.py:278`）。

manifest 的 `entry_args` 带着导出方敢背书的加载开关：两个模型都是
`weight_strict: true`，pi0.5 另有 `vision_params_dtype: "float32"`，
DreamZero 另有 `dtype`。

## 各模型的限制

**DreamZero 只支持 `backbone_variant == "wan21_14b"`。** PhyAI 的
`DreamZeroVAEConfig.__post_init__` 要求 `z_dim == 16`，而 LoongForge 的默认
`wan22_5b` 是 `z_dim = 48`。也就是说常见那一档恰好是不支持的，导出期会直接报错，
而不是写出一个会以错误潜空间几何加载的 bundle。

**DreamZero 要求单 embodiment。** 探出来的 `num_embodiments` 必须是 1；
PhyAI 的 config 没有放多个的位置。

**pi0.5 钉死了几何。** 任何偏离 `export_pi05.DEFAULT_GEOMETRY` 的探测值都会被
拒绝，因为 PhyAI 插件是按 `pi05_base` 的尺寸写的。`translate_config` 另外会拒绝
`action_dim > max_action_dim`。

**pi0.5 是故意丢键的。** `PI05Policy` 把每个参数在两个模块别名下各暴露一次，
插件自己的过滤器还会再拒两个键。所以一次最小导出报出的 dropped 键比 written 键
更多是预期现象，不是异常信号。

## 为什么 `config.json` 要一个字段一个字段地写

`PretrainedConfig.from_dict` 会把输入过滤到 dataclass 声明的字段名，而每个字段
都有默认值（`models/configuration.py:97-134`）。于是导出侧漏写或拼错一个几何键
**不会让加载失败** —— 它会静默取默认值，策略就跑在一组没人选过的数上。

导出器的应对是：从 PhyAI 声明的字段名内联一份副本（各导出器里的
`CONFIG_SCHEMA`），然后把每个字段都显式写出来。这只在副本保持同步时成立，
而这正是 `tests/embodied/unit/test_arch_export.py` 要核对的事：它用 `ast` 读上游
那九个 dataclass —— 不是 import，因为边界规则禁止 —— 并做双向字段对照。

对着一份 PhyAI 源码跑：

```bash
cd LoongForge
PHYAI_SOURCE_ROOT=/path/to/phyai/phyai/src \
  PYTHONPATH=. python -m pytest tests/embodied/unit/test_arch_export.py -q
```

不设 `$PHYAI_SOURCE_ROOT` 时，测试会去找同级的 `phyai/phyai/src`，找不到就在
module 级 skip。**skip 意味着这项对照根本没跑**，应当当成缺一项检查，而不是
当成通过。

## 手写一个 bundle

新模型的导出器拼的是两个现有导出器用的同样四件东西，全部来自
`loongforge.embodied.model.export`：

```python
from loongforge.embodied.model.export import (Capabilities, NativeManifest,
                                              ExportReport, write_bundle)

report = ExportReport(model_type="mymodel", plugin="mymodel")
report.require("gate_that_must_hold", condition, "evidence")   # 当场 raise
report.record("gate_checked_later", condition, "evidence")     # 由写入时拦
report.advise("gate_worth_noting", condition, "evidence")      # 永不拦

write_bundle(dest, tensors=tensors,
             manifest=NativeManifest(model_type="mymodel", plugin="mymodel"),
             capabilities=Capabilities(level="EXACT"),
             config=config, report=report)
```

`Capabilities.level` 取 `EXACT`、`DEGRADED`、`UNSUPPORTED` 之一；非 `EXACT` 的
等级必须至少带一条 note，说明降级在哪。

键名改写应当住在该导出器的 `key_map()` 里，由 `export/keymap.py` 的声明式算子
（`StripPrefix`、`RenameSegment`、`Drop`、`Fanout`）拼出来 —— 这样改写是报告能
描述的数据，而不是没人审得动的代码。
