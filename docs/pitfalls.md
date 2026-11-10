# 迁移踩坑记录（每条都带"症状 → 根因 → 守卫"）

只收录**在本仓复现过**的坑。上游 TileInfer 文档里的坑如果这里没有对应现象，就不写——
抄一份没验证过的清单，等于给自己埋一份假知识。

## 1. `null_block_id` 默认值是 0，于是第 0 行被静默跳过

**症状**：把 `line=0` 传给 conv 算子，torch 基线返回**全零向量**，状态也不变；内核返回正确值，
于是"内核 vs 基线"看起来像内核算错了。

**根因**：上游 `causal_conv1d_update` / `causal_conv1d_fn` 的 `null_block_id` 默认是
`NULL_BLOCK_ID = 0`，注释为 `if conv_states_input_coord == null_block_id: return`（l.136-139 /
l.814-817）。引擎里 0 号 cache 行是"这条序列没有 state 页"，跳过是对的；但在**算子库**这一层，
`line` 只是一个行号，把 0 当空页会把合法调用变成零输出。

**守卫**：`ops/gdn/__init__.py` 的 torch 基线分支显式传 `null_block_id=None`，并有
`tests/test_gdn_conv_device.py::test_conv_kernel_handles_row_zero` 钉住两侧都是真数字。

**同类第二坑**：不传 `cache_indices` 时预填形态回落到 `PAD_SLOT_ID = -1`，状态**不写**（输出对、
状态全零）。`tests/test_fla_ops.py` 里用显式行号，并注释了这一点。

## 2. TileLang：工厂函数里的局部量在 `prim_func` 注解里看不见

**症状**：`NameError: name 'C' is not defined`，只发生在编译期，源码读起来完全正常。

**根因**：`tilelang.compile` 会**重新解析** `prim_func` 的源码，注解按模块全局解析；
工厂函数里的 `C = channels` 不在全局里。`tilelang.jit` 装饰同一个工厂也会踩，因为它同样重解析。

**守卫**：沿用源仓范式——注解只用模块级名字，或在编译窗口内把 `C/W/L/taps/per/dtype`
临时塞进模块全局并在 `finally` 里还原（`kernels/gdn/conv_update_ascend950.py:build_conv_update_kernel`）。

## 3. `from __future__ import annotations` 会让 `chunk_wu` 内核挂掉

**症状**：编译期报注解不是字符串可解析的类型。

**根因**：`chunk_wu_ascend950.py` 的注解里用了运行时类型，未来注解把注解变成字符串后，
TileLang 的解析器拿不到类型对象。

**守卫**：该文件**禁止**加 `from __future__ import annotations`（文件头注释已写明）。
复制内核后若用格式化工具批量加，会静默破坏；`tests/test_import_hygiene.py` 覆盖不了，只能靠注释。

## 4. 模块级 `import tilelang` 会让整个包不可导入

**症状**：在没装 TileLang 的机器上 `import ascend_attention` 直接抛 `ModuleNotFoundError`；
在装了但没设备的机器上，`import torch` → `torch_npu` → `triton` 会去跑 `npu-smi` 并卡住。

**根因**：内核文件习惯性在模块顶部 `import tilelang.ascend.language as T`。

**守卫**：`kernels/**` 与 `backends/**` 的 TileLang 导入一律 `try/except` 守卫；
`tests/test_import_hygiene.py` 用 AST 扫描（含 `try` 体），核心模块保持严格零可选依赖。

## 5. 清单里 `dtypes` 为空 → 平台解析器直接拒收

**症状**：跨仓测试报 `ManifestError`，本地看着一切正常（`dumps()` 不校验语义）。

**根因**：`dtypes` 由选中实现的 `ShapeSpec` 聚合而来；某个 op 的实现没有声明 `shape` 时聚合为空，
而平台 `parse_manifest` 要求 `dtypes` / `kv_layouts` 非空。

**守卫**：每个 `ImplSpec` 都要有 `ShapeSpec`（哪怕只是逻辑形状），
`tests/test_manifest.py::test_platform_parser_accepts_our_manifest` 直接调平台的解析器。

## 6. `ascend_attention.manifest` 这个名字被函数占用

**症状**：`from ascend_attention import manifest as mf` 拿到的是函数，`mf.dumps()` 报
`AttributeError`。

**根因**：`__init__` 里 `from .manifest import manifest`，同名函数覆盖了子模块绑定。

**守卫**：公开函数改名 `capability_manifest`；包内测试一律 `from ascend_attention.manifest import ...`。

## 7. split-KV 默认不可用，而且扫形不收敛

**症状**：`plan(..., tokens_per_split=128)` 抛 `NotImplementedError: split-KV is opt-in: set
TILEINFER_ALLOW_SPLIT_KV=1`。

**根因**：源仓把 `max_splits`/`kv_tile_pages` 的路径标成"多形不稳"，用环境变量闸住。

**现状**：本仓尚未评估 split-KV。一次 `B∈{1,8} × S∈{2048,8192} × tps∈{0,256}` 的扫描在 25 分钟后
仍未收敛（每个变体都要重新编译），已中止。**不要把"batch 1 上 3×"当成已核实证据**。

## 8. 默认 shell 里 `/dev` 被覆盖，设备"不存在"

**症状**：`torch.npu.is_available()` 为 `False`、`/dev/davinci0` 不存在、`import tilelang` 卡 100 s。

**根因**：受限 shell 在 `/dev` 上叠了一层 tmpfs（14 个条目）；`triton/tools/get_ascend_devices.py`
会 `subprocess.run(["npu-smi","info"], timeout=100)`，在那种 shell 里等满超时。

**守卫**：设备相关命令一律在 full-access 下跑，先做 canary（`npu-smi info` →
`torch.npu.is_available()` → 一个小内核）。

## 9. bf16 大词表里 top-k 必有并列，"下标集合相同"是错的判据

**症状**：B=8 / V=152064 / k=20 时，vendor `deep_select.topk` 与 `torch.topk` 的**下标集合不同**，
但把两边的 value 排序后逐位相同（max_abs_diff = 0.0）。B=1 或 V=1024 时集合完全一致。

**根因**：bf16 只有 8 位尾数，15 万个采样必然碰撞；top-k 在并列处取哪个下标是实现自由。阈值式
top-k（保留所有等于阈值的 token）本来就**比排序式宽**。

**守卫**：判据用"选中的 value 排序后逐位相等 + 数量 == k"，不用下标集合；这条写进了
`tests/test_sampling_vendor_device.py` 的 docstring，因为第一次运行就是被它误报的。

## 10. 分派顺序会决定"性能问题"的答案

**症状**：`sampling_topk_threshold` 明明有 3.84× 的 vendor 实现且已启用，`plan()` 却返回 torch 基线。

**根因**：`select()` 按注册顺序取第一个合格实现，基线先注册就先返回。

**守卫**：候选按 `measured_ratio` 降序、torch 基线排最后（`dispatch.select` 里的 `ordered`）。
注册顺序只影响同比值候选之间的次序，不影响"选更快的那条"。

## 11. 比值必须有比较集，否则是噪声

**症状**：同一个 conv 内核，源仓记 42.5×、本仓记 14.15×；`chunk_wu` 源仓 45.6×/1.91×/1.04×，
本仓 14.55×。

**根因**：形状、比较对象（孤立核 / 链 / 整步）、缓存热度、绝对速度窗口都不同。

**守卫**：`bench/results/*.json` 必须带 `command`/`host`/`date`/`pairs`/`ratio_note`，
其中 `ratio_note` 写明**对比集**；`ImplSpec.ratio_note` 抄同一个句子。没有对比集的比值不进账本。
