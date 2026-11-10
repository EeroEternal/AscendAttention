# 完成度审计（D1–D8 + 各批验收 + §7 机制）

这份文件不是总结，是**对账**：计划文档里每一条验收，要么给出证据（测试名 / 产物路径 / 数字），
要么明写"没做"和为什么。任何一条被写成"已完成"却没有可点的证据，都算这份审计失败。

## D1–D8：决策与落地

| # | 决策 | 落地情况 | 证据 |
|---|---|---|---|
| D1 | 包名 `ascend_attention` | ✅ | `src/ascend_attention/`、`pyproject.toml`、`tests/test_import_hygiene.py` |
| D2 | per-op plan | ✅ | `plan()` 一次只产出一个 `OpPlan`；无 per-stage 聚合 |
| D3 | 状态由 caller 传入 | ✅ | `k_cache`/`v_cache` 由调用方持有并被 plan 捕获；库内无 `self._state` |
| D4 | 布局=枚举集 + `needs_pack` | ✅ | `kv_layouts`（ND/NZ）+ `layout_notes`；NZ 声明"可 pack，内核不直吃" |
| D5 | `graph` 三档 | ✅ | `spec.GRAPHS`；三档各有实证：`capture_safe` = decode 内核（guard 0 回读 + `NPUGraph` capture/replay 逐位相同），`eager_only` = conv 基线（guard 抓到 `_to_copy` 回读），`unknown` = 910B fork（不宣称） |
| D6 | 只给源码 + `prewarm()` 纯函数 | ✅ | `ascend_attention.prewarm()`（本轮新增，去重排序、不编译）；无预编译产物 |
| D7 | 只允许 `ASCEND_ATTN_*` | ✅（前缀统一 + 单一入口） | 内核里现已**零** `os.environ`：7 处读取全部走 `utils.kernel_flag()`（唯一入口、只读 `ASCEND_ATTN_*`、首次使用打一条 warning），flag 名字由 AST ratchet 钉死。**仍待办**：把它们变成显式编译规格参数（`docs/completion-audit.md` 的缺口 6） |
| D8 | 负结果归档 | ✅ | `docs/archive-moe.md`、`docs/archive-sampling.md`、`docs/archive-attention.md`、`docs/status.md` |

## P0 验收

| 条款 | 状态 | 证据 |
|---|---|---|
| 无 torch_npu/tilelang 也能 import | ✅（torch_npu 例外，已说明） | `test_import_hygiene.py::test_import_pulls_no_engine_and_no_accelerator`（子进程断言 sys.modules 无 vllm/vllm_ascend/tilelang） |
| `manifest()` 可 json.dumps，`supports()` 无需张量 | ✅ | `tests/test_manifest.py`、`tests/test_spec.py` |
| `plan/run` 在 reference 上跑通 decode 与 prefill 全分支 + fp64 oracle（尾页 poison 1e4） | ✅ | `tests/test_decode_fp64.py`（**独立 fp64 定义式**，不共用实现里的页展开代码，1e-6 + 反例）、`tests/test_attention_api.py`（decode 对 reference）、`tests/test_prefill_api.py`（prefill/append + fp64 oracle）、`tests/test_acceptance_mechanisms.py::test_partial_last_page_padding_does_not_change_the_result`（1e4 poison 逐位不变） |
| `capture_guard()` 抓得住 `torch.empty` | ✅ 本轮补齐 | `tests/test_capture_guard.py`（8 项，含 device 侧 `eager_only` 实证） |
| 零 `import vllm`；单文件与整套都绿 | ✅ | 360 tests；`test_import_hygiene` 走 AST 全包扫描 |

## P1 验收

| 条款 | 状态 | 证据 / 缺口 |
|---|---|---|
| decode 设备对数（TOL 3e-2） | ✅ | `tests/test_decode_device.py`（3 项，Ascend950PR_9579） |
| `capture_guard` 在 decode `run` 上零违规 | ✅ 本轮 | `tests/test_decode_capture_device.py::test_decode_run_has_no_host_reads`（0 syncs / 1 alloc）+ **真图捕获**：`NPUGraph` capture+replay 与 eager **逐位相同**（0.0）。据此 `graph` 由 `unknown` 升为 `capture_safe` |
| manifest 两 impl 的 `measured_ratio` 与 `ratio_note` 齐全 | ✅（口径改为"清册 + 子集"） | 两个实现的 `graph` / `requires` / `note` 都在 `capabilities()` 清册里；`manifest.json` 只收可用子集，所以 decode 整体不在其中（`measured_ratio=None` → 四道门拒绝）。这不是漏写，见 `docs/contracts/exposure.md` 与 `tests/test_exposure.py`。`tilelang.decode950` 仍**没有本仓 op 级 A/B**，0.362× 那次是 vs vendor FIA |
| `supports()` 仅凭 shape 拒绝非法 head_dim/group | ✅ | `tests/test_spec.py`、`tests/test_prefill_api.py`（head_dim=48 被拒） |
| prefill 内核 | ⚠️ **内核在仓、编排未迁** | 声明 `tilelang.prefill950`（`enabled=False`、`ratio=None`、`note` 写明原因、`provenance` 指向 `paged_prefill_ascend950`），planner 明确 `NotImplementedError` 而不是给半条路。reference 语义已可用并有 fp64 oracle |

## P2 验收

| 条款 | 状态 | 证据 |
|---|---|---|
| chunk 链对 fp64 oracle 逐段过 | ✅ | `tests/test_fla_ops.py`（kkt / solve_tril / wu 各自的 fp64 定义式） |
| 四个"静默杀手"各一条测试 | ✅ 本轮补齐 | `tests/test_fla_contract.py`（`safe_exp` 单边钳位、`where` vs 乘法 NaN、两个三角、三处 bf16 截断、参数顺序） |
| 状态张量全部由 caller 传入 | ✅ | `chunk_gated_delta_rule(..., initial_state=)`；库内无模块级状态 |
| `conv_update` 的 `graph` 声明与实测一致 | ✅ | `test_capture_guard.py::test_eager_only_declaration_is_honest_for_the_conv_baseline`（设备侧捕获到 `aten::_to_copy` 回读） |
| delta_h / chunk_o 的独立 oracle | ✅ 本轮补齐 | `test_fla_contract.py::test_chunked_chain_matches_the_recurrent_form`（分块链 vs 逐 token 递归，含尾块，相对误差 <1e-5） |

## P3 验收

| 条款 | 状态 | 证据 |
|---|---|---|
| MoE manifest 至少一个 torch 实现跑通两种专家分布 | ✅ | `tests/test_moe_ops.py`（均衡 + 单专家热点 + 空专家） |
| slot 哨兵语义（`sorted_ids == rows_static` 表示空 tile）有契约测试 | ✅ | 同上，并给出"compact→slot 映射"与独立实现的逐项比对 |
| 未接入的高性能候选显示 `enabled_by_default=False` 且 note 可读 | ✅ | `grouped_gemm`/`down_prenz` 未注册（归档）；`moe_prenz_pack` 的 note 写明等价关系 |

## P4 验收

| 条款 | 状态 | 证据 / 缺口 |
|---|---|---|
| 三个实现同族共存 | ⚠️ **两个** | `sampling_topk_threshold`（torch + vendor）与 `sampling_greedy`；`fused_sample` **归档**，理由见 `docs/archive-sampling.md`（零消费者、无端到端数字、`vocab%128`、`ub_cap` 507035） |
| 按 `(vocab, batch, do_sample)` 分流且有测试 | ✅（`do_sample` + 形状/dtype 门） | `sampler_for()` + `tests/test_sampling_topk.py`（路由、dtype 门、k 越界） |
| 两个来源（op 级 / end-to-end）分开记 | ✅ | 采样 3.8432× 是 **op 级**且 note 明写口径；无 e2e 数字就写无 |
| 无 `deep_select` 的机器上该实现消失而不是报错 | ✅ 本轮补齐 | dispatch 新增 `requires` 探测（`find_spec`，缓存）；`test_vendor_impl_is_invisible_when_its_module_is_missing` |

## P5 验收

| 条款 | 状态 | 证据 / 缺口 |
|---|---|---|
| 发布 `manifest.json`（含比值与 note） | ✅ | `src/ascend_attention/manifest.json`（8 ops，schema 1）；**每个启用的实现都有已提交的账本文件**（8 个），由 `tests/test_manifest.py::test_every_manifest_entry_points_at_an_existing_bench_result` 机械校验 |
| 可选 `vllm_adapter` | ❌ **未做** | 与"全仓零 vllm 依赖"直接冲突，且需要 vLLM 的 ABI 名。若要做，只能放 `integrations/`（不进 wheel、不被 AST 测试扫描）并单独声明依赖；本轮选择不做并记录 |
| TileInfer 反向改造为依赖本仓 | ✅（有限但真实） | `tileinfer/vllm/ops/extern.py` + 2 行接线，默认 off；off/on 逐位相同（8 组 sha256 + 双方测试） |
| `TILEINFER_OPS_CANDIDATES=off/on` 数字不变 | ✅（算子级 + 测试级） | 见 `docs/reverse-integration.md` §3–§4；**服务级 A/B 明确不做**（平台语义） |
| manifest 驱动全部 15 个绑定（或 note 缺口） | ⚠️ 13 个绑定、8 覆盖、5 缺口 | `tests/test_binding_coverage.py` 机械校验；计划里的"15"实为 13，代码为准 |

## §7 机制表：哪些真的成了机制

| 坑 | 机制 | 状态 |
|---|---|---|
| 尾页 padding 未 mask | 1e4 poison + 逐位比较 | ✅ 本轮 |
| OOB gather 返回垃圾 | `RaggedMetadata.validate()` 在 plan 期校验 | ✅ 既有 |
| 子 32 字节描述符 507035 | `ShapeSpec` 的 `min_group`/`min_stride_bytes` | ❌ **未实现，且试过一版是错的**：先按"张量最内层 stride 字节数"实现，跑既有 decode 测试立刻发现 bf16 池的最内层 stride 本来就是 1 元素 = 2 B，等于把所有合法输入全拒了。这条坑的真实含义是**单次 GM→UB 搬运的连续字节数** < 32 B，不是张量 stride；已回退，不留下一个看着像机制的错门 |
| AIV 行号不可见 → causal 阈值错 | `padded_rows` 的虚行必须被掩盖 + 两侧实测 | ✅ 本轮：`tests/test_decode_device.py::test_gqa_group_beyond_the_row_tile_is_masked[3/40]`（29 个虚行 / 跨 32 边界 24 个虚行，均对 fp32 oracle <3e-2） |
| 三处 bf16 截断 | 契约测试断言 dtype | ✅ 本轮 |
| 两个三角 | 两条契约测试（对角掩码） | ✅ 本轮 |
| 编译占 GIL | `prewarm()` 纯函数 | ✅ 本轮 |
| 编译键漏维度 | `decode_spec_key` + 逐维扰动测试 | ✅ 本轮（并写明它只是**单 plan** 的身份，不是进程级缓存） |
| 池容量击穿 | `abi` 量化 capacity | ✅ 既有 |
| `pool_view` 丢 offset | offset 断言测试 | ✅ 本轮 |
| 元数据 dtype 跨实现不一致 | ABI 把 dtype 写成数据 | ⚠️ 部分（int32 统一，fork 变体未并测） |
| `__future__` 破坏 TileLang | AST ratchet（本轮真修了 2 个文件） | ✅ 本轮 |
| 内核读 env | `utils.kernel_flag()` 单一入口 + AST ratchet（禁止直接 `os.environ`、flag 名钉死） | ✅ 本轮（未参数化，见缺口 6） |
| 测试顺序污染 | 禁止全局 fixture；整套 + 逐文件双跑 | ✅ 360 项 |
| 许可证/出处 | `NOTICE` + 每个 impl 的 `provenance` | ✅ |

## 还没做的（明确清单）

按"要么给证据，要么写没做"的规矩，这七条没有证据，所以不写成完成：

1. **prefill 的编排**：内核（`paged_prefill_ascend950` + split/head/frontier 三个变体）在仓里，
   但 prefill 一步的 tile 数取决于 batch，`plan()` 的 per-shape 模型表达不了，所以只迁了语义
   （reference + fp64 oracle）并把它声明成 `enabled=False` 的未接线实现。见 `archive-attention.md`。
2. **split-KV**：TileInfer 记 3×，但两次把设备卡死（`ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`），
   本批 25 分钟编译扫描未收敛。连候选都不注册，理由在 `archive-attention.md` §3。
3. **507035 的"单次搬运 <32 B"约束**：还没有机制。试过的一版按"张量最内层 stride"实现，被既有测试
   当场证伪并回退，见 §7 表。
4. **`vllm_adapter`**：与"全仓零 vllm 依赖"冲突，且需要 vLLM 的 ABI 名；只能放 `integrations/` 才安全，
   本轮选择不做并记录（P5 行）。
5. **head_dim=256 且 K/V 非对称**、**NZ 直吃内核**、**NHD 以外的布局**：契约里声明不支持，不是遗漏。
6. **7 个内核 flag 还没变成编译规格参数**：已是"单一入口 `utils.kernel_flag` + 统一前缀 + ratchet"，
   但仍在读进程状态。其中 5 个在 plan/build 期读（值已进编译键），真正在 trace 期读的是
   `SIMD_SOFTMAX` 与 `PREFILL_NOSCAN_OFF` 两个——参数化要动内核签名与 `spec_key`，需设备复测。
7. **服务级 TPOT A/B**：要起引擎，属于平台语义（`docs/reverse-integration.md` §6 的口径）。
