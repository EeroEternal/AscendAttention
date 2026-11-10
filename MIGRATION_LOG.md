# MIGRATION_LOG

一次搬运动作一条记录。规则（与 `docs/plan-migrate-tileinfer.md` §4.2 一致）：

1. **搬运不改行为**：第一版只改 import 路径，docstring、`file:line` 依据、注释逐字保留；
2. **契约测试随迁**：被搬模块的测试同批搬，改 import 后原样通过才算"搬对了"；
3. **记"没搬什么、为什么"**：漏掉的往往比搬走的重要。

源：`/home/userlist/zxs/TileInfer` @ `339969a`（vLLM 0.23 原生 Ascend 后端，参考机 `tileinfer-102`）。

## P0（骨架 + attention 合同层 + `stateless` 样例）

| 目标文件 | 源文件 | 改动 | 随迁测试 |
|---|---|---|---|
| `src/ascend_attention/utils.py` | `tileinfer/utils.py` | 仅包路径 | — |
| `src/ascend_attention/metadata.py` | `tileinfer/metadata.py` | 仅包路径 | `test_attention_api.py` |
| `src/ascend_attention/plan.py` | `tileinfer/plan.py` | 仅包路径；文末追加 `OpPlan`（新代码） | `test_attention_api.py` |
| `src/ascend_attention/api.py` | `tileinfer/attention/api.py` | `from ..x` → `from .x` | `test_attention_api.py` |
| `src/ascend_attention/backends/base.py` | `tileinfer/attention/backends/base.py` | `from ...x` → `from ..x`（含函数内 import） | `test_attention_api.py` |
| `src/ascend_attention/backends/reference.py` | `tileinfer/attention/backends/reference.py` | 同上 | `test_attention_api.py` |
| `src/ascend_attention/backends/__init__.py` | 同名 | 仅包路径 | — |
| `src/ascend_attention/testing/*.py` | `tileinfer/testing/*.py` | 仅包路径 | 被上面两个测试间接覆盖 |
| `src/ascend_attention/ops/gdn/gating.py` | `tileinfer/kernels/gdn/__init__.py` | 仅包路径（文件名改成 `gating.py`，因为它不含内核，是纯 torch 组合） | `test_gdn_gating.py` |
| `src/ascend_attention/abi.py` | `tileinfer/kernels/attention/paged_decode_ascend950.py`（`padded_rows`/`declared_pool_cap`/`pool_view`）+ `tileinfer/serving/buckets.py`（`capacity_view`） | 三个来源合并成一个模块；逻辑逐行保留；新增 `ASCEND_ATTN_POOL_CAP`，旧名 `TILEINFER_POOL_CAP` 仍可读 | `test_abi.py`（新写） |
| `tests/test_attention_api.py` | 同名 | 改 import 根 | — |
| `tests/test_gdn_gating.py` | 同名 | 改 import 根 | — |

新写（无源）：`spec.py`、`dispatch.py`、`manifest.py`、`device.py`、`__init__.py`、`ops/**`、
`tests/test_spec.py`、`tests/test_dispatch.py`、`tests/test_manifest.py`、
`tests/test_import_hygiene.py`、`tests/test_gating_contract.py`、`bench/bench_gdn_gating.py`。

## P1（首个编译内核算子：paged decode）

| 目标文件 | 源文件 | 改动 | 随迁测试 |
|---|---|---|---|
| `src/ascend_attention/kernels/attention/paged_decode_ascend950.py` | 同名 | ① 模块级 `import tilelang` → 与 `paged_decode.py` 相同的 `try/except` 守卫；② `from ...serving.buckets import capacity_view` → `from ...abi import capacity_view`（消除内核→serving 反向依赖）；③ `from tileinfer.plan import ...` → `from ...plan import ...`；④ `typing` 补 `Any` | `tests/test_decode_device.py`（新写） |
| `src/ascend_attention/kernels/attention/paged_decode.py` | 同名 | 仅包路径（其守卫已是源仓范式，本仓沿用） | 设备侧由 P1 后续批次覆盖 |
| `src/ascend_attention/kernels/attention/kv_nz_layout.py` | 同名 | 无改动（仅 torch） | — |
| `src/ascend_attention/backends/tilelang_ascend950.py` | 同名 | `from ...x` → `from ..x`（含函数内 import） | `tests/test_decode_device.py` |
| `src/ascend_attention/backends/tilelang_ascend.py` | 同名 | 同上 | 同上 |
| `src/ascend_attention/ops/attention/__init__.py` | 新写 | 注册 `torch.reference`（oracle，永不默认）与 `tilelang.decode950`（**未测速→不默认、不进清单**）；`plan_decode` 只做数据校验，编译发生在 planner 内 | `tests/test_decode_device.py` 前两个用例 |

新写：`bench/bench_paged_decode.py`（只记绝对延迟，拒绝把 gather 基线当 `measured_ratio`）。

## 未搬（及原因）

| 源 | 原因 |
|---|---|
| `tileinfer/kernels/attention/paged_decode_ascend950.py`（3,227 行内核主体） | P1；P0 只取 ABI 三个函数，避免在骨架批次引入 TileLang 依赖 |
| `tileinfer/kernels/attention/{paged_decode,split_prefill,paged_prefill,...}.py` | P1（decode/prefill 变体） |
| `tileinfer/kernels/gdn/{proj_fusion,reorder}.py` | **不是内核**：是 vLLM 层 monkeypatch（import `vllm.model_executor.layers.mamba.gdn...`），归 `vllm_adapter`（P5） |
| `tileinfer/kernels/gdn/conv_update_ascend950.py` | P2（需 TileLang） |
| `tileinfer/kernels/gdn/chunk*.py`（torch 基线 4 段 + `chunk_wu`） | P2 |
| `tileinfer/kernels/moe/**`（含负结果 0.86× 与 preNZ） | P3；负结果只归档 |
| `tileinfer/kernels/sampling/**` | P4 |
| `tileinfer/vllm/ops/candidates.py` | 只搬**契约**（四道闸门的语义进 `dispatch.py`），实现不搬 |
| `tileinfer/vllm/**`（模型层、worker、图捕获、`ti_trace`） | 非目标：平台语义，属 `AscendPlatform` |
| `tileinfer/serving/**` | 非目标：引擎侧调度 |
| `legacy/**` | 冻结的 vllm-ascend monkeypatch 层，无设备内核；只作契约参考 |

## P0 与源仓的行为差异（有意为之，逐条可查）

1. **`capacity_view` 的位置**：源仓内核通过 `from ...serving.buckets import capacity_view` 反向依赖
   serving 层（L5 级泄漏）；本仓把它放进 `abi.py`，语义逐行不变（storage offset 必须携带、
   `cap < pool` 报错、非连续输入先 contiguous 物化）。
2. **环境变量**：新增 `ASCEND_ATTN_POOL_CAP`；`TILEINFER_POOL_CAP` 作为兼容回退保留，两条
   都读不到时才取 next-power-of-two。
3. **`plan()` 的语义**：源仓是 attention 专用；本仓是 `plan_kind ∈ {attention, kernel, stateless}`
   的统一入口，attention 的细节仍由 `BatchAttention.plan()` 承担（P1 接内核时不再改签名）。
