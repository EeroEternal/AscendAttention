# 反向接入：让 TileInfer 依赖本仓，并证明 off/on 数字不变

目标里那句"最终把 TileInfer 反向改为依赖本仓"听起来像一次大改，实际能做成**一个文件 + 两行**
的东西——前提是先想清楚哪一层该依赖哪一层。

## 1. 方向：只允许"引擎 → 库"

```
TileInfer (引擎/服务)  ──依赖──▶  AscendAttention (算子库)
       │                                  │
       └── 不允许反向 import ──────────────┘
```

本仓零 `vllm`/`vllm_ascend`/`tilelang` 顶层依赖（`tests/test_import_hygiene.py` 用 AST 钉住），
所以 TileInfer 引用本仓是安全的；反过来一旦本仓 import 了 vLLM，这条线立刻变成环。

## 2. 落点：`tileinfer/vllm/ops/extern.py`（唯一知道对方存在的地方）

```python
BINDABLE = {
    "chunk_local_cumsum": ("ascend_attention.ops.gdn", "chunk_local_cumsum"),
    "l2norm_fwd":        ("ascend_attention.ops.gdn", "l2norm_fwd"),
}
```

- `TILEINFER_OPS_LIB=ascend_attention` → 用本仓实现；未设/空 → 用 TileInfer 自己的实现（**默认**）。
- 只绑"两边语义完全相同"的名字。依赖 vLLM ABI 形状的（`fused_experts`、`topk_softmax`、
  `install.py` 里的 takeover）**不绑**：借一个回答不同问题的实现，是伪装成重构的重写。
- 任何异常（库缺失、符号缺失、版本不符）都 `warning` 后**回落到本地实现**，绝不抛。
- 找不到安装版时按目录向上找兄弟 checkout（开发便利，首次会 warning 一次）。
- 与 `TILEINFER_OPS` 的时机差异：族开关是**每次调用**读；这个绑定在 import 时替换模块属性，
  所以翻转它需要**重启进程**。这条写进了模块 docstring，免得有人以为能热切。

接线就是 `tileinfer/vllm/ops/__init__.py` 里的两行：

```python
from .extern import apply as _apply_extern
_EXTERN = _apply_extern(globals(), ("chunk_local_cumsum", "l2norm_fwd"))
```

## 3. 证据（都不是"应该没问题"）

| 检查 | 命令 | 结果 |
|---|---|---|
| off 模式实现来源 | `TILEINFER_OPS_LIB= python -c "import tileinfer.vllm.ops as o; print(o.l2norm_fwd.__module__)"` | `tileinfer.vllm.ops.gdn` |
| on 模式实现来源 | 同上，`TILEINFER_OPS_LIB=ascend_attention` | `ascend_attention.ops.gdn.leaves` |
| 数值 | 8 组 (op, shape) 输出 sha256 | **逐位相同**（`diff` 为空） |
| TileInfer 自身测试 | `pytest tests/test_gdn_ops.py tests/test_gdn_gating.py tests/test_gdn_causal_conv1d.py`，两种模式 | 各 **50 passed**，结果一致 |
| 本仓侧溯源 | `pytest tests/test_tileinfer_equivalence.py` | gating / conv / chunk 链 / sampling oracle / moe slot **逐位相同** |

那 8 组形状是刻意挑的：`l2norm` 覆盖单行、128 行、17 行（非 2 的幂）和 64×256；
`chunk_local_cumsum` 覆盖 8 行整块、128 行多块、260 行（尾块不完整）与 7 行。

## 4. 为什么这就满足"数字不变"

"off/on 数字不变"本来有两种读法，两种都验了：

1. **算子级**：同一批输入，两边实现的输出逐位相同 → 任何下游聚合（TPOT/TTFT）都不可能有差异，
   因为进入下一步的张量就是同一个张量。这比"跑一遍服务比较耗时"更强也更便宜，且不受机器漂移影响。
2. **测试级**：TileInfer 自己的 50 个算子测试在两种模式下都通过 → 绑定没有破坏任何既有契约。

端到端服务级 A/B（起服务、打请求、比 TPOT）**没有做**，也不该由本仓做：那是平台/引擎的事，
且本机 `--enforce-eager` 与图捕获的差距会让这个数字失真（源仓 `docs/status.md:37`）。

## 5. 尚未绑定的（以及为什么）

| 候选 | 为什么不绑 |
|---|---|
| `fused_experts` / `topk_softmax` / `moe_align_block_size` | 本仓只有布局算子（slot/prenz），没有 MoE 计算算子；对标量/形状的假设不同 |
| `causal_conv1d_update` | 本仓版本已逐位等价，但它是**本仓的 opt-in 内核入口 + torch 基线**，绑定会把 TileInfer 的 fallback 语义（`null_block_id`）带进来 |
| 采样 `topk` | 本仓是 `vendor_call` 形态，直接调 `deep_select`；TileInfer 有自己的门控窗口，两边门控不同，绑定会双重门控 |

**绑定清单可以变长，但每加一个都要先有"逐位相同"的证据**，不是"看起来一样"。
