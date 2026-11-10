# 契约：TileInfer 的算子绑定 ↔ 本仓（覆盖率与缺口）

P5 的验收条款是"manifest 能驱动 TileInfer 的全部绑定——覆盖率 100%，或有 `note` 说明缺口"。
这份表由 `tests/test_binding_coverage.py` **机械校验**：它直接 import TileInfer 的 `bind()`，逐条检查
"有映射"或"有理由"，并检查映射能真的解析到可调用对象。表是给人读的，测试才是判据。

## 现状：13 条绑定，8 条覆盖，5 条缺口

> 计划文档写的是 15 条，实际 `bind()` 返回 13 条（`fused_post_conv_prep` 与 `chunk_gated_delta_rule`
> 各只有一条，早期按 `_forward_core` 变体数过一遍）。以代码为准。

| # | 绑定 | 族 | 本仓对应 | 状态 |
|---|---|---|---|---|
| 1 | `chunk_gated_delta_rule` | gdn | `ops.fla.chunk_gated_delta_rule` | 覆盖（函数） |
| 2 | `l2norm_fwd` | gdn | `ops.gdn.l2norm_fwd` | 覆盖 + **已绑**（`TILEINFER_OPS_LIB`） |
| 3 | `causal_conv1d_fn` | gdn | `ops.fla.causal_conv1d_fn` | 覆盖（函数） |
| 4 | `causal_conv1d_update` | gdn | `ops.fla.causal_conv1d_update` + op `gdn_conv_update` | 覆盖 |
| 5 | `fused_sigmoid_gating_delta_rule_update` | gdn | `ops.fla....` | 覆盖（函数） |
| 6 | `fused_recurrent_gated_delta_rule_packed_decode` | gdn | `ops.fla....` | 覆盖（函数） |
| 7 | `chunk_local_cumsum` | gdn | `ops.gdn.chunk_local_cumsum` | 覆盖 + **已绑** |
| 8 | `fused_post_conv_prep` | gdn | `ops.fla.post_conv.fused_post_conv_prep` | 覆盖（函数） |
| 9 | `topk_softmax` | moe | — | 缺口：MoE 路由算术属于引擎侧（要专家分组 + vLLM ABI） |
| 10 | `moe_align_block_size` | moe | — | 缺口：归档（测下来未达门槛，见 `docs/archive-moe.md`） |
| 11 | `fused_experts` | moe | — | 缺口：专家计算/groupe d GEMM 地带，0.86× 负结果就在那 |
| 12 | `unquantized_fused_moe` | moe | — | 缺口：vLLM `CustomOp` 子类（`custom_op_oot`），是平台 ABI 不是算子 |
| 13 | `silu_and_mul` | dense | — | 缺口：dense 族，不在本次四族范围内 |

## 为什么 5 条缺口都成立

四条是 MoE/dense，一条是平台 ABI。这不是"没做完"，是**边界**：本仓的边界是"算子"，引擎的边界是
"路由、专家分组、`CustomOp` 生命周期"。把 `fused_experts` 搬进来会立刻需要 vLLM 的 `topk_weights` 语义和
`MoeConfig`，那就是把 vLLM 依赖请回来——违反全仓最硬的那条规则。

## 8 条覆盖里只有 2 条"绑定"

覆盖（函数可用）与绑定（`tileinfer.vllm.ops.__init__` 里真的替换掉本地实现）是两件事。只绑 2 条是因为
其余 6 条要么依赖 TileInfer 的 `null_block_id` 回退语义（`causal_conv1d_*`），要么在被 `install.py` 的
takeover 包装后才被调用，直接替函数会让两套门控互相打架。**每加一条绑定，先要有逐位相同的证据**
（`tests/test_tileinfer_equivalence.py` 就是那个证据来源）。
