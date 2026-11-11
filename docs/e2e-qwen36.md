# Qwen3.6-35B-A3B 端到端：跑到哪一步、卡在哪一条

目标：**vLLM 0.23 + AscendPlatform（OOT 平台）+ AscendAttention（算子层）** 在本机把
`/home/models/Qwen3.6-35B-A3B` 跑出一次正确的生成。本文记录实测结果与唯一的阻塞项，不是计划。

## 结论（2026-11-11 实测）

```
VLLM_PLUGINS=ascend_platform ASCEND_PLATFORM_OPS=ascend_attention \
PYTHONPATH=AscendPlatform/src:AscendAttention/src python boot.py     # 原样：崩
```

原样跑：模型**加载成功**（35B-A3B 上卡、权重 70 GB），在**第一个 MoE 层的路由**处崩：

```
File ".../fused_moe/router/fused_topk_router.py", line 24, in vllm_topk_softmax
    ops.topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize)
File "AscendPlatform/src/ascend_platform/ops/install.py", line 320, in takeover
    return impl(*args, **kwargs)
TypeError: topk_softmax() takes 2 positional arguments but 5 were given
```

加上下面这个 **5 行 out-parameter 适配器**（只改 `vllm._custom_ops.topk_softmax` 这一个符号）：

```python
from ascend_attention.ops.moe import compute

def topk_softmax_abi(topk_weights, topk_ids, token_expert_indices, gating_output,
                     renormalize=False, e_score_correction_bias=None):
    w, ids = compute.topk_softmax(gating_output, int(topk_ids.shape[1]), renormalize=bool(renormalize))
    topk_weights.copy_(w.reshape(topk_weights.shape))
    topk_ids.copy_(ids.reshape(topk_ids.shape))
    token_expert_indices.copy_(ids.reshape(token_expert_indices.shape))
```

结果：

```
[e2e] model loaded in 23.4s
[e2e] generated: ' Paris, a city renowned for its rich'
[e2e] OK in 29.1s
```

即 **Qwen3.6 在本平台上端到端可跑通**，链路里真实生效的有：我们的 attention API（decode + prefill，
`engine/attention/prefill.py` 的 tile-bucket 路径）、`engine.moe_method` 注入的我们的
`moe.experts_unquantized`（MoE 专家计算）、平台自己的 GDN ports、vLLM 的采样。
生成内容正确（"Paris"），不是空转。

## 唯一的阻塞项：路由的 out-parameter ABI

本仓发布的是**函数式**签名（`topk_softmax(scores, top_k) -> (weights, ids)`），vLLM 0.23 的
`_custom_ops.topk_softmax` 是**out-parameter** 约定（`(topk_weights, topk_ids, token_expert_indices,
gating_output, renormalize) -> None`，返回值靠写进前三个张量）。平台的 `install.takeover` 把我们的函数
**直接**塞进那个符号，于是第一个 MoE 层就 `TypeError`。

这不是本仓的 bug：适配是引擎 ABI 的职责（`docs/boundary.md` §4.3，平台的 `engine/abi/`）。但"绑定成功"
和"能用"是两件事——`install_ops()` 报 `11 bound` 时它绑定的是符号，不是可调用的签名。修法只有一处：

* 在 `ascend_platform/engine/abi/vllm_0_23.py` 加一张 `ADAPTERS = {"moe.routing_topk_softmax": ...,
  "moe.align_block_size": ...}`，由 `install_engine_kwargs()` 随 `engine_factories` 一起注入，
  `install.takeover` 在 `impl(*args, **kwargs)` 前先过 adapter；或
* 在 `takeover` 里按 `platform_name` 查一次适配表（语义名 → 适配器）。

`moe.align_block_size` 是**同类但当前未触发**的第二个：`moe_align_block_size(topk_ids, E, block,
sorted_ids, expert_ids, num_tokens_post_pad, expert_map)` vs 本仓的 `(topk_ids, E, block) -> 三元组`。
本模型的 torch 专家路径不经过它（我们的 `fused_experts` 直接吃 `topk_ids`），所以没崩；换模型或换专家
内核就会崩，应该一起加适配。

## 非阻塞项（不影响"跑通"，影响"可用"）

1. **性能**：8 tokens / 5.7 s ≈ **1.4 tok/s**。主因是 MoE 专家计算走纯 torch（256 专家 × 40 层），
   即 README「明确未实现」里的 MoE 内核化；这是下一阶段的工作，不是阻塞。
2. **环境**：本机同时装了 `tileinfer` 与 `ascend_platform` 两个 vLLM 平台插件，vLLM 只允许激活一个，
   必须显式 `VLLM_PLUGINS=ascend_platform`（否则 `Only one platform plugin can be activated`）。
3. **`silu_and_mul`**：平台 `dense` 族开关默认关，本模型不吃这条；它不是阻塞项。
4. **`chunk_local_cumsum`**：平台自己 declined（它的 chunk 链内部已算），不是缺实现。

## 复现

```bash
export VLLM_PLUGINS=ascend_platform
export ASCEND_PLATFORM_OPS=ascend_attention
export PYTHONPATH=/home/userlist/zxs/AscendPlatform/src:/home/userlist/zxs/AscendAttention/src
python /tmp/e2e/boot2.py     # 带适配器的探针脚本（临时，不进仓）
```

探针脚本不进仓的原因：适配器属于平台仓；本仓只负责证明"函数式语义正确"，以及把这条边界写清楚。
