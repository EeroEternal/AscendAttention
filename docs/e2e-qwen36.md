# Qwen3.6-35B-A3B 端到端：跑到哪一步、卡在哪一条

目标：**vLLM 0.23 + AscendPlatform（OOT 平台）+ AscendAttention（算子层）** 在本机把
`/home/models/Qwen3.6-35B-A3B` 跑出一次正确的生成。本文记录实测结果与唯一的阻塞项，不是计划。

## 结论（2026-11-11 实测）

```
VLLM_PLUGINS=ascend_platform ASCEND_PLATFORM_OPS=ascend_attention \
PYTHONPATH=AscendPlatform/src:AscendAttention/src python boot.py     # 原样：崩
```

原样跑：权重**加载完成**，在 `LLM()` 内部的 profile/dummy run 走到**第一个 MoE 层的路由**时崩
（所以不是"跑不了模型"，是第一步就算不出来）：

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

即 **Qwen3.6 在本平台上端到端可跑通**，生成内容正确（"Paris"），不是空转。链路上实测确认走本仓的是：

* attention：平台的 `engine/attention/`（decode + prefill，日志里 prefill 的 tile-bucket
  "prepared kernel path" 逐 bucket 出现）；
* **GDN 的 8 个符号解到本仓**（`_Impl.impl` 实测：`chunk_gated_delta_rule → ascend_attention.ops.fla.chunk`、
  `l2norm_fwd → ops.gdn.leaves`、`causal_conv1d_fn → ops.fla.causal_conv1d`）——即 GDN 跑的是本仓实现，
  不是平台自带的 port；
* MoE 专家计算：平台 `engine.moe_method` 注入的 `moe.experts`（= `ops.moe.compute.fused_experts`），
  日志里 `registered AscendUnquantizedFusedMoEMethod`；
* 采样：vLLM 自己的 sampler（本仓的 `sampling.*` 未被绑定，走 vLLM 原生 torch 路径）。

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

## 长生成：平台侧已定位为 NaN，不是逐 token 变差

早先在这里记的是"64 tokens 退化成复读"。平台侧用 `tools/diag_longgen.py` 把它测成了具体形状：
`k=0..13` prefill 与 decode 的 argmax 一致，`k>=14` 起 `token_id` 恒为 0（`'!'`）、`lp_dec = nan`，
之后一直是 `!` ——**一次 NaN 被 greedy 锁死在 token 0，NaN 又经 KV 回灌，所以永远吐同一个 token**，
不是"模型越写越差"。那 17/256 的 MISMATCH 全部落在 11–31 的近 tie 区，是两条路径的数值差，不是损坏。

三组对照（同一平台、同一模型）：本仓当算子层 / TileInfer 当算子层 / `TILEINFER_OPS=moe`（GDN 交给
vLLM FLA）——三组都在同一位置崩。这个证据把范围压到"三者共享的东西"：平台的 attention impl 与
KV/hybrid-state 交接、模型接入（rope/位置）。**一个需要确认的点**：第三组要成立，必须先确认那台机器上
vLLM 的 FLA 真能跑——本会话里同一环境出现过 `Failed to import Triton kernels`，若 FLA 实际没生效，
"排除 GDN"这一步就不成立。

本仓这两处已经排除并留下了守卫：内核（`ASCEND_ATTN_CANDIDATES=off` 输出逐字相同）与 recurrent decode
（`tests/test_gdn_recurrent.py` 逐元素对齐）。

## prefill 的重复编译：键里有"随长度变化"的数

平台侧测到 192 token 的一轮里 GDN prefill 编译花掉 129.7 s（对照同一步 0.8 s）。根因不是
`TILELANG_CACHE_DIR` 没接上（磁盘缓存是按 key 存的，key 稳定就命中），而是**编译键里含长度派生值**：
本仓 `build_prefill_kernel` 的 spec 带 `num_tiles`（= Σ ceil(qo_len·group/block_q)，长度的线性函数），
平台侧 prefill 的 key 带 `width`（这一步要扫的 kernel pages）。两边都每步换一个数，于是每步一个新变体。

本仓已改成按**桶**编译：`num_tiles` 先量化到 2 的幂（`abi.quantise_tiles`），多出来的 tile 是空 tile
（`q_len == 0`，内核与 pack/unpack 本来就按 P3 跳过），所以只是缓冲区略大，计算不变。键因此对
"同一桶内的任意长度"稳定；`tests/test_prefill_tile_bucket.py` 用假编译器在 CPU 上验了这一点（两个不同
长度的 spec/key 完全相同，跨桶则不同）。平台侧对应的是把 `width` 也量化到桶。

## 非阻塞项（不影响"跑通"，影响"可用"）

1. **性能**：8 tokens / 5.7 s ≈ **1.4 tok/s**（vLLM 自带进度条读数）。**归因未做**：候选是纯 torch 的
   256 专家 MoE、GDN 的 torch chunk 链、attention 的 reference/内核路径三者之一或组合，得先 profile
   再下结论。这是下一阶段的工作，不是"跑不通"。
2. **环境**：本机同时装了 `tileinfer` 与 `ascend_platform` 两个 vLLM 平台插件，vLLM 只允许激活一个，
   必须显式 `VLLM_PLUGINS=ascend_platform`（否则 `Only one platform plugin can be activated`）。
3. **`silu_and_mul`**：平台 `dense` 族开关默认关，本模型不吃这条；它不是阻塞项。
4. **`chunk_local_cumsum`**：平台自己 declined（它的 chunk 链内部已算），不是缺实现。

## 复现

**先读 [docs/process-device-access.md](process-device-access.md)：这台机器的 HBM 是共享的，部署模型前要先取得同意。**
下面是"已经拿到同意之后"的命令。

```bash
export VLLM_PLUGINS=ascend_platform
export ASCEND_PLATFORM_OPS=ascend_attention
export PYTHONPATH=/home/userlist/zxs/AscendPlatform/src:/home/userlist/zxs/AscendAttention/src
python /tmp/e2e/boot2.py     # 带适配器的探针脚本（临时，不进仓）
```

探针脚本不进仓的原因：适配器属于平台仓；本仓只负责证明"函数式语义正确"，以及把这条边界写清楚。
