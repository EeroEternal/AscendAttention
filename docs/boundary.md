# 边界：MoE 计算掉在两条声明之间，以及现在的归属

这份文件是**算子仓一侧**的复盘与归属表。跨仓的事故记录与平台侧修复清单在 AscendPlatform 的
`docs/notes/repo-boundary.md`；这里只写"哪件事是本仓的、证据是什么、还差什么"。

## 1. 之前错在哪（结论：对方说得对，我这边的那句话是错的）

本仓 README 曾写：

> MoE：token 对齐（slot 布局）与权重打包（ND → compact-NZ）；**路由与专家计算属本仓算子层**（已实现，
> 纯 torch；内核化未做）。此句曾是平台层的误记，见本文件 §1。

平台仓 README 同时写"非目标：算子数学"，并且**有测试钉住**（`tests/test_no_operator_math.py`）。
两句话各自自洽，合起来把 MoE 路由 / 块对齐 / 专家计算判给了对方——于是这条计算路径两个仓都不认领。

我这边的错误是**归属写错**，不是"漏迁"：P3 批次的迁移范围本来就只列了
`slot_layout` / `prenz` / `shared_fold`（见 `docs/plan-migrate-tileinfer.md`），所以没迁 MoE 计算并不
违反当时的计划；违反的是"算子数学在算子仓"这条边界——把计算说成"平台层"，等于替平台认领了它
明确不做的部分。

## 2. 三条让错误一直看不见的机制（已复核）

| 机制 | 复核方式 | 结果 |
|---|---|---|
| 平台 13 个必需算子按模块无条件 import，缺一个就整张表塌 | `_families()` 直接调用 | `ModuleNotFoundError: ascend_attention.ops.fused_moe_oot` → 整表为空（0 bound / 0 unbound） |
| 平台有 `load_manifest`，但 `src/` 里零调用点 | `grep -rn load_manifest src/` | 只有定义与内部调用，启动判断从不读清单 |
| 端口按包内路径解析，"能 import"就被当成"接上了" | `ports._LAYOUTS` | 是；8/13 的 GDN 算子确实能 import，掩盖了 5 个不存在 |

冲突时**顺序**也值得记一笔：平台 `ports.OPERATOR_ROOTS` 是 `(ascend_attention, tileinfer)`，
其 `_acl.OPERATOR_ROOTS` 是反过来的 `(tileinfer, ascend_attention)`。后者只用于成员判断
（`any(_matches(module, root) ...)`），所以**顺序不影响行为**——这是重复声明会漂移的卫生问题，
不是 bug。我没有把它当成错误来记。

## 3. 本仓现在的归属（13/13，仍未绑到引擎，那部分归平台）

`tests/test_platform_requirements.py` 把这张表钉住；平台在时它还要求**平台列表与本表完全相等**，
两边任何一方增删名字都会红。

| 引擎侧名字 | 本仓落点 | 形态 |
|---|---|---|
| `chunk_gated_delta_rule` | `ops/fla/chunk.py` | 纯 torch 链 |
| `l2norm_fwd` | `ops/gdn/leaves.py` | 纯 torch |
| `causal_conv1d_fn` / `causal_conv1d_update` | `ops/fla/causal_conv1d.py` | 纯 torch |
| `fused_sigmoid_gating_delta_rule_update` | `ops/fla/gating.py` | 纯 torch |
| `fused_recurrent_gated_delta_rule_packed_decode` | `ops/fla/recurrent.py` | 纯 torch |
| `chunk_local_cumsum` | `ops/gdn/leaves.py` | 纯 torch |
| `fused_post_conv_prep` | `ops/fla/post_conv.py` | 纯 torch |
| `topk_softmax` | `ops/moe/compute.py` | 纯 torch（本轮新增） |
| `moe_align_block_size` | `ops/moe/compute.py` | 纯 torch（本轮新增） |
| `silu_and_mul` | `ops/moe/compute.py` | 纯 torch（本轮新增） |
| `fused_experts` | `ops/moe/compute.py` | 纯 torch（本轮新增） |
| `unquantized_fused_moe` | `ops/moe/compute.py` | 纯 torch（本轮新增） |

五个新的计算算子都是**基线**：`measured_ratio` 是定义性的 1.0（没有可比的第二个实现），账本
`bench/results/moe_compute_cpu.json` 只记绝对延迟（T=256/H=1024/I=512/E=32/K=4 下
`fused_experts` 12.8 ms、`topk_softmax` 0.110 ms、`moe_align_block_size` 0.124 ms）。
正确性不是靠"看起来对"：`tests/test_moe_compute.py` 用 fp64 逐 token 循环对拍，并覆盖并列 logits、
零 token 专家、`apply_router_weight_on_input`、`expert_map` 跳专家、`quant_config` 必须拒绝这几条。

## 4. 还没做（明确）

1. **MoE 计算的内核化**：现在全是 torch。`grouped_gemm` 0.86×、`prenz` 直吃、`align_*` 未达门槛
   仍是归档状态（`docs/archive-moe.md`），没有新的加速证据。
2. **供给契约**：**两侧都已落地，并已跨仓实测**。本仓：`vocabulary.py` 的语义名、`manifest.json` 每条
   op 的 `platform_name`（schema required）、`pyproject.toml` 的 `ascend_platform.providers` entry
   point、包根 `describe()`、`provider.py` 的工厂（描述符 + `load(语义名)`）。平台：`contracts/provider.py`
   （它自己的 `PROVIDER_GROUP`，与本仓常量由 `tests/test_provider.py` 钉成相等）。
   实测结果（`VLLM_PLUGINS=ascend_platform ASCEND_PLATFORM_OPS=ascend_attention`，跑它的
   `install_ops()`）：**10 bound / 2 unbound / 1 off-by-switch**，此前是 0 bound / 12 unbound。
   `topk_softmax`、`moe_align_block_size`、`fused_experts` 明确是**经 provider** 绑定的；
   `unquantized_fused_moe` 这个它旧表里的键由本仓 `ALIASES` 兼容（`alias_of`）后也解析到了本仓函数。
   两个 unbound 的原因都不在本仓：`chunk_local_cumsum` 是它自己的 declined（理由充分）；
   `unquantized_fused_moe` 的 patch 把数学函数当类工厂用（`subclass = impl(base)`）。
   **仍有偏差，且不在本仓**：GDN 的 8 个名字平台仍按 vLLM 符号去问（走它自己的 layout 回退，等它 P1 的
   `requirements.py` 改语义名）；`fused_moe_oot.make_oot_method` 是引擎 ABI，归它的 `engine/abi/`。
3. **引擎 ABI 适配**：把引擎符号接成本仓算法签名，归平台的 `engine/abi/`（本仓不写，也不该写）。
4. **内核化**：MLA 只有 reference（无语义缺口、无加速内核）；fp8 有定义式（`quant.dequant_fp8` /
   `quant.fp8_gemm`）无 fp8 内核，fp4 未实现（格式/block scale/硬件路径无可验证依据）；MoE 计算同上。
   这三条都是"没有加速证据就不写内核"，不是遗忘。
