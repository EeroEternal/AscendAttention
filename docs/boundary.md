# 边界：MoE 计算掉在两条声明之间，以及现在的归属

这份文件是**算子仓一侧**的复盘与归属表。跨仓的事故记录与平台侧修复清单在 AscendPlatform 的
`docs/notes/repo-boundary.md`；这里只写"哪件事是本仓的、证据是什么、还差什么"。

## 1. 之前错在哪（结论：对方说得对，我这边的那句话是错的）

本仓 README 曾写：

> MoE：token 对齐（slot 布局）与权重打包（ND → compact-NZ）；**路由与专家计算属平台层，未实现**。

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
2. **供给契约的代码侧**：语义算子名词表、描述符里的 `platform_name`、provider 发现入口——这三件
   要两仓一起定。平台侧先要有 `contracts/provider.py` 与描述符 schema 第二版；在那之前本仓单方面加
   字段，只会让清单"看起来契约化"而没人读。
3. **引擎 ABI 适配**：把引擎符号接成本仓算法签名，归平台的 `engine/abi/`（本仓不写，也不该写）。
