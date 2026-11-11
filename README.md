# AscendAttention

面向昇腾的高性能**融合算子库**：引擎无关，只提供算子，不提供平台。

## 定位：这一层负责什么

**一句话**：提供**引擎无关的融合算子**（按"算什么"命名，入参只有张量与纯数据），并把
"我能做什么、做到什么程度"以纯数据声明出来；**不认识任何推理引擎，也不提供平台语义**。

两个仓库的边界用**两条可机器检查的判据**划：

| 判据 | 本仓（算子） | 平台仓 |
|---|---|---|
| 可以出现引擎的名字（`vllm.`、`vllm_ascend.`、引擎元数据类） | ❌ 出现即越界 | ✅ 引擎符号、补丁点、entry point 都归它 |
| 可以出现内核与算子数值实现 | ✅ 这正是本仓的内容 | ❌ 一行都不该有 |
| 可以出现算法签名（如 FLA 的 `(A_log, a, b, dt_bias, q, k, v, …)`） | ✅ 这是算子自己的 API | ✅ 它按需关键字调用 |
| 决定"用哪个算子的哪个实现" | ✅ `spec/plan/run` 与候选实现 | ❌ 只做符号映射与总开关 |

本仓的"零引擎依赖"有测试钉住（`tests/test_import_hygiene.py`：AST 扫 `vllm`/`vllm_ascend`/`torch_npu`）；
"README 不许宣传不存在的算子"也有测试钉住（`tests/test_readme_contract.py`）。

## 安装

```bash
pip install -c constraints.txt -e ".[npu,tilelang]"   # 参考机（昇腾 950 + TileLang）
pip install -e .                                      # 只有 torch：能 import、能查询能力、能跑纯 torch 基线
pip install -e ".[dev]"                               # 加 pytest
```

两种依赖声明是**故意分开**的：

| 文件 | 作用 | 内容 |
|---|---|---|
| `pyproject.toml` → `dependencies` | **契约**：陌生人 `import ascend_attention` 需要什么 | 只有 `torch>=2.5`（下限，不是精确钉版）——没有 NPU、没有 TileLang 也必须能 import 并说清"缺什么" |
| `constraints.txt` | **证据**：账本里每个数字是用什么版本测出来的 | `torch 2.13.0+cpu`、`torch_npu 2.13.0rc1`、`tilelang 0.1.15`、CANN 9.3 |

可选 extra：`npu`（`torch_npu`）、`tilelang`、`dev`、`pinned`（参考机的精确版本集）。
**不要在测比值的过程中升级** `torch_npu`/`tilelang`：代码生成会变，升级后测出的比值与已提交的账本不可比
（见 `docs/perf-rules.md`）。CANN 不是 pip 包，参考机为 9.3，`torch_npu` 跨大版本会直接拒绝运行。

## 仓库功能

- **注意力**：paged decode（GQA、head_dim 128/256、NHD 池、分页表）与 paged prefill（reference 语义 + fp64 oracle；
  prefill 内核在仓，tile-count 编排属平台侧，未接线）
- **线性注意力（GDN）**：卷积更新、门控、分块递推 chunk 链、权重整形
- **MoE**：路由 top-k（`moe_topk_softmax`）、块对齐（`moe_align_block_size`）、SwiGLU
  （`moe_silu_and_mul`）、非量化专家计算（`moe_fused_experts` / `moe_unquantized_fused_moe`）
  五个**算子**，加 token 对齐（slot 布局）与权重打包（ND → compact-NZ）——MoE 的数学全在本仓；
  平台只负责把它们绑到引擎符号上
- **采样**：greedy、阈值式 top-k（保留并列）；top-p **未实现**
- **多形态实现 + 库内分派**：目前落地三种形态——TileLang 内核、torch 组合、直调厂商算子
  （AscendC 形态与"搬入的厂商代码"按当前边界**未实现**）；按设备、形状、dtype、布局、**能否进图**在库内分派，
  每个实现声明能力与实测比值，没测过的实现不许默认启用
- **验证设施**：数值 oracle（离线可跑）、单算子基准与账本（`bench/results/`）

## 已迁算子（完整清册见 `python -m ascend_attention --capabilities`）

| op | 实现 | 默认 |
|---|---|---|
| `paged_decode_attention` | `torch.reference`（oracle）、`tilelang.decode950`（上卡验证 + 真图捕获） | 否（内核无 op 级 A/B） |
| `paged_prefill_attention` | `torch.reference`、`tilelang.prefill950`（内核在仓、未接线） | reference 是 |
| `gdn_gating` | `torch.gating` | 是（基线） |
| `gdn_conv_update` | `torch.causal_conv1d_update`、`tilelang.conv950`（op 级 14.15×） | 基线是 |
| `gdn_chunk_wu` | `torch.solve_tril_recompute_wu`、`tilelang.chunk_wu950`（14.55×） | 基线是 |
| `moe_slot_layout` | `torch.slot_layout` | 是 |
| `moe_prenz_pack` | `torch.prenz_pack` | 是 |
| `moe_topk_softmax` | `torch.topk_softmax` | 是（基线） |
| `moe_align_block_size` | `torch.align_block_size` | 是（基线） |
| `moe_silu_and_mul` | `torch.silu_and_mul` | 是（基线） |
| `moe_fused_experts` | `torch.fused_experts` | 是（基线） |
| `moe_unquantized_fused_moe` | `torch.unquantized_fused_moe` | 是（基线） |
| `sampling_topk_threshold` | `torch.topk_threshold`、`vendor.deep_select`（**3.8432×**，唯一默认开启的非基线条目） | 两者都是 |
| `sampling_greedy` | `torch.argmax` | 是 |

`manifest.json` 只收"有实测比值且默认启用"的实现（13 条）；默认关的实现连同 `graph`/`requires`/拒绝原因在完整清册里，
两者的分工见 `docs/contracts/exposure.md`。

### 明确未实现（写在前面，免得被集成方当成已有能力）

- **MoE 的内核化**：计算已是纯 torch 基线（正确性有 fp64 定义式取证），但**没有加速内核**；
  `grouped_gemm` 0.86×、`prenz` 直吃、`align_*` 未达门槛都只归档，理由见 `docs/archive-moe.md`。
- **MLA**、**量化（fp8/fp4）算子**：属本仓的算子面，未实现。
- **采样**：top-p / min-p 未实现（只有 greedy 与阈值式 top-k）。
- **prefill 编排接线**：内核在仓，tile-count 编排属平台侧，尚未接上。

## 解耦设计

- 不依赖任何推理引擎，**不注册任何引擎入口**；供平台发现的是**纯数据描述符**
  （`manifest.json` / `capabilities()`），不含任何引擎符号
- 对外暴露四项：**能力查询 / 规划 / 执行 / 描述符**（描述符是纯数据，供平台启动期校验）；
  入参只接受张量与纯数据，不接受任何引擎的元数据对象
- 每个算子声明两项能力：**能否进图**、**支持哪些 KV 布局**——平台据此提前拒绝，而不是等集成时暴露
- 分派必须廉价：每次调用只做形状分类与查表，**不在调用时测速**（测速需同步，会破坏图捕获并扭曲被测）
- 两种产物流转并存：Python 级合并（无编译产物）、编译内核（按形状编译并缓存）

## 与平台层仓库的关系

- 依赖方向**单向**：平台层 → 本仓库；本仓库不反向依赖
- 平台层未安装本仓库时，应能回退到厂商原生算子
- 本仓库以**纯数据清单**（能力 + 实测比值 + 支持布局 + 能否进图）向平台层暴露自己；
  **"用哪一家、绑到引擎的哪个符号"的策略不在本仓库**——那是引擎知识，归平台
- **引擎适配不在本仓**：把引擎的算子符号接到本仓算子上，是平台的 `engine/abi/` 的职责。本仓提供的是
  算法签名，不是引擎 ABI；两者形状相似只是因为引擎也按算法调（例如
  `fused_sigmoid_gating_delta_rule_update` 是 FLA 的签名，引擎只是按关键字调用同名函数）
- **供给契约的代码侧尚未定稿**：语义算子名词表、`platform_name` 字段、provider 发现入口要两仓一起定
  （平台先有 `contracts/provider.py` 与描述符 schema 的第二版），在那之前本仓不单方面加字段——
  加一个没人读的字段只会让清单看起来比实际更契约化

## 支持的组合

厂商平台 + 本仓算子 · 本仓平台 + 本仓算子 · 本仓平台 + 厂商原生算子（基线：厂商平台 + 厂商算子）

## 非目标

设备语义、KV 分配、批元数据、图捕获、worker、多卡通信、引擎适配——这些属于平台层。

注意区分：**MLA 与量化是算子**（属本仓，未实现），不是平台语义。把"MoE 路由/专家计算"写进
"平台层"曾让一条真实计算路径无人认领——复盘与两仓修复清单在平台仓的 `docs/notes/repo-boundary.md`（跨仓文，不在本仓）。

## 状态

昇腾单卡（Ascend 950）：上表算子已实现，每个都有 fp64/fp32 数值取证与账本；负结果
（MoE `grouped_gemm` 0.86×、`prenz` 直吃、decode 内核 0.362×、split-KV）与死代码只归档、不进主干，
理由与翻案条件写在 `docs/archive-*.md`。其他国产卡：设备能力声明与布局声明已预留，实现未开始。

逐条验收与"未做"清单在 `docs/completion-audit.md`；迁移计划与决策（D1–D8）在
`docs/plan-migrate-tileinfer.md`。
