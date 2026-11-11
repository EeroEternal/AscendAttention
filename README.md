# AscendAttention

面向昇腾的高性能**融合算子库**：引擎无关，只提供算子，不提供平台。

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
- **MoE**：token 对齐（slot 布局）与权重打包（ND → compact-NZ）；路由与专家计算属平台层，**未实现**
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
| `sampling_topk_threshold` | `torch.topk_threshold`、`vendor.deep_select`（**3.8432×**，唯一默认开启的非基线条目） | 两者都是 |
| `sampling_greedy` | `torch.argmax` | 是 |

`manifest.json` 只收"有实测比值且默认启用"的实现（8 条）；默认关的实现连同 `graph`/`requires`/拒绝原因在完整清册里，
两者的分工见 `docs/contracts/exposure.md`。

## 解耦设计

- 不依赖任何推理引擎，**不注册任何引擎入口**
- 对外只暴露三项能力：**能力查询 / 规划 / 执行**；入参只接受张量与纯数据，不接受任何引擎的元数据对象
- 每个算子声明两项能力：**能否进图**、**支持哪些 KV 布局**——平台据此提前拒绝，而不是等集成时暴露
- 分派必须廉价：每次调用只做形状分类与查表，**不在调用时测速**（测速需同步，会破坏图捕获并扭曲被测）
- 两种产物流转并存：Python 级合并（无编译产物）、编译内核（按形状编译并缓存）

## 与平台层仓库的关系

- 依赖方向**单向**：平台层 → 本仓库；本仓库不反向依赖
- 平台层未安装本仓库时，应能回退到厂商原生算子
- 本仓库以**纯数据清单**（能力 + 实测比值 + 支持布局 + 能否进图）向平台层暴露自己；**"用哪一家"的策略不在本仓库**
- 可选提供"寄生适配器"，用于把算子挂到厂商平台上；该路径**只提供算子，不提供平台语义**
  （vLLM 适配器**未实现**：它与"全仓零 vllm 依赖"冲突，要做只能放 `integrations/`）

## 支持的组合

厂商平台 + 本仓算子 · 本仓平台 + 本仓算子 · 本仓平台 + 厂商原生算子（基线：厂商平台 + 厂商算子）

## 非目标

设备语义、KV 分配、批元数据、图捕获、worker、量化、MLA、多卡通信——这些属于平台层。

## 状态

昇腾单卡（Ascend 950）：上表算子已实现，每个都有 fp64/fp32 数值取证与账本；负结果
（MoE `grouped_gemm` 0.86×、`prenz` 直吃、decode 内核 0.362×、split-KV）与死代码只归档、不进主干，
理由与翻案条件写在 `docs/archive-*.md`。其他国产卡：设备能力声明与布局声明已预留，实现未开始。

逐条验收与"未做"清单在 `docs/completion-audit.md`；迁移计划与决策（D1–D8）在
`docs/plan-migrate-tileinfer.md`。
