# MoE 归档：不迁的四样东西，以及为什么

迁移最容易犯的错不是漏迁，而是**把上一个仓库的负结果当成特性搬过来**——它会以"我们也支持"的名义
留在主干里，然后有人把它打开。这一页就是"决定不迁"的书面记录，每条都写了数字、出处和重新立项的条件。

## 1. `grouped_gemm`（TileLang，Ascend 950）—— 0.86×，负结果

- **出处**：`tileinfer/kernels/moe/grouped_gemm_ascend950.py`（480 行），`docs/notes/perf-candidates.md`。
- **数字**：同进程交替 A/B 中位数 **0.86×**（比厂商 grouped matmul 慢 16%），门槛 1.0×。
- **为什么不迁**：判据是"比值 ≥ 1.0 才进默认"。0.86× 进来只能默认关着，而默认关着的内核在本仓
  没有价值——它不会被任何路径选中，却要跟着编译、跟着回归。
- **重新立项条件**：换 tile 形状或换 `block_m` 后，A/B 中位数 ≥ 1.0 且 e2e ≥ 1.02；届时作为
  `moe_grouped_gemm` 注册，`measured_ratio` 写当次实测。

## 2. `down_prenz`（TileLang）—— 结果错（~1e36），死代码

- **出处**：`tileinfer/kernels/moe/`，被 `grouped_gemm_ascend950.py` 引用。
- **数字**：输出量级 ~1e36（fp16 溢出特征），与 torch oracle 差到没有意义。
- **为什么不迁**：算错的内核不是"待优化"，是"不存在"。归档，不注册，不进清单。
- **重新立项条件**：先有独立的最小复现 + 设备数值测试通过，再谈性能。

## 3. `align_gate_up` / `align_down`（TileLang + c_api）—— 没到 1.0× 门槛

- **出处**：`tileinfer/kernels/moe/align.py`（348 行）。
- **数字**：未达 1.0×（源仓记录为"门槛未达"，无落盘账本）。
- **为什么不迁**：[待复核] 的数字不进主干。而且它依赖 c_api 直调，属于"平台/引擎"边界，
  与本仓"引擎无关"的定位冲突。
- **重新立项条件**：能在纯 TileLang 侧表达（不引 c_api）且 A/B ≥ 1.0。

## 4. `shared_expert_fold`（torch，权重拼接 + 路由改列）—— 只有 [待复核] 的 ~1.2 ms/step

- **出处**：TileInfer legacy 层。
- **数字**：~1.2 ms/step，[待复核]（单窗测量）。
- **为什么不迁**：形状是对的（纯 torch、无设备依赖），但证据等级不够。本仓不接受单窗数字。
- **2026-11-10 复核（更硬的理由）**：`tileinfer/integrations/shared_expert_fold.py` **已不存在**
  （该目录只剩 `__init__.py` 与 `prewarm.py`），共享专家的代数已并入 `fused_experts`
  （`tileinfer/vllm/ops/moe.py:106-110, 818-880`：`sigmoid(x @ shared_gate_weight.T)` 乘共享 MLP 输出）。
  也就是说它今天**不是一个可单独搬的布局资产**，而是 MoE 计算内部的一条分支——迁它等于迁 MoE 计算，
  那是引擎侧（路由 + 专家分组 + vLLM ABI），明确在本仓边界之外。这条理由与"数字不够"独立，且更强。
- **重新立项条件**：补一个同进程交替 A/B（≥4 对），并说明它改变的是**权重布局**还是**计算图**——
  如果是后者，它属于模型实现，不属于算子库。

## 迁了的那两个（对比说明）

| op | 为什么可以迁 | 它证明了什么 |
|---|---|---|
| `moe_slot_layout` | 纯 torch、静态形状、零主机同步；是每个 MoE 内核的**前置**，引擎需要它做回退 | 与 `testing/moe_reference.build_slot_layout`（循环写法）在 compact→slot 映射上逐项一致 |
| `moe_prenz_pack` | 纯 torch、可往返验证 | 只证明**记账**自洽（pack→unpack 复原）与物理序公式；**不**证明设备上可用，docstring 已写明 |

两者的 `measured_ratio` 都是定义性的 1.0（唯一实现，无可比对象），账本里只有绝对延迟：
`bench/results/moe_layout_cpu.json`（slot_layout 0.0878 ms、prenz_pack 0.0433 ms，CPU）。
