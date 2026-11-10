# 归档：`sampling.fused_sample`（不迁）

判决：**不迁**。按"死代码与负结果只归档不进主干"的规则，代码留在 TileInfer，理由落在这里。

## 事实

| 项 | 值 | 依据 |
|---|---|---|
| 消费者 | **零** | `grep -rn "fused_sample" tileinfer/ --include=*.py` 只命中 `kernels/sampling/` 自身 |
| 端到端数字 | **无** | `docs/notes/perf-candidates.md` 无该条目；没有 A/B 就没有 `measured_ratio` |
| 词表约束 | `vocab_size % 128 == 0`，否则 `ValueError` | `fused_sample.py` 的 `_VL` 检查 |
| UB 上限 | `ub_cap=8192` 叠加后越界，报 `507035`；4096 可用、2048 最稳 | `fused_sample.py:104-114` 的 docstring 实测记录 |
| 形态 | `kernel`，需要编译 + workspace | 因此只能是 `eager_only` |

## 为什么"没有消费者"是决定性的

一个没有调用点的内核，进主干后要付三份成本：manifest 里一条永远为 `None` 的比值、一条只能靠人工
维持的编译键、以及每次有人改采样族时都要问"这个要不要一起改"。收益是零。**归档不是否定它**：
`fused_sample` 的 builder/缓存缺口（源仓 `perf-candidates.md` 记的"完全没有"）本身也是个待办，
只是那个待办属于引擎侧（谁来建、谁缓存、`vocab` 约束怎么在服务启动时就报错），不属于算子库。

## 留下的两个采样算子

| op | 形态 | 为什么留 |
|---|---|---|
| `sampling_topk_threshold` | `torch` + `vendor_call` | 真实收益 3.84×，且 vendor 路径是第五种形态的第一次落地 |
| `sampling_greedy` | `torch` | `do_sample=False` 的分流目标；语义是 argmax 本身，无可比对象（定义性 1.0） |

分流规则只有一条，落在 `ops/sampling/sampler_for(do_sample)`：`False` → `sampling_greedy`，
`True` → `sampling_topk_threshold`。**不做**"一个 op 三个 impl 按布尔分流"——`do_sample` 不是形状，
把它塞进 `ShapeSpec` 会让声明式契约开始撒谎。

## 如果以后要迁回来

前置条件，缺一不可：(1) 引擎侧有真实调用点；(2) 一条 ≥4 对交替的端到端 A/B；
(3) `vocab % 128` 约束在**服务启动时**校验，而不是第一次 decode 时抛；(4) `ub_cap` 由 plan 期参数化
（本仓的规矩：库内不读环境变量）。
