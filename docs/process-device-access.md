# 过程约束：用算力卡之前先问

这条约束来自实际影响：机上的 HBM 不是我们一个人的，同事同时在用。一次 35B 模型的部署会占掉约
70 GB 权重 + KV cache（本机 123 GiB 里留 0.8 的占比就是近百 GB），跑起来是分钟级；别人这时候跑
profiling 或复现，看到的是 OOM 或者吞吐腰斩，而且他们不知道是自己代码的问题还是有人在占卡。

## 规则

**需要用到计算卡（NPU/HBM）做验证时，先说明、等到同意，再部署。** 不自动起模型，不自动跑设备用例。

"用到计算卡"包括：`LLM(...)` 部署模型、`torch.npu` 上的 kernel 编译/数值验证、任何 `-m npu` 的用例、
profiling 采集。**不包括**：CPU 套件、`torch` 参考实现、静态检查、文档、清单生成。

## 申请时给出这四项

1. **目的**：要回答什么问题（例："确认 64 token 退化是否由路由适配器引入"）；
2. **占用**：模型与规模（例：`Qwen3.6-35B-A3B` bf16 ≈ 70 GB 权重，`gpu_memory_utilization=0.8`）；
3. **时长**：预计分钟数（含首次编译；本机 prefill 内核首次约 39 s，模型加载约 24 s）；
4. **收尾**：跑完如何释放（进程退出、确认 `mem_get_info` 回到空闲）。

## 不用卡也能做的部分

| 想验证的 | 不用卡的做法 |
|---|---|
| 算子契约、plan/run 形状、dtype、门控状态 | `PYTHONPATH=src pytest tests -m "not npu"` |
| 数值正确性（reference 语义） | 仓内 fp64 oracle，CPU 上跑 |
| 清单/描述符/跨仓解析 | `python -m ascend_attention`、`tests/test_platform_requirements.py` |
| 内核分派决策（不执行内核） | `capabilities()` / `ascend_attention.describe()` 的 `requires`/`graph` 字段 |

设备用例是**标记驱动**的（`@pytest.mark.npu`，本仓现有 18 条），所以在有卡的机器上 `pytest tests` 会真的
用卡——要保持在 CPU 上，加 `-m "not npu"`（实测：`458 passed, 1 skipped, 18 deselected`），不要指望
"刚好跳过"。**默认命令就是这个**；只有在拿到同意、并且确实要验证数值/内核时，才跑全量或 `-m npu`。

## 本约束写在这里的原因

写在仓里而不是只在对话里：换一次会话、换一个 agent，读到的第一条上下文可能是 `docs/`。
复现 e2e 的那份命令清单在 `docs/e2e-qwen36.md`，它的"复现"一节现在也先指向这里。
