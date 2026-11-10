# 契约：平台读什么（`manifest.json` vs `capabilities()`）

两个产物，同一份 `ImplSpec` 数据，回答两个不同的问题。混用会导致两种相反的错误：把"能用"当成
"全都有"，或者把"存在"当成"能用"。

| 产物 | 命令 | 内容 | 平台什么时候读 |
|---|---|---|---|
| `manifest.json` | `python -m ascend_attention` | **可用子集**：只有 `advertised` ∧ `enabled_by_default` ∧ `measured_ratio >= 1.0` 的实现 | 启动时决定"今天能调什么"；CI 用 git diff 防止清单漂移 |
| `capabilities()` | `python -m ascend_attention --capabilities` | **完整清册**：每个实现，含 `enabled_by_default=False`、`measured_ratio=None`、`graph`、`requires`、`provenance`、`note` | 需要**提前拒绝**时（例如某形状要求 `capture_safe`，而该 op 当前没有启用的实现） |

## 为什么 decode 不在 manifest 里，这不是遗漏

`paged_decode_attention` 有两个实现：`torch.reference`（oracle，不对外）与 `tilelang.decode950`
（设备实测通过、`capture_safe`、但**没有本仓 op 级 A/B**，所以 `measured_ratio=None` 且默认关）。
按四道门，它进不了 manifest——这正是规则在生效，而不是清单漏写。平台若要用它：

1. 显式 `plan(..., impl="tilelang.decode950")`（pin 不受比值门限制）；
2. 或等一次真实 A/B 补上 `measured_ratio`，届时它会自动出现在 manifest 里。

## 两边不可能不一致

`manifest()` 是 `_eligible()` 对同一批 `ImplSpec` 的过滤结果；`capabilities()` 不过滤。没有第二份
手写的名字表，所以"清单说能调、实际调不到"这种漂移在结构上不存在。
