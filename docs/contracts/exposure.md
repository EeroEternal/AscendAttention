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

## 语义名与 provider 入口（供给契约，本仓侧）

平台按**语义名**绑定算子（`moe.experts`），不按本仓内部名（`moe_fused_experts`）——否则平台就得
硬编码我们的实现细节，换一家算子库就全断。本仓这一半已经就位：

| 件 | 落点 | 谁读 |
|---|---|---|
| 语义名词表 | `src/ascend_attention/vocabulary.py`（`family.member`，一 op 一名） | 本仓写清单时；测试钉住唯一性与格式 |
| 描述符字段 `platform_name` | `manifest.json` 每条 op 一个，且是 JSON Schema 的 **required** | 平台的 `contracts/manifest.py`（未知字段它本来就忽略） |
| 发现入口 | `pyproject.toml` 的 `[project.entry-points."ascend_platform.providers"]`，值 `ascend_attention.provider:provider`；另外包根也暴露 `describe()`，供平台 `ASCEND_PLATFORM_OPS=<pkg>` 那条路 | 平台启动期扫描 |
| 工厂 | `ascend_attention/provider.py`：`describe()` 返回描述符，`load(语义名)` 返回 `OperatorHandle` | 平台的 `contracts/provider.py` |

描述符的形状（平台 `parse_descriptor` 定义的，本仓照此产出）：

```python
{"name": "ascend_attention", "version": "0.1.0", "requires_platform": ">=0.1,<0.2",
 "ops": {"moe.experts": {"target": "ascend_attention.ops.moe.compute:fused_experts",
                         "dtypes": ["bfloat16", "float16", "float32"],
                         "graph": "capture_safe", "note": "moe_fused_experts via torch.fused_experts",
                         # 以下字段平台解析器忽略，留给人看/给 doctor 用
                         "platform_name": "moe.experts", "op": "moe_fused_experts",
                         "impl": "torch.fused_experts", "impls": [...], "measured_ratio": 1.0}}}
```

`ops` 的键是**语义名**；`target` 必须 `module:attribute` 且可调用。`attention.paged_decode` **不在**描述符里：
它有上卡验证的内核，但没有 op 级 A/B（唯一的数是 0.362×，比厂商融算子慢），发布它等于让平台按名字选到更差的
实现；它仍可通过 `plan(impl="tilelang.decode950")` 显式使用。

`OperatorHandle` 同时给两条路：`.plan(**纯数据)` + `.run(plan, *tensors)`（严格路径，所有 op 都有），
以及直接调用 `handle(*tensors)`（只对**签名就是张量函数**的 op，目前是 MoE 计算的五个；其余 op
调用它会明确报"plan-driven"而不是猜一个 plan 出来）。明面表在 `provider._DIRECT` 里，不走属性反射。

组名 `ascend_platform.providers` 本仓有一份常量副本（平台那份在它的 `contracts/provider.py`）；
`tests/test_provider.py::test_entry_point_group_matches_the_platform_constant_when_the_platform_is_present`
在两仓都能 import 时要求两者相等——上一轮边界漂移就是"两份声明互相假设对方"造成的，这次把假设写成断言。

**还没兑现的那一半仍是平台的**：`contracts/provider.py` 尚不存在，所以今天没有进程会去扫这个 entry point；
在它落地前，本仓的入口是"装好就绪、没人读"。这不是本仓可以单方面解决的部分，`docs/boundary.md` §4 记着。
