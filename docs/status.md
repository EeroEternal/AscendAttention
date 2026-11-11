# 状态与账本（迁移进度，按批次更新）

参考机：`tileinfer-102`（`Ascend950PR_9579`，1 die / 131072 MiB / 28 cube / 56 vector）。
环境：与 TileInfer 完全一致——`torch 2.13.0+cpu`、`torch_npu 2.13.0rc1`、`tilelang 0.1.15`、
CANN 9.3（`ASCEND_HOME` 指向 TileInfer 的 `.work/cann930`），缓存复用 `tl_cache_213`。
没有自建 venv：任何版本漂移都会让本仓的比值和源仓不可比。

## 已迁算子

| op | impl | 形态 | 默认 | 记到的比值 | 证据等级 |
|---|---|---|---|---|---|
| `gdn_gating` | `torch.gating` | torch 组合（`stateless`） | 是（基线） | 1.0（基线定义） | [核实] 与 fp64 oracle 逐例对齐 |
| `paged_decode_attention` | `torch.reference` | torch oracle（`attention`） | 否 | — | [核实] 数值参照 |
| `paged_decode_attention` | `tilelang.decode950` | TileLang 内核 | **否** | **0.362**（vs 厂商 FIA） | [核实] 设备实测 + fp32 oracle 对齐 |
| `gdn_conv_update` | `torch.causal_conv1d_update` | torch 组合 | 是（基线） | 1.0 | [核实] |
| `gdn_conv_update` | `tilelang.conv950` | TileLang 内核 | **否** | **14.15**（op 级） | [核实] 设备实测；e2e 0.995 来自源仓 [待复核] |
| `gdn_chunk_wu` | `torch.solve_tril_recompute_wu` | torch 组合 | 是（基线） | 1.0 | [核实] |
| `gdn_chunk_wu` | `tilelang.chunk_wu950` | TileLang 内核 | **否** | **14.55**（vs 链中 2 个算子） | [核实] 设备实测；输出相对误差 1.3e-3 |
| `paged_prefill_attention` | `torch.reference` | torch oracle（`attention`） | 是 | 1.0（基线定义） | [核实] 独立 fp64 定义式（页号展开到 token 槽） |
| `paged_prefill_attention` | `tilelang.prefill950` | TileLang 内核 | **否** | — | **未接线**：内核在仓（`paged_prefill_ascend950`），编排是 tile-count 驱动、非 plan-backed；`plan()` 显式 `NotImplementedError` |
| `moe_slot_layout` | `torch.slot_layout` | torch 组合 | 是 | 1.0（定义性） | [核实] 与 `testing/moe_reference` 的循环写法逐项一致 |
| `moe_prenz_pack` | `torch.prenz_pack` | torch 组合 | 是 | 1.0（定义性） | [核实] pack→unpack 逐位复原 |
| `moe_topk_softmax` | `torch.topk_softmax` | torch 组合 | 是（基线） | 1.0（定义性） | [核实] fp64 逐 token 对拍 + 并列按最小 id |
| `moe_align_block_size` | `torch.align_block_size` | torch 组合 | 是（基线） | 1.0（定义性） | [核实] 与自身循环定义逐项一致 |
| `moe_silu_and_mul` | `torch.silu_and_mul` | torch 组合 | 是（基线） | 1.0（定义性） | [核实] 定义式逐位（bf16 舍入一致） |
| `moe_fused_experts` | `torch.fused_experts` | torch 组合 | 是（基线） | 1.0（定义性） | [核实] fp64 逐 token 循环对拍（含 expert_map / 权重前置） |
| `moe_unquantized_fused_moe` | `torch.unquantized_fused_moe` | torch 组合 | 是（基线） | 1.0（定义性） | [核实] 与 `fused_experts` 同解 |
| `sampling_topk_threshold` | `torch.topk_threshold` | torch 组合 | 是（基线） | 1.0 | [核实] 与 `mask_top_k_by_logits` 对齐 |
| `sampling_topk_threshold` | `vendor.deep_select` | vendor_call | **是** | **3.8432**（B=8/V=152064/k=20/bf16，6 对） | [核实] 设备实测；清单里唯一非基线条目 |
| `quant_dequant_fp8` | `torch.fake_quant` | 纯 torch | 是（基线） | 1.0（定义性） | [核实] 饱和边界、scale 轴校验、fp8 网格幂等、bf16 不提精度 |
| `quant_fp8_gemm` | `torch.dequant_gemm` | 反量化 + matmul | 是（基线） | 1.0（定义性） | [核实] 对 fp64 定义式一致；"先反量化再乘"与"先乘再折叠 scale"在正确轴上等价 |
| `mla_decode_attention` | `torch.reference` | 纯 torch 循环 | 是（基线） | 1.0（定义性） | [核实] 对 fp64 逐请求/逐头定义式一致；含"部分页尾部不参与""零长请求返回 0 不返回 nan" |
| `sampling_min_p` | `torch.min_p` | torch 组合 | 是（基线） | 1.0（定义性） | [核实] 与独立定义逐 mask 一致；含"恰好等于下限"的边界用例 |
| `sampling_top_p` | `torch.top_p` | torch 组合 | 是（基线） | 1.0（定义性） | [核实] 与独立前缀式定义逐 mask 一致（p∈{0.1,0.5,0.9,0.99}） |
| `sampling_greedy` | `torch.argmax` | torch 组合 | 是 | 1.0（定义性） | [核实] 与 `torch.argmax` 逐位相同 |

## 账本（`bench/results/*.json`）

| 文件 | 内容 | 结论 |
|---|---|---|
| `decode_b8_s2048_d128.json` | B=8 / S=2048 / d=128 / group=8，vs `npu_fusion_attention`（dense KV） | **0.362×，慢 2.76×**；与源仓自述「TPOT 2.53×」一致，方向不矛盾 |
| `gdn_conv_c4096.json` | C=4096 / L=8 / W=4 / bf16，vs `causal_conv1d_update` | **14.15×**（0.0168 ms vs 0.272 ms），输出相对误差 4.1e-3 |
| `gdn_chunk_wu_t256.json` | T=256（2×128）/ HV=4 / D=128，vs `solve_tril`+`recompute_w_u` | **14.55×**（0.167 ms vs 2.600 ms），w/u 相对误差 1.3e-3 |
| `gdn_gating_t128.json` | T=128 / H=16，基线校准 | 1.0（定义性；0.019 ms，CPU） |
| `moe_layout_cpu.json` | slot_layout 0.0878 ms / prenz_pack 0.0433 ms（CPU） | 绝对延迟，无比值（唯一实现） |
| `sampling_topk_b8.json` | B=8 / V=152064 / k=20 / bf16，vs `torch.topk_threshold` | **3.8432×**（时间比 0.2602，6 对） |
| `moe_compute_cpu.json` | T=256 / H=1024 / I=512 / E=32 / K=4 / blk=32（CPU） | 绝对延迟：`fused_experts` 12.8 ms、`silu_and_mul` 1.45 ms、`align` 0.124 ms、`topk` 0.109 ms |
| `sampling_greedy_b8.json` | B=8 / V=152064，argmax 绝对延迟 | 1.0（定义性对等，0.0179 ms，上卡） |
| `prefill_reference_b4_s4096.json` | B=4 / S=4096，reference 语义 | 绝对延迟，无数值比值 |

测量协议（两个脚本都实现）：同进程交替 A/B、≥4 对、取每对比值中位数、记录命令/卡/日期；
op 级与端到端分开记；**端到端没过关的 op 不进默认**。

## 两个必须记住的判读

1. **decode 内核是负结果，不是失败**：0.362× 说明它在这个形状上不如厂商 FIA。源仓自己的
   `docs/status.md` 也写着 TPOT 2.53×、门是 1.10×。负结果只归档、不进主干——这正是
   `spec.ImplSpec` 的注册期断言与四道闸门要防的事。
2. **conv 内核 14.15× 但源仓 e2e 0.995×**：单核快 14 倍不等于一步快。所以它保持 opt-in，
   `measured_ratio` 记 op 级并在 `ratio_note` 里写明 e2e 无收益。谁要把它转默认，必须先在
   平台上拿出端到端 ≥1.10× 的 A/B。

## 清单（`src/ascend_attention/manifest.json`）当前内容

平台清单只收「有 `measured_ratio` + `bench` 且默认启用」的实现：现在 **19 条**，其中 12 条是
`torch` 基线（比值 1.0，含义是"有这个 op、与厂商路径持平"，不是性能声明），1 条是真加速
（`vendor.deep_select` 3.8432×）。默认关的实现（decode 的两个、conv/chunk_wu 的内核、prefill 内核）
**不进清单**，但都出现在完整清册里，带 `graph`/`requires`/`note`：

```
python -m ascend_attention                 # 清单（可用子集）
python -m ascend_attention --capabilities  # 清册（含默认关的实现与原因）
```

两者的分工与"decode 为什么不在清单里"见 `docs/contracts/exposure.md`。

## P2c（权重整形 + 基线自身取证）

- `ops/gdn/weights.py`：`fuse_inproj_weight` / `split_interleaved` / `reorder_rows_plan` /
  `apply_row_order` 四个纯函数。两种独立推导（循环 vs reshape）在 6 组头配置上逐位一致，
  GEMM 列置换 `torch.equal` 成立（**位级**等价，不是近似）。
- `tests/test_fla_ops.py`：`ops/fla` 的 6 个基线算子对 fp64/oracle 取证——conv（decode + prefill 两种
  形态，含状态写入）、`l2norm`、`chunk_local_cumsum`、`kkt`（对定义式）、`solve_tril`（对
  `torch.linalg.inv`）、`wu`（对 fp64 matmul），以及 strictly-lower 守卫。
- 顺手挖到一个**会静默出零**的坑：上游 `null_block_id` 默认 0，第 0 行被整段跳过。已修 planner 并加
  测试，记入 `docs/pitfalls.md` 第 1 条。

## P3（MoE：判决后只迁两个基线）

- 迁入 `moe_slot_layout`（compact→slot 布局，纯 torch、零主机同步）与 `moe_prenz_pack`
  （ND→compact-NZ 权重打包）。两者都与另一份独立实现/往返语义对齐：
  `moe_slot_layout` 与 `testing/moe_reference.build_slot_layout`（循环写法）在 compact→slot 映射上
  **逐项一致**；`prenz` pack→unpack 逐位复原且物理序对公式。
- 不迁的四样写进 `docs/archive-moe.md`：`grouped_gemm` **0.86×**（负）、`down_prenz` 结果错（~1e36）、
  `align_*` 未到门槛且依赖 c_api、`shared_expert_fold` 只有 [待复核] 单窗数字。
- 两个 op 的 `measured_ratio` 是**定义性的 1.0**（唯一实现、无可比对象），账本只记绝对延迟。

## P4（采样：第一个 vendor_call，且是本仓第一个正收益入清单）

- `sampling_topk_threshold`：语义是"保留 logit ≥ 第 k 大的 token"（阈值式 top-k，**保留并列**）。
  - `torch.topk_threshold`（基线，1.0）：`topk` 取阈值再 mask。
  - `vendor.deep_select`：**实测 3.8432×**（B=8、V=152064、k=20、bf16，6 对交替，时间比 0.26；
    `bench/results/sampling_topk_b8.json`）。**已默认启用并入清单**——这是清单里唯一一个非基线条目。
- 顺带补了两道原先没有的闸门：
  - **设备闸**：`ImplSpec.device` 不属于 `{"any", 当前画像}` 直接拒绝（否则 CPU 机器会路由到只有 950
    才有的实现）；
  - **基线最后尝试**：候选按比值降序、torch 基线排最后，注册顺序不再决定性能选择。
- `sampling_greedy` 也是本轮落地：`torch.argmax` 基线（定义性 1.0），是 `do_sample=False` 的分流目标；
  分流只有一条规则 `sampler_for(do_sample)`，不把布尔塞进 `ShapeSpec`。
- `sampling.fused_sample` **不迁**，理由与前置条件写进 `docs/archive-sampling.md`（零消费者、无端到端数字、
  `vocab % 128` 约束、`ub_cap=8192` 触发 507035）。
- 顺带修掉一个真 bug：`accepts_shape` 只在传了 `shape=` 时才查 dtype，于是 `plan(..., dtype="float32")`
  会选中 bf16-only 的 vendor 实现，然后在 planner 里抛。现在 dtype 门与 shape 无关。
- 并列语义记进 `docs/pitfalls.md`：bf16 词表 152k 必有碰撞，vendor 与 torch 可能选到**不同下标、
  相同数值**；用下标集合做判据会误报（第一次就是这么误报的）。
## 批次完成情况（P0–P5）

| 批 | 内容 | 状态 |
|---|---|---|
| P0 | 骨架 + 三入口 + fp64 oracle + `capture_guard` | ✅ |
| P1 | attention：decode（设备对数、真图捕获）、prefill 语义、布局声明 | ✅（prefill 编排按边界不迁，见 `archive-attention.md`） |
| P2 | GDN 全链 + 四个"静默杀手"契约 | ✅ |
| P3 | MoE：两个基线 + 四样归档 | ✅ |
| P4 | 采样：vendor_call（唯一正收益）+ greedy + 分流规则 | ✅（`fused_sample` 归档） |
| P5 | 清单/账本/反向接入/绑定覆盖/审计 | ✅ |

逐条验收与"没做"清单在 `docs/completion-audit.md`；反向接入的 off/on 证明在
`docs/reverse-integration.md` §6。

## attention 族的归档（`docs/archive-attention.md`）

- `page_pipeline`：TileInfer 侧**没有任何 A/B 记录**（`grep PAGE_PIPELINE docs/` 为空）→ opt-in、默认关，
  通过 `utils.kernel_flag("PAGE_PIPELINE")` 读取，一次 A/B 即可翻案。
- `simd_softmax`：TileInfer 记 ~2.6×，但是**单窗探针**不是交替对 → 本仓 `measured_ratio` 仍为 `None`。
- `split_kv`：3× 但两次把设备卡死（vector-core exception）、25 分钟编译扫描未收敛 → **连候选都不注册**。
- `paged_decode.py`（910B fork）：死代码，留作对照物与共享设施的测试对象。
