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

## 账本（`bench/results/*.json`）

| 文件 | 内容 | 结论 |
|---|---|---|
| `decode_b8_s2048_d128.json` | B=8 / S=2048 / d=128 / group=8，vs `npu_fusion_attention`（dense KV） | **0.362×，慢 2.76×**；与源仓自述「TPOT 2.53×」一致，方向不矛盾 |
| `gdn_conv_c4096.json` | C=4096 / L=8 / W=4 / bf16，vs `causal_conv1d_update` | **14.15×**（0.0168 ms vs 0.272 ms），输出相对误差 4.1e-3 |
| `gdn_chunk_wu_t256.json` | T=256（2×128）/ HV=4 / D=128，vs `solve_tril`+`recompute_w_u` | **14.55×**（0.167 ms vs 2.600 ms），w/u 相对误差 1.3e-3 |

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

平台清单只收「有 `measured_ratio` + `bench` 且默认启用」的实现。于是现在清单里只有两个
**torch 基线**条目（`gdn_gating`、`gdn_conv_update`，比值 1.0），含义是「本仓有这个 op，但
没有可用的加速实现，用它与厂商路径持平」——不是性能声明。两个内核都因为「不默认」而不出现。

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
- 并列语义记进 `docs/pitfalls.md`：bf16 词表 152k 必有碰撞，vendor 与 torch 可能选到**不同下标、
  相同数值**；用下标集合做判据会误报（第一次就是这么误报的）。
- P3：MoE 三个 op（`slot_layout` 2.569×、`prenz` 负结果归档、`shared_fold`），先判决接不接。
- P4：采样（`vendor_call` 形态）。
- P5：端到端账本 + TileInfer 反向接入等价证明 + 图捕获守卫审计。
