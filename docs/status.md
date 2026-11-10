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

## 下一批

- P2b 已完成：`gdn_chunk_wu` 设备数值（含尾块 96/32）+ A/B 14.55×，仍为 opt-in。
- P2c：权重整形纯函数、`fla` 链的 CPU 数值契约（fp64 oracle）、`gdn_gating` 与 `fla.gating` 去重。
- P3：MoE 三个 op（`slot_layout` 2.569×、`prenz` 负结果归档、`shared_fold`），先判决接不接。
- P4：采样（`vendor_call` 形态）。
- P5：端到端账本 + TileInfer 反向接入等价证明 + 图捕获守卫审计。
