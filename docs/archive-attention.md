# 归档：attention 族里不默认开的三条路与一份死代码

按"负结果与死代码只归档不进主干"的规则，这些不进 manifest 的默认集，代码留在仓里、理由留在这里。
**归档不是删除**：每一条都写清"什么时候可以翻案"。

## 1. page-pipeline（双缓冲页循环，`ASCEND_ATTN_PAGE_PIPELINE=1`）

- **是什么**：`kernels/attention/paged_decode_ascend950.py:paged_decode_pipeline`，把页循环做成
  双缓冲，让 GM→UB 的搬运与计算重叠。
- **证据**：**无**。TileInfer 仓库里 `TILEINFER_PAGE_PIPELINE` 只出现在内核注释与读取处，
  `docs/` 与 `docs/notes/` 都没有它的 A/B 记录（`grep -rn PAGE_PIPELINE docs/` 为空）。
- **判决**：保持"存在、opt-in、默认关"。它已经通过单一入口 `utils.kernel_flag("PAGE_PIPELINE")`
  读取，因此**不需要改内核**就能被一次 A/B 打开。
- **翻案条件**：≥4 对交替 A/B，且比值 ≥1.0；届时要么升为默认，要么进 manifest 作候选。

## 2. SIMD softmax（`ASCEND_ATTN_SIMD_SOFTMAX=1`）

- **是什么**：把 per-page 在线 softmax 从 SIMT reducer 换成 `T.simd.*` 打包形式。
- **证据**：TileInfer 记 decode 形状 **~2.6×**，设备探针在
  `benchmarks/probes/simd_softmax_*_device.py`（**单窗探针，不是交替 A/B**）。
- **判决**：opt-in，默认关。本仓不接受单窗数字（`docs/perf-rules.md` §3），所以它的
  `measured_ratio` 仍是 `None`。
- **翻案条件**：在本仓 `bench/bench_paged_decode.py` 里加一组 `SIMD=0/1` 交替对，取中位数。

## 3. split-KV（`TILEINFER_ALLOW_SPLIT_KV` → 现为 `ASCEND_ATTN_...`，且仍是硬编码门）

- **是什么**：长 KV 时把 KV 维切分再 merge，缓解单核 starving。
- **证据**：TileInfer 记"starved 形状 3×"，**但**同一进程连续编译并运行多个形状时两次把设备卡死
  （`ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`），单形状不可复现、20 次连续启动也不复现。
  本仓 P1 期间试过一次 25 分钟的编译扫描，**没有收敛**，因此没有任何可用的比值。
- **判决**：**不进 manifest**（连 `enabled=False` 的候选都没注册）：一个会挂死设备的路径，
  在"默认关"和"可用"之间没有中间态。
- **翻案条件**：先能稳定复现并解释那次 vector-core 异常；在此之前不谈性能。

## 4. `kernels/attention/paged_decode.py`（非 950 的 fork 变体）

- **状态**：**死代码**。`backends/tilelang_ascend.py` 是唯一引用者，而该后端对应 910B，本阶段不迁移。
- **为什么留着**：它是 `abi.pool_view`/`ShapeSpec` 这些共享设施的第一个使用者，删掉会让
  `tests/test_abi.py` 的一部分失去被测对象；且它带着 `paged_decode_ascend950.py` 之前的历史修复
  （`null_block_id`、尾页掩码），是排障时的对照物。
- **判决**：不进 manifest，`graph` 记为 `unknown`，不宣称任何性能。

## 5. 可复用的 prefill plan（不是缺口，是定义）

`tilelang.prefill950` 已接线：`plan_prefill` 用本仓 `plan_query_tiles_from_host` 生成本步的
schedule（`qo_lens * group` 行，M tile 是扁平 `(q_pos, group)` 行），`run()` 走 `forward_prefill`；
上卡与 `torch.reference` 一致，append 的 `causal_shift = kv_len - qo_len` 由 row-threshold 实现，
测试里用"改动某 token 之后的 KV 不影响该 token 输出"验证。

所谓"prefill 编排未迁移"的真实含义只有一条：**一个 plan 的 tile 数就是这一步的 prompt 长度**
（`Σ ceil(qo_len_b·group/block_q)`），所以 plan 不能跨步复用。这不是没接线，也不可能靠加字段解决：
若要跨步复用，就得让 plan 携带一个"最大 tile 数"的 schedule 并在每步用设备侧操作重填
`q_len/seq_ids`（`forward_prefill` 的 pack 已经是设备侧索引驱动的，具备这个条件），代价是每步仍要
一次 host→device 的 schedule 搬运。当前选择是不做：平台的 prefill 路径本来就每步重建 schedule，
而一个"看起来可复用、实际靠 padding 掩盖"的 plan 比明确的步级 plan 更容易被误用。

没有厂商 prefill 的 A/B，所以 `measured_ratio=None`；给它写 1.0 会让它进 manifest，等于声称
"和厂商一样快"——那是编的。

