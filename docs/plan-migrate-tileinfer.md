# 把 TileInfer 的算子特性迁入 AscendAttention：识别、切分与分批方案

> 性质：方案稿，供评审。范围只覆盖**算子（implementation + contract + evidence）**与验证设施；
> 平台语义（worker、图捕获、KV 分配、批元数据、调度、模型实现）不在本文范围。
>
> 依据：
> - 本仓 `README.md`（定位与硬约束的唯一来源）
> - `/home/userlist/zxs/TileInfer` @ `339969a`（2026-11-10），及其 `docs/status.md`、`docs/architecture.md`、
>   `docs/plan-vllm-native-backend.md`、`docs/extending.md`、`docs/notes/{perf-candidates,perf-comparison-rules,fla-chunk-contract,op-coverage-qwen36}.md`、
>   `docs/known-issues.md`
> - 对 `tileinfer/kernels/**`（18 模块 / 6413 行）、`tileinfer/{attention,plan,metadata,testing}`、
>   `tileinfer/vllm/ops/**`、`legacy/**` 的逐文件盘点（行号即 `../TileInfer` 内行号）

---

## 0. 结论（先看这六条）

1. **要迁的是三样东西，不是一次搬家**：算子实现（代码）、算子契约（形状/dtype/布局/不变量）、
   算子证据（实测比值、负结果、踩过的坑）。第二、三样比第一样值钱——第一样（代码）实际只有两三个文件在承重，其余是实验与死代码。
2. **6413 行里真正承重的只有 2 个在役文件**：`attention/paged_decode_ascend950.py`（decode+prefill，
   默认在役）和 `gdn/conv_update_ascend950.py`（opt-in，端到端 0.995）；另有 1 个备选
   （`attention/paged_decode.py`，fork 工具链变体，本机跑不起来）。
   `gdn/chunk_wu_ascend950.py`、`sampling/fused_sample.py` 有强证据但未接线；
   `moe/grouped_gemm_ascend950.py` 是**已记录的负结果（0.86×）**；
   `gdn/{proj_fusion,reorder}.py` 根本不是内核，是 vLLM 模型层的 monkey-patch。
   **按文件平铺搬过来，会把负结果和死代码变成新仓库的技术债。**
3. **不能照抄的只有三处结构**：(a) 分派门现在是 callable predicate，本仓要求"纯数据清单"→
   必须改成声明式 Spec；(b) 现在只有 attention 有 plan/run，本仓把 query/plan/execute 作为全库接口
   → 要给 kernel/stateless 算子补 Plan 协议；(c) 内核里有 5 类反向依赖（vLLM 模型层、
   `tileinfer.serving`、`tileinfer.plan`、`tileinfer.utils`、`deep_select`）→ 必须反转。
   其余（`CandidateSet` 的门序、oracle 分层、bench 纪律、编译键规则）可原样继承——那是 TileInfer 最硬的部分。
4. **顺序按"契约复杂度 × 证据强度"排**，不按 README 的营销顺序：P0 三入口 + attention 合同层（纯 CPU 可验）
   → P1 attention 内核（首个"编译内核"形态）→ P2 GDN（契约最复杂但基线是 torch）→
   P3 MoE（先判决接不接，负结果只归档）→ P4 采样（首个"直调厂商算子"形态）→ P5 账本 + 反向接入。
5. **设备就在这台机器上（102，Ascend950PR_9579），但 agent 的默认 shell 看不到它**：
   agent 沙箱用 tmpfs 覆盖了 `/dev`（只剩 14 项，无 `davinci*`），于是 `npu-smi` 报 dcmi `-8005`、
   `torch.npu.is_available()` 为 False、设备相关 import 卡住——这三条都是**沙箱假象**，不是环境问题。
   已验证：full-access shell 下 `/dev/davinci0` 存在，`torch.npu.is_available()=True`、
   设备名 `Ascend950PR_9579`、npu 小算子可跑、`tilelang 0.1.15` + `tilelang.ascend` 可导入（0.5 s）。
   **纪律**：设备窗一律在 full-access 下执行，且每次先跑 canary（npu 小算子 + 一次内核编译），
   canary 失败时先怀疑沙箱而不是环境；P0 仍保持"纯 CPU 全绿"，因为它是回退路径的正确性基线。
6. **工作量**：单人力约 11 周（P0 5d、P1 12d、P2 12d、P3 8d、P4 10d、P5 8d）；两人并行约 6 周。

---

## 1. 本仓库定位，以及它给出的硬约束

README 不是宣传稿，是接口契约。逐条翻译成可验收的约束，是后面所有取舍的判据：

| README 的句子 | 落地约束（会写进测试） | 怎么验 |
|---|---|---|
| 引擎无关，只提供算子，不提供平台 | 包内**零** `import vllm` / `vllm_ascend`；不注册任何 entry point；不提供 console script | AST + `sys.modules` 双重守卫测试（照抄 `tests/test_no_vllm_ascend.py` 的写法） |
| 对外只暴露**能力查询 / 规划 / 执行** | 公共 API 只有三个动词；其余模块默认私有 | `__all__` 白名单测试 + 文档只写这三个 |
| 入参只接受张量与纯数据 | 不接收任何引擎对象（`RaggedMetadata`、`TileSchedule`、`model_config`） | 类型守卫测试：public API 用假引擎对象调用必须报错 |
| 每个实现声明**能否进图**、**支持哪些 KV 布局** | 每个 `ImplSpec` 必须带 `graph` 与 `layouts` 两个字段，且**不可为空** | schema 校验：缺字段的注册直接失败 |
| 分派必须廉价，不在调用时测速 | `select()` 零分配、零张量值读取、零同步 | 用"读值就抛"的假张量钉住（`tests/test_op_candidates.py` 已有同款） |
| 两种产物流转并存 | `form ∈ {torch, tilelang, ascendc, vendor_code, vendor_call}`，其中 `torch` 形态允许没有编译产物 | manifest 里每族至少各有一次这两种形态 |
| 依赖方向单向：平台层 → 本仓 | 本仓不得出现任何平台名（`vllm`、`worker`、`block_table`、`aclgraph`） | 关键字黑名单 grep + 评审 |
| 平台未安装本仓时应能回退厂商原生算子 | 每族**必须**有一个可用的 `torch`/`vendor_call` 基线，它不是测试夹具，是产品的一部分 | 每个算子至少一个 baseline 实现，且 baseline 永不依赖设备 |
| 以**纯数据清单**暴露自己 | `manifest()` 返回可 JSON 序列化的 dict；**不包含 callable** | `json.dumps(manifest())` 必须通过；CI 对比快照 |
| "用哪一家"的策略不在本仓 | 库不做端到端择优、不读模型配置、不做 bucket 策略 | 评审 + 依赖检查 |
| 非目标：设备语义/KV 分配/批元数据/图捕获/worker/量化/MLA/多卡 | 库**不持有**状态、不申请 KV、不建 stream、不 capture | API 签名里不出现 allocator/stream/config |
| 昇腾单卡：算子族已实现并在役 | 迁移不是从零，是"把已经在跑的东西换归属地"——所以要保 blame、保证据 | 见 §4"历史保留" |

**定位的一句话**：本仓是算子**供给侧**，成功标准不是"能跑通一个模型"，而是
**平台只靠清单就能决定"用不用、用哪个"，并且拒绝得可预测**。
推论三条，容易被忽略：

- 清单必须能被**没有 torch 张量**的人读懂 → 能力声明必须是数据，不能是 lambda。
- "拒绝"必须发生在**平台组批之前**，而不是第一次 forward 时 → 能力查询要能只凭 shape/dtype/layout 回答。
- 回退路径是产品 → `torch` 形态要与内核形态**逐位可比**（同 ABI、同 dtype 策略），否则"回退"等于换了个算法。

---

## 2. TileInfer 是什么（识别结果）

### 2.1 一句话与分层

TileInfer 是"把 vLLM 0.23 的昇腾后端整个重写一遍"的项目：它能起服务、出正确结果、跑混合 KV，
但性能未打平（TPOT 2.53×、TTFT 11.8× 于 vllm-ascend 基线），图捕获未接上
（`docs/status.md:10-13`）。它的分层正好决定哪些能搬：

```
tileinfer/vllm/      平台层：platform plugin、worker、attention backend、graph bridge   ← 不迁
tileinfer/kernels/   TileLang 算子 + 少量 torch 组合                                   ← 迁（挑）
tileinfer/attention/ 引擎无关的 plan/run、metadata、参考后端、oracle                    ← 迁（几乎原样）
tileinfer/vllm/ops/  torch 基线实现 + CandidateSet 分派 + 装载器                        ← 拆：基线迁，装载器不迁
tileinfer/serving/   分相计时、bucket、planner（引擎无关但偏平台）                       ← 只迁 planner 的键规则
tileinfer/testing/   fp64/torch oracle                                                 ← 迁（这是资产）
reference/vllm-0.23/ vLLM 源码逐字副本，只作行号锚点                                    ← 不迁
legacy/              冻结的 vllm-ascend 时代补丁层                                      ← 不迁（但 op 契约表可抄）
```

### 2.2 资产盘点（按"迁不迁"分类）

| 资产 | 规模 | 属性 | 判决 |
|---|---|---|---|
| `tileinfer/kernels/attention/paged_decode_ascend950.py` | 3227 行 | decode / split / merge / prefill / prefill-split 六个 TileLang 内核 + 6 个 torch 组合入口 | **迁**，但只迁内核 + 两个组合入口；其余按 §2.4 L1 / §3.3 重写签名 |
| `tileinfer/kernels/attention/paged_decode.py` | 344 行 | 同一算子的 fork 工具链变体（`target="pto"`），本机不可用 | **不迁**；作为"同算子多形态"的第二个形态在 P5 之后评估 |
| `tileinfer/kernels/attention/kv_nz_layout.py` | 72 行 | 纯 torch 布局数学（ND↔NZ fractal） | **迁**（它同时是本仓"KV 布局能力"的第一个真实样本） |
| `tileinfer/kernels/gdn/conv_update_ascend950.py` | 181 行 | TileLang，conv+bias+SiLU+state shift，唯一"import 安全"的内核模块 | **迁**（opt-in 形态的样板） |
| `tileinfer/kernels/gdn/chunk_wu_ascend950.py` | 379 行 | TileLang，solve_tril + recompute_w_u 融合。**内核本身没有记录比值**；它取代的方法（solve_tril 的 Newton 倍增求逆）孤立 45.6×、整链 1.91×、端到端 1.04×（`perf-candidates.md:26`） | **迁**（作为候选，默认关） |
| `tileinfer/kernels/gdn/{proj_fusion,reorder}.py` | 294 行 | **不是内核**：vLLM 模型层的 monkey-patch（`torch.ops.vllm.qwen_gdn_attention_core`、`_encode_layer_name`） | **不迁代码**，只把"权重行序重排 + 两个瘦 GEMM 合成一个"写成算子级契约条目 |
| `tileinfer/kernels/gdn/__init__.py` | 52 行 | torch 组合的门控（softplus·sigmoid），带 fp64 oracle | **迁**（第一个垂直切片用它，最便宜） |
| `tileinfer/kernels/moe/grouped_gemm_ascend950.py` | 480 行 | TileLang 分组 GEMM，**0.86× 负结果** | 代码归档分支出，主干只留 manifest 条目与 `note` |
| `tileinfer/kernels/moe/align.py` | 348 行 | TileLang + 裸 CANN c_api（preNZ 直载），门槛 ≥1.0× 未达 | 同上级：P3 再判决 |
| `tileinfer/kernels/moe/{prenz,slot_layout}.py` | 151 行 | 纯 torch 布局/槽位数学 | **迁**（无设备、可 CPU 验证，是 MoE 的契约底座） |
| `tileinfer/kernels/sampling/fused_sample.py` | 616 行 | TileLang 三分位二分采样，**没有 build/cache 包装**，无消费者 | P4 迁"内核源码 + 补 builder"，之前不动 |
| `tileinfer/kernels/sampling/topk.py` | 187 行 | **直调厂商算子**（`deep_select.topk`）+ 窗口门，无消费者 | P4 迁（本仓 `vendor_call` 形态的首个样例） |
| `tileinfer/attention/{api,metadata,plan}.py` | 1406 行 | 引擎无关的 plan/run、元数据模型、LPT 调度、workspace | **迁**，这是 P0 的主体 |
| `tileinfer/attention/backends/{base,reference}.py` | 414 行 | 后端注册表 + torch oracle 后端 | **迁**（注册表要按 §3.2 重塑） |
| `tileinfer/attention/backends/tilelang_ascend950.py` | 358 行 | 把 plan 接到内核上的胶水（含 `declared_pool_cap`/`pool_view` 用法） | **迁**，去 engine 化后作为 `impl` 的适配层 |
| `tileinfer/testing/*` | 743 行 | attention/GDN/MoE/sampling 的 fp64 与 torch oracle | **迁**（与算子同批，不许后补） |
| `tileinfer/vllm/ops/{fla,moe,gdn,fused_moe_oot}.py` | ~6.7k 行 | torch 基线实现（很多是逐位复刻的 triton 重写） | **按需迁**：只迁"某算子需要 torch 基线"的那些函数 |
| `tileinfer/vllm/ops/candidates.py` | 440 行 | 分派门 + CallSpec + 报告 | **迁并改造**（§3.2） |
| `tileinfer/vllm/ops/{install,switch,families}` | ~1200 行 | 装载器/开关/按 vLLM 名绑定的族表 | **不迁**（这是平台侧，正是"寄生适配器"要长出来的地方） |
| `tileinfer/serving/{planner,buckets}.py` | 646 行 | plan_key 规则、桶与池容量量化 | 只迁**规则**（进本源码），planner 本体留平台 |
| `tileinfer/serving/{phases,step,counters,flags}.py` | ~741 行 | 分相计时、文件开关 | 不迁（观测属平台），但 `flags` 的"文件开关"形态可借 |
| `tileinfer/vllm/{platform,worker,graphs,attn,models,profiles,doctor,compat}` | ~15000 行 | 平台层 | 不迁 |
| `legacy/**` | ~5945 行 | vllm-ascend 时代补丁层，18 个 op 的接管契约与守卫 | 不迁代码；**契约表与守卫条件是素材**（§7 引用） |
| `docs/*`（perf rules、known-issues、fla-chunk-contract、op-coverage） | ~2100 行 | 方法论 + 契约 + 坑 | **迁**（改成新仓库的 ADR 与算子契约页，见 §5 P0） |

### 2.3 算子清单与证据分级

等级按 TileInfer 自己的口径（`docs/status.md:47-57`）：**[核实]** 有落盘日志且可复现、**[待复核]** 单窗测量、
**[作废]** 已被推翻。本仓只接受 [核实] 进默认。

| 族 | 算子（已注册名） | 形态 | 设备 | 证据 | 现在状态 |
|---|---|---|---|---|---|
| attn | `paged_decode_attention` | tilelang | 950 | 设备 fp32 oracle 对齐（ragged/尾页/乱序页表/d=128,256）[核实]；**vs 厂商 FIA 0.362×**（负）[核实] | **已迁，不默认**（`tilelang.decode950`） |
| attn | `paged_prefill` | tilelang + torch 打包 | 950 | 源仓占比 1.000，本仓未复核 | **未迁**（950 后端对 prefill 抛 `NotImplementedError`） |
| attn | `paged_decode_split` + `merge` | tilelang | 950 | "batch 1 上 3×" [待复核]；本仓扫形 25 min 未收敛，未采信 | opt-in（`TILEINFER_ALLOW_SPLIT_KV`），未评估 |
| attn | `simd_softmax` / `page_pipeline` 变体 | tilelang | 950 | [待复核] / [作废] | 不迁（负与已知回归） |
| layout | `kv_nz_layout` pack/unpack | torch | 无 | 只有往返语义 | 随内核迁入（`kernels/attention/kv_nz_layout.py`） |
| gdn | `gdn_gating` | torch | 无 | vs fp64 oracle 1e-5 [核实] | **已迁，默认（基线，比值 1.0）** |
| gdn | `gdn_conv_update` | tilelang | 950 | 本仓 **14.15×**（op 级）[核实]；源仓端到端 0.995 [核实] | **已迁，opt-in** |
| gdn | `gdn_chunk_wu` | tilelang | 950 | 本仓 **14.55×**（vs 它取代的两个算子）[核实]；源仓孤立 45.6×、整链 1.91×、端到端 1.04× | **已迁，opt-in** |
| gdn | 链上四算子（kkt / solve_tril / delta_h / chunk_o） | torch | 无 | fp64 oracle TOL 3e-2 | **已迁**（`ops/fla/`，5000 行纯 torch，零 triton/vllm 依赖） |
| gdn | `inproj_fusion` / `inproj_reorder`（权重整形） | torch | 无 | 逐位等价 [核实] | 未迁（P2c） |
| moe | `moe_slot_layout` | torch | 无 | 结构性测试 | 未迁（P3，待判决） |
| moe | `moe_prenz_pack` / `moe_down_prenz` | torch / tilelang | 950 | 往返语义 / **结果错 ~1e36** [核实，负] | 归档，不迁 |
| moe | `moe_grouped_gemm` | tilelang | 950 | **0.86×** [核实，负结果] | 归档，不迁 |
| moe | `moe_shared_expert_fold` | torch（权重拼接 + 路由改列） | 无 | ~1.2 ms/step [待复核] | 未迁（需先补 A/B） |
| sampling | `sampling_topk_threshold` | **vendor_call**（`deep_select`） | NPU | 时间比 0.39/0.25/0.23 [待复核] | 未迁（P4） |
| sampling | `sampling_fused_sample` / `sampling_greedy` | tilelang / torch | 950/NPU | 2.88 ms 恒定 vs sort 0.98→4.44 ms [待复核] | 未迁（P4） |

**读法**：第一批只需 4 个算子（`paged_decode_attention`、`gdn_conv_update`、`gdn_chunk_wu`、
`gdn_gating`）就已经覆盖 `torch` 与 `tilelang` 两种形态、覆盖"默认"与"opt-in"两种默认值，并且
**当场撞上了一个负结果**（decode 0.362×）——这不是意外，是这套判据本来就会给出的东西：
源仓自己的 `docs/status.md` 写着 TPOT 2.53×、门是 1.10×。账本见 `docs/status.md`。

**AscendC 形态**：用户已判定不在本次迁移范围内，本表不再预留行；`spec.FORMS` 里的
`ascendc` 只是保留字符串，不是承诺。

### 2.4 必须反转的 5 类依赖

盘点里 18 处"引擎概念泄漏"可以归成 5 类，每类的动作都是**反转依赖方向**，而不是删 import：

| # | 泄漏 | 位置 | 反转动作 |
|---|---|---|---|
| L1 | 内核组合入口吃引擎元数据对象（`RaggedMetadata`）与调度对象（`TileSchedule`） | `paged_decode_ascend950.py:2240,2399,3154`；`from tileinfer.plan import TileSchedule`（`:2652`） | 拆两层：**编译内核只吃张量**（现状已如此），**组合入口改成吃 `Plan`**（§3.3）；`TileSchedule` 下沉进本仓 `plan.py` |
| L2 | 内核反向 import 服务层做存储重绑（`from ...serving.buckets import capacity_view`，`:3145`） | 同上 | `capacity_view` 与 `declared_pool_cap`/`pool_view` 一起下沉成库内 `abi.py`（池容量量化是**内核 ABI 问题**，不是 serving 问题） |
| L3 | 直接 import vLLM 模型层（`_encode_layer_name`、`torch.ops.vllm.qwen_gdn_attention_core`） | `gdn/proj_fusion.py:133-137`、`gdn/reorder.py:137-141` | 不迁这两个文件；只把纯函数（`reorder_rows_plan`、`split_reordered`、`build_fused_inproj_weight`）搬成"权重整形的算子"，caller 传形状参数 |
| L4 | 依赖 `tileinfer.utils.have_tilelang`、进程级环境变量（`TILEINFER_POOL_CAP`/`TILEINFER_SIMD_SOFTMAX`/… 共 8 个） | `paged_decode.py:52`、`paged_decode_ascend950.py:79,84,95,3093,3102` | 环境变量改成**显式编译规格参数**；可用性探测下沉 `device.py`；库内不读环境（只允许读 `ASCEND_ATTN_*` 的一个总开关，见 §6 D7） |
| L5 | 依赖厂商/第三方包（`deep_select`、torch_npu 私有 op） | `sampling/topk.py:122` | 变成显式 `provider` 依赖：`impl` 声明 `requires={"deep_select>=x"}`，缺失时该实现**不可见**而不是抛错 |

顺带一条"分层错误"的清理：`sampling/topk.py` 的 k 标量缓存按 **vLLM 采样对象的身份**做键
（`:62-69,88-100`），其正确性论证依赖引擎每批重建对象。迁入后必须改成 caller 传入的
`(k_tensor, vocab, batch_signature)` 显式键，否则在别的引擎里会静默用错 k。

### 2.5 与 README 承诺的落差（必须承认的覆盖缺口）

| README 承诺 | TileInfer 实际 | 缺口怎么补 |
|---|---|---|
| 注意力 paged decode/prefill（GQA、head_dim 256、K=V=128） | 内核只有一个 `dim` 参数且 `dim ∈ {128,256}`（`paged_decode_ascend950.py:142,397`），即 **Q 与 K/V 必须同宽** | 覆盖 "head_dim 128/256 且 q/kv 同宽"；**Q=256 而 K/V=128 的非对称组合没有实现**——要么新内核变体，要么改 `head_dim` 参数化成 `(dq, dkv)`。这是 P1 之后第一个真实的新需求 |
| 注意力 "head_dim 256" | dsv41 是 `head_dim=512`、`qk_rope=64`、`nope=448`（`notes/dsv41-facts.md:38-39,98`），MLA 明确不做 | 本仓只承诺 GQA 路径；MLA 保持非目标 |
| GDN：卷积/状态更新/分块递推/chunk 链 | 四个都有（conv_update 内核、recurrent torch、chunk 链 torch、chunk_wu 融合内核） | 覆盖；chunk 链的内核化只做到 `solve_tril+w_u` 一个融合点 |
| MoE：路由/对齐/专家计算/**共享专家折叠** | 路由是 torch，对齐与专家计算有内核（0.86×），共享专家折叠只在 `legacy/`（`shared_expert_fold.py:81-118` 是纯函数，可搬） | "折叠"要重写成算子（权重拼接 + 路由列扩展），P3 做 |
| 采样 greedy/top-k/top-p | greedy 在 legacy；top-k 走 `deep_select`；top-p/top-k 融合内核无 builder | 三者形态齐了，但只有 vendor_call 有证据 |
| 多形态 + 库内分派 | `CandidateSet` 只覆盖"vLLM ABI 名 + callable predicate" | §3.2 改造 |
| AscendC 内核形态 | **仓库里没有任何 AscendC 源码**（legacy 也没有） | 形态字段先留空位，P4 之后单独立项，不在本方案承诺 |
| 其他国产卡 | 无 | 只留 `device.py` 的声明位 |

---

## 3. 三处结构性改造

### 3.1 命名与 ABI：从"vLLM 名字"到"算子名 + 纯数据签名"

TileInfer 的族表是"**vLLM 的 ABI 名字 → 我们的实现**"（`families/gdn_moe.py:211-500`，
15 个 `OpBinding`）。本仓必须反过来："**算子名 + 形状/dtype/布局 → 实现**"。
名字表属平台，不进本仓；但迁移时要有一张对账表，否则搬漏：

| TileInfer 的 vLLM ABI 名 | 本仓算子名 | 备注 |
|---|---|---|
| `chunk_gated_delta_rule` | `gdn.chunk_chain` | 4 段链，对外可按段调用或整链 |
| `fused_post_conv_prep` | `gdn.post_conv` | |
| `causal_conv1d_fn` / `causal_conv1d_update` | `gdn.conv1d` / `gdn.conv_update` | 后者是本仓第一个 kernel 形态 |
| `fused_sigmoid_gating_delta_rule_update` / `fused_recurrent_*` | `gdn.gating` / `gdn.recurrent` | |
| `chunk_local_cumsum` / `l2norm_fwd` | `gdn.chunk_cumsum` / `gdn.l2norm` | |
| `topk_softmax` / `moe_align_block_size` / `fused_experts` / `unquantized_fused_moe` | `moe.router` / `moe.align` / `moe.grouped_gemm` / `moe.fused_mlp` | |
| `silu_and_mul` | `dense.swiglu` | 本仓 `dense` 族先只放它一个 |
| （attention 无 ABI 名，走 backend 接口） | `attn.paged_decode` / `attn.paged_prefill` | 走 plan/run，不走"函数名接管" |
| （采样无 ABI 名） | `sampling.topk` / `sampling.sample` / `sampling.greedy` | 走 `vendor_call` 与 `tilelang` |

规则三条：
1. **算子名是稳定键**，实现名是二级键（`attn.paged_decode` / `tilelang.ascend950.decode.v1`）。
2. **ABI 只描述张量**：`(name, shape, dtype, layout, optional)` 的列表，写成数据，可 JSON 化。
3. **可选参数不进 ABI**（例如 "是否 fp32 累加" 属于编译规格，不是调用参数）。

### 3.2 能力声明：从 callable predicate 到 Spec + 清单

`CandidateSet` 的四道门是对的，必须继承（`ops/candidates.py:23-34`）：

```
① 全局熔断（只能关） → ② enabled_by_default → ③ measured_ratio ≥ 阈值 → ④ predicate
```

问题只在第 ④ 道：本仓要求"纯数据清单"，而 predicate 是 lambda。
**改造**：把 predicate 降级为"ShapeSpec 的求值"，并保留一个可选的 `extra_guard`（只能是纯数据判断）。

```python
# ascend_attention/spec.py
Dim = int | tuple[str, object] | None      # ("mul", 16) / ("in", (128, 256)) / ("min", 32) / None = 任意

@dataclass(frozen=True)
class ShapeSpec:
    ndim: int
    dims: tuple[Dim, ...]
    dtypes: frozenset[str]                  # {"bf16","fp16"}
    contiguous: bool = True
    def matches(self, shape, dtype, contiguous) -> bool: ...   # 只读元数据，无张量运算

@dataclass(frozen=True)
class ImplSpec:
    op: str
    impl: str
    form: Literal["torch","tilelang","ascendc","vendor_code","vendor_call"]
    device: str                              # "ascend950" / "cpu" / "any"
    shape: ShapeSpec
    layouts: frozenset[str]                  # KV/权重布局：{"NHD"} / {"NZ"} / {"ND","NZ"}
    graph: Literal["capture_safe","eager_only","unknown"]
    plan_kind: Literal["attention","kernel","stateless"]
    enabled_by_default: bool                 # False 的永远不默认启用（候选/实验）
    measured_ratio: float | None             # None = 未测
    ratio_note: str                          # 怎么测的：卡型/日期/bench 命令/覆盖与未覆盖形状
    requires: frozenset[str] = frozenset()   # 可选依赖（如 "deep_select"）
    provenance: str = ""                     # 来源（仓库@commit:文件）
    note: str = ""
```

三个"清单即门"的强制点（这是本仓与 TileInfer 最大的机制差异，也是 README"平台据此提前拒绝"的兑现方式）：

1. **`manifest()` 只能由 `ImplSpec` 生成**。CI 里重建一次并与已提交的 `manifest.json` 比对，
   防止"文档写了能力、代码没声明"或反之。
2. **`supports(op, shape=..., dtype=..., layout=..., graph=...) -> bool`**：平台在组批之前问，
   不需要 torch 张量、不需要设备。它和 `select()` 用**同一份数据**求值。
3. **不变量：`measured_ratio is None` ⇒ `enabled_by_default is False`**（baseline 例外，它记 `1.0`：
   基线就是自己，比值有定义）。这条把 README 的"没测过的实现不许默认启用"变成注册期断言，
   而不是靠人记得。缺字段、空 `layouts`、违反本不变量，注册直接失败。
4. **"能否进图"必须被机械验证**，不能靠注释。两步：
   - 静态：AST 扫描 `run` 可达代码里是否出现 `torch.empty/zeros/ones`、`.item()`、`.cpu()`、`synchronize`、`compile`；
   - 动态：`with capture_guard():` 上下文里打桩计数，`run` 期间任何一次分配/同步/编译即 fail。
   （TileInfer 已经付过这个代价：桶填充 slice 的 bug 只在图捕获时才炸，`known-issues.md:8-57`。）

多形态与"同族混用"由此自然成立：一个 `op` 下多个 `ImplSpec`，清单里全部列出，
`select()` 按"优先级 = 注册顺序"取第一个合格者，baseline 永远合格。
这与 `CandidateSet` 逐位同构，只是第 ④ 步从"调 lambda"变成"查 Spec"。

### 3.3 Plan 协议：从 attention-only 到全库

TileInfer 只有 attention 有 plan/run（`attention/api.py:75-240`），其余算子即调即算。
本仓把 "query/plan/execute" 作为**全库**接口，但允许三类 plan，避免为简单算子强行造对象：

| `plan_kind` | 是什么 | 例子 | run 期约束 |
|---|---|---|---|
| `attention` | 有调度、workspace、split/merge | paged decode/prefill | 无分配/无编译/无同步 |
| `kernel` | 有编译句柄与 staging buffer | conv_update、chunk_wu、fused_sample | 同上 |
| `stateless` | 纯 torch 组合，无编译产物、无 workspace | gating、prenz pack、slot layout | 允许分配，但必须声明 `graph=eager_only` |

```python
# ascend_attention/plan.py
class Plan(Protocol):
    op: str
    impl: str
    spec: Mapping[str, Any]        # 静态形状/编译规格，可 JSON 化
    workspace: Mapping[str, Tensor] | None
    kernel: Any | None             # 编译句柄（stateless 为 None）
    graph: str                     # 与 ImplSpec 一致

def plan(op: str, **pure_data) -> Plan: ...
def run(p: Plan, *tensors, out=None) -> Tensor | tuple[Tensor, ...]: ...
```

职责切分（写进 README，避免和平台抢）：

- **库**：`plan` 里允许编译（JIT），并把**编译键**作为数据导出；`run` 里禁止一切 host 侧决策。
- **平台**：决定何时 build、build 几次、workspace 放哪、是否 capture、什么时候释放。
- **状态张量**（GDN recurrent state、conv state、KV cache）**一律由平台拥有并传入**，
  库不分配、不持有、不跨调用记忆。这条直接来自 README 的非目标清单。

编译键必须由库给出，且必须覆盖"任何会改变编译产物或 staging buffer 的维度"。TileInfer 的两次事故是现成教材：
`plan_key` 漏了 `split_pages` 导致两种 split 因子共享 plan（静默算错，`docs/extending.md:478-484`）；
池容量用精确页数导致 9425/9899 漂移击穿缓存（`architecture.md:199-206`）。
本仓因此要求每个实现导出 `compile_signature(plan) -> tuple`（即"这份 plan 需要哪份编译产物"），
并配一条**逐维扰动测试**：对 spec 的每个维度，构造两个只在该维不同的 plan，断言签名不同；
再对一个确定不影响编译的维度（如 batch 大小）断言签名相同。
枚举式测试比注释可靠——`split_pages` 那次就是这么漏的。

---

## 4. 目标结构与文件映射

```
AscendAttention/
├── pyproject.toml                  # torch 为唯一硬依赖；npu/tilelang/vendor 为 extras
├── LICENSE  NOTICE                 # Apache-2.0 + 来源声明（现在两个都没有）
├── README.md
├── docs/
│   ├── plan-migrate-tileinfer.md   # 本文
│   ├── contracts/{attention,gdn,moe,sampling}.md   # 算子契约（迁自 fla-chunk-contract 等）
│   ├── pitfalls.md                 # 迁自 known-issues，按"编码成什么机制"重写
│   └── perf-rules.md               # 迁自 perf-comparison-rules，铁律照抄
├── ascend_attention/
│   ├── __init__.py                 # 唯一公共面：capabilities / plan / run（+ manifest）
│   ├── spec.py                     # ShapeSpec / ImplSpec / 注册表
│   ├── dispatch.py                 # 四道门 + select + describe（移植 candidates.py）
│   ├── plan.py                     # Plan 协议、WorkspaceManager、LPT/分块调度
│   ├── abi.py                      # declared_pool_cap / pool_view / capacity_view / padded_rows
│   ├── device.py                   # 设备探测与能力声明（ascend950 / 910b / 海光占位）
│   ├── manifest.py                 # 清单生成、JSON 导出、与 CI 快照比对
│   ├── ops/
│   │   ├── attention/  {entry.py(=api), metadata.py, impl/{reference,tilelang950}.py, kernels/*.py}
│   │   ├── gdn/        {gating.py, conv_update.py, chunk_chain.py, weight_layout.py}
│   │   ├── moe/        {slot_layout.py, prenz.py, align.py, grouped_gemm.py, shared_fold.py}
│   │   └── sampling/   {threshold.py(vendor_call), fused_sample.py, greedy.py}
│   ├── oracle/         # fp64 数学定义（离线可跑）+ torch 组合参考
│   └── bench/          # 单算子基准，产出 measured_ratio 的 JSON
├── tests/              # 纯 CPU 套件（每个文件可独立跑）
└── scripts/env-ascend.sh          # 迁自 scripts/env-ascend102.sh（改路径，保留 CANN 与缓存纪律）
```

映射动作分五类，**每类都要在 PR 描述里写明**：

| 动作 | 含义 | 例子 |
|---|---|---|
| A 原样 | 文件头改包名，逻辑不动 | `kv_nz_layout.py`、`prenz.py`、`slot_layout.py`、`oracle/*` |
| B 改签名 | 去引擎对象、环境变量改参数 | `paged_decode_ascend950.py` 的 6 个组合入口、`plan.py`、`metadata.py` |
| C 拆两层 | 内核留、胶水重写 | `tilelang_ascend950.py`（backend 胶水）→ `ops/attention/impl/tilelang950.py` |
| D 只取纯函数 | 从引擎补丁里挑出无副作用的部分 | `reorder.py` → `gdn/weight_layout.py`；`shared_expert_fold.py:81-118` → `moe/shared_fold.py` |
| E 不迁 / 归档 | 负结果、死代码、平台层 | `moe/grouped_gemm`、`/vllm/{worker,graphs,...}`、`legacy/**` |

历史保留：本仓与 TileInfer 是两个独立 git 仓。要保 blame 就用
`git filter-repo --path tileinfer/kernels --path tileinfer/attention --path tileinfer/plan.py --path tileinfer/metadata.py`
在 TileInfer 的**克隆**上切出子历史，再 `git pull --allow-unrelated-histories` 合并进本仓；
不要用 `cp`，否则第一周就会出现"这行为什么这么写"无据可查。

---

## 5. 分批计划

每批都遵循同一条纪律：**先有 oracle，再有实现，最后才有 `measured_ratio`；没有比值的实现不许默认开**
（`docs/status.md:116-120`、"仓库纪律"）。

### P0 · 三入口 + attention 合同层（5 天，纯 CPU）

内容：包骨架、`spec/dispatch/plan/manifest`、`LICENSE/NOTICE`、`env.ascend.sh`、CI（CPU job）；
迁 `attention/{metadata,plan,api}` + `backends/{base,reference}` + `oracle/reference.py`；
同时用 `gdn.gating` 做第一个 `stateless` 形态的样例（`torch`，带 fp64 oracle）。

验收（全部本机可跑）：
- `import ascend_attention` 在没有 torch_npu / 没有 tilelang 的机器上成功；
- `manifest()` 能 `json.dumps`，且 `supports()` 无需张量即可拒绝/接受；
- `plan/run` 在 `reference` 实现上跑通 decode 与 prefill 全部分支，与 fp64 oracle 对拍（尾页 poison `1e4`）；
- `capture_guard()` 能抓住故意插入的 `torch.empty`（自测）；
- 零 `import vllm`；每个测试文件单独跑绿、整套跑绿（不重复 TileInfer 的 80 项污染，`status.md:93-107`）。

风险：`dispatch` 的 Spec 化最容易做过头（把 predicate 的表达力砍掉）。兜底：`extra_guard` 保留，
但**manifest 必须能只凭 ShapeSpec 生成**。

### P1 · attention 内核与两个真实形态（12 天，需设备窗）

内容：迁 `paged_decode_ascend950.py`（decode/split/merge/prefill + 6 个组合入口 B 类改造）、
`abi.py`（池容量量化/`pool_view`/`capacity_view`）、`ops/attention/impl/tilelang950.py`；
`layouts` 第一次真正被使用（NHD 支持、NZ 声明为"需 pack，不直接支持"）；
`graph` 第一次被机械验证（decode 捕获安全，prefill 组合入口 `eager_only`）。

验收：设备上 decode 与 prefill 对数通过（TOL 3e-2）；`capture_guard` 在 decode `run` 上零违规；
`manifest` 里两个 impl 的 `measured_ratio` 与 `ratio_note` 齐全；`supports()` 能仅凭 shape 拒绝非法的 `head_dim`/`group`。

风险：split-KV 的编译病态（`known-issues.md:355-359`，dim 256 下编译 17 分钟、dim 128 下 2-3.9 小时）。
对策：split 形态默认不进 manifest 的默认集，标 `eager_only` + `note` 写明编译代价；提供 `prewarm()` 纯函数
（库只给"要编译哪些键"，什么时候调是平台的事）。

### P2 · GDN 族（12 天，CPU + 设备）

内容：`gdn.gating`（torch，先做）、`gdn.conv_update`（kernel 形态，opt-in）、
`gdn.chunk_chain` 的四段 torch 基线（迁 `ops/fla/{kkt,wu,delta_h,chunk_o,solve_tril}.py` 的纯计算部分）、
`gdn.chunk_wu`（融合内核，候选、默认关）、`gdn.weight_layout`（D 类，来自 `proj_fusion`/`reorder` 的纯函数）。

验收：chunk 链对 fp64 oracle 逐段过（四个"静默杀手"要各有一条测试：
`exp` 只作用绝对 g、mask 用 `where`、两个不同的三角、三处 bf16 截断——`fla-chunk-contract.md:553-642`）；
状态张量全部由 caller 传入，库内无 `self._state`；`conv_update` 的 `graph` 声明与实测一致。

风险：`conv_update` 端到端 0.995 —— **不要因为内核级 42.5× 就默认开**。本仓的 `measured_ratio`
一律记端到端或明确标注"op 级"（`perf-candidates.md:43` 的教训）。

### P3 · MoE 族（8 天，CPU 为主）

内容：先做判决——`moe.slot_layout`、`moe.prenz`、`moe.shared_fold`（纯函数）三个是资产，
`align/grouped_gemm` 是"未达门槛 + 一条负结果"。默认只迁前三个 + 一个 torch 组合基线；
后两个在 `docs/contracts/moe.md` 里记档（含 0.86× 与 preNZ 结果错的证据），代码放 `archive/` 分支。

验收：`moe` 族的 `manifest` 至少有一个 `torch` 实现能跑通两种专家分布（均衡 / 单专家热点）；
slot layout 的哨兵语义（`sorted_ids == rows_static` 表示空 tile）有契约测试；
"未接入的高性能候选"在清单里显示为 `enabled_by_default=False` 且 `note` 可读。

风险：MoE 是唯一"基线也不好写"的族（路由 + 对齐 + 分组 GEMM 三段耦合）。
对策：先只承诺"对齐与槽位"两个纯数据算子，专家计算以 torch 组合形态提供，不做性能承诺。

### P4 · 采样族与"直调厂商算子"形态（10 天，需设备）

内容：`sampling.threshold`（`vendor_call`，`requires={"deep_select"}`，缺失即不可见）、
`sampling.fused_sample`（迁内核 + **补 builder/缓存**，现在完全没有）、`sampling.greedy`（torch，来自 legacy）。
`vendor_call` 是本仓第五种形态的第一次落地，要顺带把"厂商实现与自研实现同族混用"的报告打通。

验收：三个实现同族共存，`supports()` 按 `(vocab, batch, do_sample)` 分流且有测试；
`measured_ratio` 两个来源（op 级与 end-to-end）分开记；无 `deep_select` 的机器上 `manifest` 里该实现消失而不是报错。

风险：`fused_sample` 没有消费者、没有端到端数字（`perf-candidates.md` 无其条目），
且 `ub_cap=8192` 会触发 `507035`（`fused_sample.py:110-114`）。所以它进 manifest 时必须是 `eager_only` + 默认关。

### P5 · 账本、寄生适配器与反向接入（8 天）

内容：
1. 发布 `manifest.json`（含全部 `measured_ratio` 与 `ratio_note`），作为对平台的唯一交付物；
2. 可选包 `ascend_attention.vllm_adapter`：把本仓算子挂回 vLLM 的 ABI 名字（"寄生适配器"），
   **只提供算子，不提供平台语义**，且不随主包安装；
3. TileInfer 侧反向改造：把 `tileinfer/kernels/**` 与 `candidates.py` 换成对本仓的依赖，
   `CandidateSet` 的 baseline 变成"本仓的 torch 实现"，候选变成"本仓的 kernel 实现"。
   这一步是本方案真正的验收：**同一份算子被两个仓库共用，且数值逐位相同**。

验收：TileInfer 在 `TILEINFER_OPS_CANDIDATES=off` 与 `on` 两种设置下数字不变（行为等价）；
本仓的 `manifest` 能驱动 TileInfer 的全部 15 个绑定（覆盖率 100% 或有 `note` 说明缺口）。

---

## 6. 待拍板的决策（每条都影响结构，建议先定 D1/D3/D4）

| # | 决策 | 选项 | 推荐 | 影响 |
|---|---|---|---|---|
| D1 | 包名与 import 名 | `ascend_attention` / `ascend_attn` / `aa` | `ascend_attention` | 与仓名一致；代价是 import 路径长 |
| D2 | plan 粒度 | per-op plan / per-model-stage plan | **per-op** | per-stage 会把平台语义吸进库（违反单向依赖） |
| D3 | 状态张量归属 | 库持有 / caller 传入 | **caller 传入** | 库持有就要管生命周期与图捕获，直接越界 |
| D4 | 布局能力的表达 | 布尔 / 枚举集 / 枚举集 + 需 pack 标记 | **枚举集 + `needs_pack`** | NZ 目前是"可 pack 但内核不直吃"，布尔会撒谎 |
| D5 | `graph` 的三档怎么定义 | 两档 / 三档 | **`capture_safe` / `eager_only` / `unknown`** | 两档会把"有分配但无同步"误判为可捕获 |
| D6 | 编译产物形态 | 只给源码 / 源码 + 预编译 `.so` | **先只给源码 + `prewarm()` 纯函数** | 预编译产物与 CANN 版本强耦合，进仓会腐烂 |
| D7 | 环境变量策略 | 完全禁读 / 允许一个总开关 | **允许一个 `ASCEND_ATTN_DEBUG`**，其余一律参数化 | 内核调试确实需要开关注入，但不能有 8 个 |
| D8 | 负结果怎么留 | 删掉 / 代码留默认关 / 归档分支 + 文档 | **归档分支 + 文档条目** | 代码留主干会让 6413 行里一半是永不启用的东西 |
| D9 | 是否保留 `reference` torch 后端 | 保留 / 只留 oracle | **保留**（它是回退路径的产品形态） | 与 README"回退到厂商原生"并行不冲突 |
| D10 | Python/torch 下限 | 3.9（TileInfer 现状）/ 3.12（实机） | **3.10 + torch 2.5** | 需要 `Protocol`、`|` 联合类型；实机 3.12 满足 |

---

## 7. 迁移时必须编码成机制的坑

TileInfer 的 `known-issues.md` 与 `fla-chunk-contract.md` 里的坑，逐条对应到本仓的一个机制。
**"写进文档"不算解决，"有测试"才算。**

| 坑 | 出处 | 编码成什么 |
|---|---|---|
| 尾页 padding 未在 `exp` 前 mask → softmax 分母偏大、数值仍在合理区间 | `architecture.md:104-109` | oracle 用 `1e4` poison padding 的固定测试；每个 attention 实现强制跑 |
| Ascend OOB gather 返回垃圾而非报错 | `known-issues.md:59-95` | `Plan` 持有已 bounds-check 的页表副本；`metadata.validate()` 在 plan 期拒绝 |
| 子 32 字节 GM→UB 描述符 → 507035 且**挂死设备** | `known-issues.md:163-211` | `ShapeSpec` 里加 `min_group`/`min_stride_bytes` 约束，`supports()` 直接拒绝；oracle 加 group=4 用例 |
| AIV 局部行号在 `T.SimtVF` 里不可见 → causal 阈值错 | `known-issues.md:246-267` | 逐行阈值必须由 host 预算并作为张量传入；契约测试覆盖"上下半区" |
| bf16 往返是契约不是缺陷（`A_inv`/`w,u`/`h` 三处截断） | `fla-chunk-contract.md:553-642` | oracle 必须复现截断点，否则 A/B 被噪声吃掉；测试断言"改精度即失败" |
| 两个不同的三角（kkt `>` vs chunk_o `>=`） | 同上 | 契约测试各一条，且用"对角不为零"的输入 |
| 编译占 GIL 70–100 s，在服务里像引擎挂了 | `architecture.md:185-197` | `prewarm(keys)` 纯函数 + 文档写明"必须在进程外调用"；库内不做后台编译 |
| 编译键漏维度 → 两份 plan 共享一个内核，静默算错 | `extending.md:478-484` | §3.3 的 `compile_signature` + 逐维扰动测试 |
| 池容量精确值导致缓存击穿 | `architecture.md:199-206` | `abi.declared_pool_cap()` 量化（pow-2 或显式钉住），键里只放量化值 |
| `pool_view` 丢了 storage offset → 内核把 K 读成 V（静默错数） | `paged_decode_ascend950.py:3140-3144` | `abi.pool_view()` 自带 offset 断言测试（用不同 offset 的切片构造） |
| 同一算子的两个实现的元数据 dtype 不一致（`kv_last_page_len` int32 vs fork 变体的 fp32） | `paged_decode_ascend950.py` vs `paged_decode.py:196-208` | ABI 把每个参数的 dtype 写成数据；跨实现一致性测试 |
| `from __future__ import annotations` 破坏 TileLang 解析 | `architecture.md:228` | 内核目录 lint 规则（ruff `flake8-future-import` / 自定义 check） |
| env 变量在 trace 时读 → 编译结果依赖进程状态 | 内核里 8 处 | 内核模块禁止读 `os.environ`（CI grep + 代码评审），全部改成显式编译规格 |
| 测试顺序污染（同文件单跑绿、整套红 80 项） | `status.md:93-107` | 新仓禁止"全局 fixture + 假引擎"；CI 双跑：整套 + 逐文件 |
| 搬入/移植代码的许可证与出处 | `fused_sample.py:1-3`（移植自 DeepInfer）、`reorder.py`（镜像 vLLM PR） | `NOTICE` + 每个 `vendor/`、`archive/` 文件头带 `provenance` 字段 |

---

## 8. 验证设施（本仓最有复用价值的部分）

### 8.1 三层 oracle（全部离线可跑）

1. **fp64 数学定义**：直接按公式算，不走任何 fast path（TileInfer 每族都有，`gdn/__init__.py:39-52` 是模板）。
2. **torch 组合**：与内核同 ABI、同 dtype 策略，是"回退路径"的实体，也是逐位比对对象。
3. **设备对数**：`log` 域相对误差 + 绝对容差，按 dtype 分档（bf16 → 3e-2）。

规则照抄：**内核永不与内核比**（`architecture.md:392-394`）。失败时能定位到层，而不是"某处不对"。

### 8.2 bench 与账本（四条铁律，逐字继承）

- 两臂同一环境、同一 torch/CANN/tilelang、同一 `TILELANG_CACHE_DIR`、同一卡状态；
- 只引用**同进程交替 A/B ≥4 对**的比值（`perf-comparison-rules.md` 铁律）；跨运行绝对值不可比；
- 端到端只用差分法（N1=8、N2=72），非物理样本作废；
- 每条比值必须同时给：命令、卡型、日期、覆盖与未覆盖形状、测量脚本。

账本 schema（`bench/results/*.json`，被 `manifest.py` 读取）：

```json
{"op":"gdn.conv_update","impl":"tilelang.ascend950.conv_update.v1","baseline":"torch.causal_conv1d_update",
 "level":"op","ratio":42.5,"pairs":["..."],"date":"2026-10-21","card":"Ascend950PR_9579",
 "covers":["bf16,width=4,channels=2048"],"not_covers":["fp16","width>4"],
 "end_to_end":{"ratio":0.995,"note":"no gain; 4/4 valid pairs"}}
```

### 8.3 CI 两个 job

- `cpu`：`pytest -m "not npu and not tilelang"`，**必须逐文件与整套都绿**；外加 `manifest` 快照比对、
  `json.dumps` 可序列化、import hygiene（零 vllm / 零 torch_npu 在基线路径）、`capture_guard` 自测。
- `device`（102 机器、手动触发）：先跑 canary（一个已知正确的解码用例）再跑套件，
  失败即中止并打印 `npu-smi`（照抄 `scripts/run_device_ci.sh` 的 canary-then-suite 模式）。

---

## 9. 明确不迁（与 README 非目标逐条对齐）

| 不迁 | TileInfer 里的位置 | 为什么 |
|---|---|---|
| 平台注册（platform plugin、`AttentionBackendEnum.CUSTOM` 接管） | `vllm/platform.py`、`plugin.py` | README：不注册任何引擎入口 |
| worker 四个 hook、显存查询、prewarm 的 serve 派生 | `vllm/worker.py`（2059 行） | 平台语义 |
| 图捕获（ACLGraph）与 dynamo opaque | `vllm/graphs.py`（929 行） | 本仓只声明"能否进图"，不实现捕获 |
| KV cache 分配、块表、混合 KV 池 | `vllm/attn/kv_cache.py`、`metadata.py`(builder 部分) | 本仓只吃 KV 张量 |
| 模型实现与 profile（dsv41、qwen3_6） | `models/**`、`profiles/**` | 平台侧"加模型" |
| 装载器与开关（`install/switch/families`） | `vllm/ops/**` | 这是"寄生适配器"的原料，不是库 |
| 观测（`ti_trace.py`、phases、counters、doctor） | `serving/**`、`vllm/doctor.py` | 平台观测 |
| `reference/vllm-0.23/**` | — | 本仓不引用引擎行号 |
| `legacy/**`（5945 行） | — | 冻结的补丁层；只当契约素材 |
| 量化 / MLA / 多卡 / TP-EP | — | README 非目标 |

---

## 10. 前三个 PR（可立即开工）

**PR#1（本机可完成）**：骨架 + 规格 + 分派 + 清单 + 守卫测试。
内容：`pyproject.toml`、`LICENSE`、`NOTICE`、`spec.py`、`dispatch.py`、`manifest.py`、`device.py`、
`tests/test_spec.py`、`tests/test_dispatch.py`、`tests/test_import_hygiene.py`、`tests/test_manifest.py`。
判据：三个动词可用（此时只有 `gdn.gating` 一个算子、一个 torch 实现，够证明链路不缺件），
`manifest()` 出 JSON，`supports()` 无需张量能拒绝，CI 绿。

**PR#2**：attention 合同层（`metadata.py`/`plan.py`/`api.py`/`backends/reference.py` + oracle）。
判据：decode 与 prefill 在 reference 实现上对 fp64 oracle 全分支通过，含尾页 poison、乱序页、causal offset、GQA 连续四类用例。

**PR#3**：`gdn.gating`（torch）+ `gdn.conv_update`（kernel 的 `eager_only` 声明，先不接内核）+ 第一次 `capture_guard`
动态验证 + bench 脚本骨架（只测 CPU 可测的 `torch` 形态）。
判据：一个算子族具备"实现 + 契约 + oracle + 账本 + 门"五件套，后续各族照抄这一个模板。

---

## 附录 A：manifest 片段（示意）

```json
{
  "schema": 1,
  "library": {"name": "ascend_attention", "version": "0.1.0"},
  "device_profiles": {"ascend950": {"arch": "dav-3510", "cores": 28, "graph": "aclgraph"}},
  "ops": {
    "attn.paged_decode": {"impls": [
      {"impl": "torch.reference", "form": "torch", "device": "any",
       "dtypes": ["bf16","fp16","fp32"], "layouts": ["NHD","HND"],
       "graph": "eager_only", "plan_kind": "attention",
       "enabled_by_default": true, "measured_ratio": 1.0,
       "ratio_note": "基线定义本身；与 TileInfer reference 后端逐位一致"},
      {"impl": "tilelang.ascend950.decode", "form": "tilelang", "device": "ascend950",
       "dtypes": ["bf16","fp16"], "layouts": ["NHD"],
       "graph": "capture_safe", "plan_kind": "attention",
       "enabled_by_default": false, "measured_ratio": null,
       "ratio_note": "对数已过（2026-10-09, 950PR, TOL 3e-2）；P1 验收项：同进程配对 A/B 落地后补比值，才允许转默认",
       "provenance": "tileinfer@339969a:tileinfer/kernels/attention/paged_decode_ascend950.py"}
    ]}
  }
}
```

## 附录 B：能力声明与查询（示意）

```python
from ascend_attention import capabilities, plan, run

capabilities(op="attn.paged_decode")                    # 全部实现及其声明
capabilities(op="attn.paged_decode", graph=False)       # 只问"能进图的"
capabilities(device="ascend950", layout="NZ")           # 只问"支持 NZ 的" → 空集 + 原因

# 平台在组批之前问，不需要张量、不需要设备
from ascend_attention import supports
supports("attn.paged_decode", shape=(32, 8, 128), dtype="bf16", layout="NHD", graph=True)  # -> True

p = plan("attn.paged_decode", num_qo_heads=32, num_kv_heads=8, head_dim=128, page_size=128,
         kv_indptr=..., kv_indices=..., kv_last_page_len=..., kv_layout="NHD", k_cache=..., v_cache=...)
out = run(p, q)      # 无分配、无编译、无同步；失败即"该实现不合格"，不静默回退
```

---

## 附：工作量与依赖关系一览

| 批次 | 依赖 | 工时 | 主风险 |
|---|---|---|---|
| P0 基建 + attention 合同层 | — | 5d | Spec 化做过头 |
| P1 attention 内核 | P0、设备窗 | 12d | split/prefill 编译病态 |
| P2 GDN | P0（P1 可并行部分） | 12d | bf16 截断契约、状态归属 |
| P3 MoE | P0 | 8d | 基线难写、负结果诱惑 |
| P4 采样 | P0、设备窗 | 10d | 外部依赖与 `507035` |
| P5 账本 + 反向接入 | P1–P4 | 8d | TileInfer 侧行为等价证明 |

单人力 ≈ 11 周；两轨并行（P0→{P1,P2} 与 P3/P4 分头）≈ 6 周。
