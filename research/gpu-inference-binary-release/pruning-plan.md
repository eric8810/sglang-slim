# sglang 单机版精简清单与预期产出（实施版）

基于本目录三轮调研 + 独立复核结论。产品定位：**单机（PCI 互联）NVIDIA GPU 的精简推理服务，自包含目录 + 可选 OCI 镜像双交付，目标机仅需 NVIDIA 驱动**。

核心策略：**fork 面积从零起步**。删减分四层，层 0-2 对 sglang 源码零 diff（不 fork、不 rebase，上游更新只需重跑构建管线）；层 3 才引入私有 patch（进入 fork 管理模式）。

---

## 一、删减清单（按 fork 面积分层）

### 层 0：纯打包裁剪（零 sglang diff）

| 对象 | 方式 | 量级 |
|---|---|---|
| 顶层 test/ docs/ benchmark/ scripts/ examples/ tools/ assets/ docker/ 3rdparty/ | 构建期不进产物 | ~59 MB / 44% 仓库体积 |
| sgl-model-gateway/ experimental/ | 同上（集群路由，Python 零耦合） | 8.3 MB |
| python/sglang/lang/（遗留 DSL 前端） | 同上 | 4.7K 行 |
| kernels/ 中 diffusion / inkling / kimi_k3 等专用内核 | 按目标模型保留 | ~8K 行内核源 |
| rust/ 构建裁剪 | `SGLANG_BUILD_RUST_EXTS=grpc,radix-tree`（不编 sglang-server/mm） | 两个 crate |

### 层 1：运行时配置关闭（零 diff，文件进产物但不激活）

| 对象 | 开关 | 说明 |
|---|---|---|
| 模型注册白名单 | `SGLANG_DISABLED_MODEL_ARCHS` | 不减小体积，只减 import 时间；减体积走层 2 |
| 投机解码 / LoRA / 结构化输出 / beam search / pdmux / PD 分离 | 对应 server args 默认关 | 零成本保险 |

### 层 2：构建期文件排除（零 diff，文件不进产物；源码树不删，rebase 不受影响）

| 对象 | 量级 | 安全依据 |
|---|---|---|
| models/ 非白名单（保留 llama/qwen/deepseek 三族 + registry + utils + transformers.py + deepseek_common） | **~13 万行** | registry pkgutil 扫描，文件缺失自动跳过；核心路径仅 1 处函数级引用 |
| mem_cache/storage/ 远端后端（mooncake/nixl/hf3fs/umbp/flexkv/lmcache/aibrix/eic/simm/npu，保留 file/mmap/shm） | ~14.9K | backend_factory 懒加载注册 ⚠️ 需验证文件缺失时注册是否静默跳过 |
| disaggregation/ 传输后端（mooncake/nixl/mori/ascend/fake） | ~8.6K | get_kv_class 延迟 import，null 模式不触发 |
| hardware_backend/ npu/xpu/musa | ~23.6K | 无核心静态依赖 |
| device_communicators/ 跨节点与他厂卡（mooncake_transfer_engine/triton_symm_mem_ag/pymscclpp/hpu/npu/xpu） | ~1.7K | parallel_state 内函数级延迟 import |
| function_call/ 各家 detector（保留 parser 框架 + 1 个 detector） | ~13K | pkgutil 扫描按需触发 |
| debug_utils/ comparator、schedule_simulator | ~8K | 独立工具 |
| entrypoints/ anthropic/ ollama/ search/（若不用这些 API） | ~2.9K | ⚠️ 需验证 import 方式 |
| ray/、checkpoint_engine/、gated_launch、naive_distributed、legacy grpc_server | ~3K | 零静态依赖或延迟 import |

层 2 合计：srt 产物再减 **~17.5 万行**；加上层 0 后，产物代码量 ≈ 79 万 → **~60 万行**。

### 层 3：私有 patch（引入 fork，按 tag 同步 + rerere + PR 回上游）

| 对象 | 量级 | patch 内容 |
|---|---|---|
| hardware_backend/mlx | 7.2K | 5 处静态 import 的 `use_mlx` 改 env 判断 |
| speculative/ | 28.7K | scheduler.py 等 4-5 处顶层 import 延迟化 |
| lora/ | 15.6K | scheduler/model_runner/server_args 三处延迟化 |
| multimodal/ | 25K | 纯文本场景，managers 约 10 处补丁 |
| kv_canary/ state_capturer/ elastic_ep/ dllm/ constrained/ beam_search/ observability/ weight_cache/ compilation 部分 | ~30K | 各 1-3 处顶层 import 延迟化 |
| disaggregation/ 框架部分（decode/prefill mixin、encoder） | ~1.9 万 | 剥离 utils 通用件（DisaggregationMode/prepare_abort）到公共模块 + mixin 条件化 |
| arg_groups/ PD 与跨节点参数 | ~70 个参数 | 清理 CLI 面（可选，纯体验） |

层 3 全做：srt 达到 **~45 万行（-43%）**。建议按需渐进，每项独立可交付。

### 依赖精简（层 2/3 伴随）

pyproject 删：ray extra、torchvision/torchaudio/torchcodec/soundfile/av/timm/pillow（多模态）、outlines/llguidance/xgrammar/interegular（结构化输出，若层 3 删 constrained）、anthropic、modelscope/blobfile（本地权重）。保留：nvshmem4py、sgl-deep-ep、sgl-deep-gemm、flashinfer、sglang-kernel（单机 MoE/通信需要）。

---

## 二、构建管线

```
上游 sglang tag (pin, 已核实 tag 密集可用)
  │
  ├─ [层0] 源码装配：排除顶层 + rust 裁剪 + venv(python-build-standalone)
  ├─ [层1] 配置注入：默认 args + SGLANG_DISABLED_MODEL_ARCHS
  ├─ [层2] 打包排除：rsync 白名单清单 → site-packages
  ├─ [层3] (可选) 私有 patch 分支 rebase
  │
  ├─ JIT 预热（GPU arch 匹配的构建机）:
  │    deep-gemm (DG_JIT, 启动预编 3072 kernels)
  │    triton / tilelang / cutlass-DSL(cute_aot) / sglang 自有 jit (~/.cache/sglang)
  │    flashinfer: INSTALL_FLASHINFER_JIT_CACHE=1 + flashinfer-cubin
  │    收集缓存目录 → 校验 key 无绝对路径漂移（cache.py 已归一化）
  │
  ├─ 产物 A: sglang-lite-<ver>-<cuda>-<arch>.tar.zst（自包含目录）
  └─ 产物 B: OCI 镜像 = COPY 产物 A（非 devel base，无 toolkit）
```

构建机约束：与交付目标同 GPU arch、同 CUDA 大版本、包版本锁定（缓存 key 三要素）。

## 三、最终预期产出

```text
sglang-lite-<ver>/
├── bin/sglang-serve            # 单入口 launcher（设 env + exec python -m ...）
├── runtime/                    # python-build-standalone CPython（可重定位）
├── lib/site-packages/          # 精简 sglang + 全依赖（nvidia-*-cuXX 轮子携带 CUDA 库）
├── lib/jit-cache/              # 预热缓存（deep_gemm/triton/tilelang/cute/sglang/flashinfer-cubin）
├── models/                     # 空目录 + registry 配置（模型旁路分发）
├── etc/                        # 默认 server args 模板（单机 profile）
├── LICENSE                     # Apache 2.0 全文（条款4a）
├── NOTICE-THIRD-PARTY          # 依赖声明 + 修改声明（4b/4c）
└── SBOM.spdx                   # 版本清单（含 sglang 上游 tag、包版本、CUDA 版本）

镜像: sglang-lite:<ver>-cu<x>   # FROM 非 devel 基础 + COPY 上目录，entrypoint bin/sglang-serve
```

量化预期（估算，未实测）：

| 指标 | 官方镜像参照 | 预期 |
|---|---|---|
| 自包含目录 | — | 3-6 GB（CUDA 轮子为大头） |
| OCI 镜像 | devel base + 全量（大） | 接近目录体积 + 薄层 |
| 目标机要求 | 驱动 + toolkit（pip 路线会踩 JIT） | **仅 NVIDIA 驱动** |
| 部署动作 | — | 解压即跑 / docker run |
| 产物代码量 | 79.2 万行 | 层2: ~60 万行；层3: ~45 万行 |

## 四、验收标准（Go/No-Go）

1. **JIT 命中完整性**（一票否决项）：无 toolkit 目标机，`SGLANG_CRASH_ON_JIT_COMPILE=1`，目标模型全链路冷启动（prefill/decode/batch 变化/CUDA graph 重放/量化路径）不崩
2. 零私 diff 下（层 0-2）产物可正常运行：对拍官方镜像同 prompt 输出
3. Apache 2.0 合规：LICENSE + NOTICE 齐备（ollama 教训）
4. 层 3 patch 分支：对上游最新 tag rebase 无阻塞性冲突（CI 演练一次）

## 五、实施前待验证（小时级实验）

1. backend_factory 文件缺失时注册行为（静默跳过 or 报错）——决定 mem_cache/storage 排除是否零 patch
2. entrypoints 可选协议（anthropic/ollama/search）的 import 方式
3. 层 0-2 产物在无 toolkit 机器的 JIT 缓存命中集合收集（冒烟模型即可起步）
4. sglang 官方 Dockerfile 以层 0-2 清单重建的体积实测
