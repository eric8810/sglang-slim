# Slim single-node packaging (layer 0/2, zero sglang diff)

本目录是 sglang 单机精简发行版的构建脚手架。核心约束：**对 sglang 源码零 diff**——
上游更新只需重跑构建管线，不存在 rebase。

## 实际运行演示（2026-09-16，Qwen3-4B，bundle 部署）

部署形态：tar.zst 解压到 /tmp，`env -i`（无任何环境变量，PATH 仅 /usr/bin:/bin）
启动 launcher：

| 指标 | 值 |
|---|---|
| 启动到 healthy | ~60s（权重加载 1.2s + prefill CUDA graph 捕获 42s[42 形状] + decode graph 0.7s） |
| KV cache 容量 | 38,925 tokens（bf16，K+V 5.3GB，context 40,960） |
| 单流 decode 速度 | **~49 tok/s**（Qwen3-4B bf16） |
| 4 并发 chat | 全部 0.56s 完成（continuous batching 生效，prefill 吞吐 3072 tok/s） |
| Radix cache | 命中（重复 prompt #cached-token 21） |
| chat API | 思考模式与 /no_think 均正常，代码类回答正确 |

运行期观察（非阻塞，记录备查）：
- 3 个 Triton kernel（write_req_to_token_pool 等）在 serving started 后才
  device-load（从缓存加载非编译，毫秒级；上游提示 pre-load 以避 OOM 风险——
  产品化可在 warmup 请求中覆盖）
- 第四处依赖闭包：deepseek_v4_dspark → dspark.py（已入白名单）

## 用法（完整管线）

```bash
python3 packaging/build_slim.py                                # 1. slim 树 + AST 契约
python3 packaging/install_deps.py --venv <V> --python <PBS_PY> # 2. 依赖装配（瘦身清单内置）
SGLANG_SMOKE_MODEL=<M> bash packaging/smoke_test.sh            # 3. 预热（首轮含 JIT 编译）
SGLANG_SLIM_BUILD=<B> bash packaging/make_bundle.sh            # 4. 自包含 bundle + tar.zst
SGLANG_SMOKE_MODEL=<M> bash packaging/e2e_verify.sh <TARBALL>  # 5. E2E 交付验证
# 目标机（仅需 NVIDIA 驱动）：
#   tar --zstd -xf sglang-lite-*.tar.zst && ./sglang-lite/bin/sglang-serve --model-path <M> ...
```

### E2E 交付验证抓到的四个真实缝隙（每个都只有端到端才能暴露）

1. **flashinfer import 时写日志**到其 workspace（被重定向进缓存目录）→ 只读交付
   崩溃。解法：launcher **seed 模式**——包保持只读，首启把预热缓存复制到
   `${XDG_CACHE_HOME:-~/.cache}/sglang-slim-runtime`（~30MB，秒级），此后指向它
2. **`cp -a` 保留只读权限位**：从只读包 seed 出来的缓存仍只读 → seed 后
   `chmod -R u+w`
3. **flashinfer 0.6.18 设计上每次启动 spawn ninja**：`try_load` 对 JIT 路径永远
   返回 None（core.py:396 注释明说 freshness 交给 ninja 扫描），即使全缓存命中。
   venv 冒烟时被 PATH 里的 ninja 静默掩盖（教训：**工具链在场会掩盖预热缺口**，
   严格验证必须 `env -i`）。解法：bundle 自带 ninja 二进制（wheel 装在 venv/bin
   而非 site-packages），launcher 把 `bin/` 加进 PATH
4. **`SGLANG_JIT_CACHE_DIR` 不从 `SGLANG_CACHE_DIR` 派生**（environ.py:1155 默认
   None → cache.py:302 硬编码 fallback `~/.cache/sglang/jit`，与 DG/cute_aot 的
   派生行为不一致）。解法：launcher 显式设置两个变量

附带发现：**deepseek_v4.py 依赖 dbrx.py**（又一处白名单依赖闭包，dbrx 已加入
`MODELS_KEEP_PREFIXES`）。依赖闭包目前靠"运行时看 fallback 日志"发现，后续应把
models 排除纳入 build_slim 的 AST 检查（warn 级）。

## 验证机制（零 patch 契约的执行者）

1. `compileall`：staged 树全量语法检查
2. **AST 模块级 import 检查**：对每个被排除模块，扫描全树确认**无任何模块级 import
   引用它**（函数内延迟 import 不算）。违规即构建失败并给出 file:line。

   > 该检查器已证伪多处人工调研结论——AST 是 ground truth，人工 grep 有盲区。

## 2026-09-16 首次构建 + 冒烟结果（全部 PASS）

| 阶段 | 结果 |
|---|---|
| 构建 | srt 792,340 → 626,206 行（**-21%**），排除 356 文件，AST 契约通过 |
| 环境 | python-build-standalone 3.12.14 + venv（76 个依赖轮子，8.9GB） |
| 冒烟 | Qwen3-4B bf16 顺利启动，60s healthy（含首轮 JIT），生成正常 |
| **Go/No-Go** | `SGLANG_CRASH_ON_JIT_COMPILE=1` 冷启动 **25s healthy，零编译触发**，输出与首轮一致（temp=0），e2e 0.66s vs 首轮 4.49s |
| **真无-toolkit Go/No-Go（dense）** | ✅ 卸载 nvcc 轮子 + 屏蔽系统 CUDA + crash-on-jit：25s healthy，零编译，输出一致 |
| **MoE 冒烟（granite-3.0-1b-a400m）** | ✅ 原生 sglang MoE + Triton fused MoE kernel（E=32,N=512），fp8 在线量化亦 PASS |
| **真无-toolkit Go/No-Go（MoE）** | ✅ 20s healthy，零编译 |
| **DeepGEMM JIT 触发验证** | ✅ `--moe-runner-backend deep_gemm` + fp8：GROUPED_GEMM_NT_F8F8BF16（32 groups）编译成功 + 512 warmup 完成，缓存落盘 `~/.cache/sglang/deep_gemm`；fp8 per-token-group-quant kernel 亦被 JIT |
| **自包含 bundle + E2E 交付验证** | ✅ 8.0G 目录 / 3.3G tar.zst；解压到异路径 + 假 HOME + 只读缓存 + `env -i` 无 toolkit + crash-on-jit：**60s healthy，生成正常**（granite MoE） |
| **运行库 block-list 裁剪** | ✅ `/proc/maps` 实测：cu13 死重 310M（nvrtc.alt/cusolverMg/nvvm/nvperf）未加载 → 删除后重验 PASS。多模态决策后 cudnn/头文件恢复保留 |
| **多模态支持（决策 B，2026-09-17）** | ✅ 依赖装回（torchaudio/torchcodec/av/timm，版本对齐）+ 闭包自动化；**Qwen3-VL-2B 视觉验证**：venv 与 bundle（env -i）均 PASS，图片文字逐字识别正确 |

### 交付物双形态（2026-09-17 完成）

| 形态 | 命令 | 产物 | 体积 |
|---|---|---|---|
| 自包含目录 | `make_bundle.sh` | `sglang-lite-*.tar.zst` | 3.2 GB（解压 7.8 GB） |
| OCI 镜像 | `make_image.sh --save` | `sglang-lite-*.oci.tar` | 3.6 GB（`docker load` 即用） |

镜像设计：ubuntu:24.04 base（glibc 2.39 与构建机一致）+ **COPY 自包含目录**
（非 pip install——Python/包/JIT 缓存全部构建期锁定，镜像即 immutable artifact）。
目标机：docker + nvidia-container-toolkit；模型走 volume 挂载（/models）。

```bash
docker load -i sglang-lite-bd45cd50.oci.tar
docker run --gpus all -v /path/to/models:/models -p 30000:30000 \
  sglang-lite:bd45cd50 --model-path /models/qwen3-4b --host 0.0.0.0 --port 30000 --mem-fraction-static 0.78
```

注意：本机未装 nvidia-container-toolkit（sudo 受限），镜像做了构建 +
容器内 bundle 树验证；**GPU 运行级验证需在有 toolkit 的主机补做**。

### 多模态验证细节（Qwen3-VL-2B）

- 视觉链路完整：base64 图片 → qwen_vl processor → vision encoder → 正确描述
  （背景色 / 逐字文字 / 图形形状全部准确）；bundle 级 env -i 启动 50s healthy
- VL 显存提示：`--mem-fraction-static` 建议 **0.78**（多模态 feature-transport 池
  + attention workspace 额外占显存，0.85 会挤到 OOM 边界）
- **闭包自动化（系统性修复）**：`compute_models_closure()` 从目标模型前缀自动展开
  传递依赖（三种 import 形式全覆盖：名字导入 / 模块导入 / 目录依赖）。manifest
  只声明目标族（llama/qwen/deepseek/granite），148 文件 + 3 目录自动展开——
  此前 7 次手动踩坑（mixtral/dbrx/dspark/clip/cosmos3/interns2/dots3_common）终结
- 已知边界：inkling 系模型完整运行需 nvidia-cutlass-dsl（已卸载；注册静默跳过，
  不影响 Qwen/DeepSeek 路径；将来支持时装回 +450M）

结论：**预生成 JIT 缓存替代 CUDA toolkit 的路线在本机验证成立**。
验证覆盖：dense（Qwen3-4B）+ MoE（granite，Triton fused kernel）+ fp8 量化 +
DeepGEMM JIT 编译机制（granite 形状）。
**未覆盖边界**：DeepSeek 官方形状（MLA + 大专家）的 deep-gemm 预热需更大显存
（DeepSeek-V2-Lite FP8 权重 15.7GB > 本机 15.5GB；AWQ 版无官方发布）；granite +
deep_gemm 完整服务冒烟被上游 activation kernel bug 阻断（CUDA launch failure，
发生在预热成功之后，不影响缓存面结论）。复现无-toolkit 验证：
`pip uninstall nvidia-cuda-nvcc` 后 `smoke_test.sh --no-toolkit`。

### MoE 验证的关键发现（模型白名单的依赖闭包）

granitemoe.py 模块级依赖 mixtral.py——首轮白名单漏掉 mixtral 后，registry
**静默降级到 Transformers fallback**（不 crash、只打一条日志），MoE 走 HF eager
实现而非 sglang fused kernel，预热面作废。白名单加入 mixtral 后原生实现恢复。
教训：**白名单必须做依赖闭包检查**（白名单内模型 import 的白名单外模型文件会
静默降级）；后续 build_slim.py 应把 models 排除纳入 AST 检查（warn 级）。

其他实测事实：
- JIT 缓存全部集中在 `~/.cache/sglang/` 一个目录（jit + triton + deep_gemm + inductor +
  nv，三套 JIT 体系被 sglang 统一重定向管理）——打包器只需收集这一个目录
- granite + CUDA graph capture 有上游 bug（非 pinned 拷贝），冒烟用
  `--cuda-graph-backend-decode=disabled` 绕过（SGLANG_SMOKE_EXTRA_ARGS 透传）
- granite 缺 RTX 5060 Ti 的 fused MoE tuned config，走默认配置（性能次优不影响正确性）
- granite fp8 + deep_gemm 组合死于 activation kernel CUDA launch failure
  （上游 bug，发生在 DeepGEMM 预热成功之后；sglang 的 compile_deep_gemm 预编译入口
  `python3 -m sglang.compile_deep_gemm` 可在打包时单独跑，不受该 bug 影响）

实测体积（自包含目录估算的输入）：

| 组件 | 体积 |
|---|---|
| PBS Python 3.12.14 | 357 MB |
| 依赖 venv（瘦身后：nvidia 运行时 3.2G + torch 1.2G + sgl_kernel 1.2G + triton 691M） | **7.5 GB**（原 8.9G，-16%） |
| slim 源码树 | 126 MB |
| JIT 缓存（Qwen3-4B 场景） | 10 MB |
| **自包含目录合计（未压缩）** | **≈ 8.0 GB** |

### 依赖瘦身实验记录（2026-09-16，全部经 --no-toolkit 冒烟验证）

| 结果 | 包 |
|---|---|
| ✅ 已卸载 | torchaudio、torchcodec、av、timm、datasets(+pyarrow)、modelscope、blobfile、py-spy、watchfiles、anthropic、mistral_common、outlines、llguidance、interegular、tilelang(+numba/llvmlite)、nvidia-cutlass-dsl、nvidia-mathdx |
| ⚠️ 换 CPU 版 | torchvision（`common.py:98` 顶层 import decode_jpeg → 6.6M CPU 轮替代 CUDA 轮） |
| ❌ 不可卸（启动硬依赖） | pillow（`srt/utils/common.py:93`）、soundfile（`entrypoints/openai/audio_chunking.py:24` → http_server 链）、xgrammar（`function_call/inkling_detector.py:6` → server_args 链，114M）、gguf（quantization 注册表）、IPython（`sglang/__init__` → `sglang/utils.py:26`）、tokenspeed-triton（deepseek MLA 链） |

**关键发现（mathdx 实验）**：sglang JIT 缓存 key 指纹包含已安装包集合
（torch/flashinfer/deep_gemm/nvidia-mathdx/tvm-ffi 版本，见
`kernels/jit/utils/compile/cache.py`）。卸载 mathdx 后旧缓存全部 miss。
推论：**目标机的 pip 包集合必须与预热环境完全一致**——瘦身必须在预热之前
完成，且打包器需以 freeze 清单锁定。

已知限制：
- Qwen3-8B fp8 在线量化在 15.5GB 显存上 OOM（bf16 峰值 16.4GB），量化冒烟需更小模型或预量化权重
- modelscope_cache 的 Qwen3.5-4B 不完整（.incomplete），未使用

## 2026-09-16 零 patch 排除清单实测修正

零 patch 实际可排除（已验证）：disaggregation 传输后端 5 个、mem_cache/storage
远端后端 10 个、ray/、checkpoint_engine、legacy grpc_server、debug_utils 工具、
models 白名单（231 → 45 文件）。

AST 检查证伪的"看似可删"项（全部归层 3，需 patch）：
- `hardware_backend/{npu,xpu,musa}` ← cuda_graph_setup.py:16-17、moe_runner/ascend.py 等裸 import
- `device_communicators/mooncake_transfer_engine` ← model_runner.py:37 模块级
- `distributed/gated_launch` ← bootstrap.py:23；`naive_distributed` ← host_shared_memory.py:10
- `sglang/test`（含 scripted_runtime 运行时组件）← scheduler.py 等 4 处模块级
- `sglang/benchmark`、`sglang/lang` ← 被 test/* 模块级引用
- `models/{dots3,inkling}_common` ← multimodal/lora 模块级引用

## 项目状态：产品可用态（2026-09-17 收尾）

核心命题全部实证：**零 diff 跟上游、srt -21%、双形态交付（tar 3.2G / 镜像 3.6G）、
目标机仅需 NVIDIA 驱动、dense + MoE + fp8 + 多模态（VL）全验证**。

### 日常维护（唯一例行事项）

跟上游 tag 重跑管线（约 15 分钟）：

```bash
git fetch --tags && git checkout <new-tag>
python3 packaging/build_slim.py                                # AST 契约自动把关
SGLANG_SMOKE_MODEL=<M> bash packaging/smoke_test.sh            # 预热 + 冒烟
SGLANG_SLIM_BUILD=<B> bash packaging/make_bundle.sh            # tar.zst
SGLANG_SLIM_BUILD=<B> bash packaging/make_image.sh --save      # OCI 镜像
```

### 部署触发清单（等真实条件，非开发任务）

| 项 | 触发条件 | 风险预判 |
|---|---|---|
| GPU 容器运行验证 | 有 nvidia-container-toolkit 的机器 | 低（镜像即已验证目录） |
| 跨机缓存移植（glibc 边界） | 第一台真实客户机 | 中（glibc ≥ 2.39 唯一真风险） |
| DeepSeek 形状 deep-gemm 预热 | 有 DeepSeek 部署需求 + 大显存 | deep-gemm 机制已验，只差官方形状 |

### 已评估放弃 / 按需的项

- **层 3 patch**（-21% → -43%）：放弃——收益 <0.5% 体积，代价是破坏零 diff 优势（负 ROI）
- **fatbin 瘦身**（-1G）：按需——客户抱怨下载体积再做；strip 会引入 arch 锁定 + 升级重验负担
- **Triton 预加载**：已自然解决——首次请求覆盖，且 0.78 mem-fraction 有 0.5G 余量

研究文档：`~/research/gpu-inference-binary-release/`（pruning-plan.md 为总计划）

