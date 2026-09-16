# Slim single-node packaging (layer 0/2, zero sglang diff)

本目录是 sglang 单机精简发行版的构建脚手架。核心约束：**对 sglang 源码零 diff**——
上游更新只需重跑构建管线，不存在 rebase。

## 用法

```bash
python3 packaging/build_slim.py            # 装配 + 双重验证 → dist/sglang-slim/
python3 packaging/install_deps.py --venv <dir> --python <pbs-python>   # 依赖装配
bash   packaging/smoke_test.sh             # 冒烟（首轮 = JIT 预热）
bash   packaging/smoke_test.sh --crash-on-jit   # Go/No-Go：缓存命中验证
```

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

结论：**预生成 JIT 缓存替代 CUDA toolkit 的路线在本机验证成立**（Qwen3-4B/dense 场景）。
MoE 模型（deep-gemm 3072 kernel 预编译）与真无-toolkit 环境（换机或卸 nvcc 轮子）仍待验证。

实测体积（自包含目录估算的输入）：

| 组件 | 体积 |
|---|---|
| PBS Python 3.12.14 | 357 MB |
| 依赖 venv（nvidia 轮子 3.6G + torch 1.2G） | 8.9 GB |
| slim 源码树 | 126 MB |
| JIT 缓存（Qwen3-4B 场景） | 10 MB |
| **自包含目录合计（未压缩）** | **≈ 9.4 GB**（压缩后约 4-5 GB；依赖层 2/3 剔除后可再减） |

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

## 下一步

1. 依赖瘦身：剔除多模态（torchvision/torchaudio/av/timm）与未用依赖 → venv 8.9G 目标 <7G
2. MoE 模型冒烟（Qwen3-30B-A3B 之类，验证 deep-gemm JIT 预热面）——需更大显存或量化权重
3. 真无-toolkit 验证：卸 venv 内 nvidia-cuda-nvcc 轮子 + 屏蔽系统 CUDA 路径后重跑 --crash-on-jit
4. 自包含目录打包器（tar.zst + launcher 脚本）+ OCI 镜像包装
5. 缓存可移植性：把 ~/.cache/sglang 等搬到新目录/机器验证 key 无绝对路径漂移

研究文档：`~/research/gpu-inference-binary-release/`（pruning-plan.md 为总计划）

