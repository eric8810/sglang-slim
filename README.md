# sglang-slim

**SGLang 的单机精简发行版构建器** —— 把 [sglang](https://github.com/sgl-project/sglang)
裁剪成面向单机（本地 PCI 互联）NVIDIA GPU 的轻量推理服务，并产出免安装的自包含交付物。

## 核心思路

| 层 | 做法 | 对 sglang 源码的改动 |
|---|---|---|
| 0 | 仓库级裁剪（顶层目录不进产物） | **零 diff** |
| 2 | 构建期文件排除（跨节点传输后端、远端 KV 存储、非白名单模型等） | **零 diff** |
| 1 | 运行时白名单（`SGLANG_DISABLED_MODEL_ARCHS`） | **零 diff** |
| 3 | 静态 import 延迟化（可选，逐项独立交付） | fork + tag 同步 |

零 diff 是本项目的核心约束：**上游 sglang 更新时只需重跑构建管线，不存在 rebase**。
"零 patch 契约"由构建器内置的 AST 检查器自动保证（任何模块级 import 引用被排除
模块即构建失败），不依赖人工记忆。

## 验证状态（2026-09-16，本机实测）

| 实验 | 结果 |
|---|---|
| 构建 + AST 契约 | ✅ srt 792,340 → 626,206 行（-21%），排除 356 文件 |
| Qwen3-4B bf16 冒烟 | ✅ 60s healthy（含首轮 JIT），生成正常 |
| **Go/No-Go：JIT 缓存命中** | ✅ `SGLANG_CRASH_ON_JIT_COMPILE=1` 冷启动 25s，**零编译触发**，输出一致 |
| 自包含目录体积 | ≈ 9.4 GB 未压缩（PBS Python 357M + 依赖 8.9G + slim 树 126M） |

> Go/No-Go 的意义：预生成 JIT 缓存可以**完全替代目标机上的 CUDA Toolkit**，
> 目标机只需要 NVIDIA 驱动。

## 用法

前提：一个 sglang 检出（本仓库验证锚点：`bd45cd50`，2026-09 快照）+ 一块 NVIDIA GPU
+ [python-build-standalone](https://github.com/astral-sh/python-build-standalone) 解释器。

```bash
# 1) 构建精简源码树（在 sglang 检出内，或用 --src 指向其 python/sglang）
python3 packaging/build_slim.py

# 2) 装配依赖（venv 内含全套 nvidia cu13 轮子，CUDA 库随包携带）
python3 packaging/install_deps.py \
  --venv ./build-slim/venv \
  --python <python-build-standalone>/bin/python3

# 3) 冒烟（首轮 = JIT 预热）
SGLANG_SMOKE_MODEL=/path/to/Qwen3-4B bash packaging/smoke_test.sh

# 4) Go/No-Go：验证缓存命中、无运行时编译
SGLANG_SMOKE_MODEL=/path/to/Qwen3-4B bash packaging/smoke_test.sh --crash-on-jit
```

## 目录

```
packaging/    构建脚手架（manifest + build/install/smoke 三件套）
research/     完整调研资产：binary 形态先例、Python→binary 打包技术、
              上下游同步实践（含独立复核记录）与精简实施计划
```

## 路线图

- [ ] 依赖瘦身（剔除多模态依赖，venv 8.9G → <7G）
- [ ] 真无-toolkit 环境 Go/No-Go（卸 nvcc 轮子 + 屏蔽系统 CUDA）
- [ ] MoE 模型预热面验证（deep-gemm 3072 kernel 预编译）
- [ ] 自包含目录打包器（tar.zst + launcher）与 OCI 镜像包装
- [ ] JIT 缓存跨机器可移植性验证（换机后 key 命中）
- [ ] 层 3 patch 集合（speculative/lora/multimodal 延迟 import 化，目标 srt -43%）

## License

本仓库构建脚手架与文档以 Apache 2.0 发布（见 LICENSE）。
产物中包含的 sglang 及其依赖遵循各自的 license；对外分发 binary 时须随附
sglang 的 LICENSE 与第三方声明（Apache 2.0 条款 4a/4b/4c）。
