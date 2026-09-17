# Upstream sync runbook

sglang 上游机制画像（2026-09-17 实测）与 sglang-slim 的定期更新手册。

## 上游机制

| 维度 | 状态 |
|---|---|
| Release 节奏 | **双周**（周六 UTC，v0.5.x 系列，13-14 天/版；下一版预计 09-19 前后） |
| Release notes | 高质量结构化：新模型表（含 PR 链接）、性能特性、**依赖提示**（如 "Needs FlashInfer 0.6.18"） |
| PR 密度 | ~59/天 merged（单 release 786 PRs / 214 contributors） |
| 依赖 bump | 无固定周期、频繁；锚点 bd45cd50 之后已有 `sgl-deep-gemm 0.2.0`、`sglang-kernel 0.4.7` 两个 |
| Open 状态 | 907 issues / 4494 PRs（高流量仓库常态水位） |
| LTS | 无——只有滚动 release |

## 更新策略：跟 tag，不跟 main

main 每天 ~59 个 PR，不可能逐 commit 跟；tag 是稳定锚点。**每次 release 后重跑管线**
（约 15 分钟），流程本身就是为此设计的。

## 更新步骤

```bash
cd <sglang-checkout>
git fetch --tags && git checkout v0.5.20     # 新 tag

# 管线（顺序不可变：瘦身 → 装依赖 → 预热 → 打包）
python3 packaging/build_slim.py              # AST 契约自动把关（结构变化会在这里报警）
python3 packaging/install_deps.py --venv <V> --python <PBS_PY>
SGLANG_SMOKE_MODEL=<M> bash packaging/smoke_test.sh          # 预热 + 冒烟
bash packaging/smoke_test.sh --crash-on-jit                  # 缓存命中验证
SGLANG_SLIM_BUILD=<B> bash packaging/make_bundle.sh          # tar.zst
SGLANG_SLIM_BUILD=<B> bash packaging/make_image.sh --save    # OCI 镜像
```

## 上游变化的四类影响与应对（按预期频率排序）

| 变化 | 频率 | 信号 | 应对 |
|---|---|---|---|
| JIT 指纹依赖 bump（torch/flashinfer/deep_gemm/mathdx/tvm-ffi） | 每版都可能 | smoke 第一轮变慢（重编译）或 crash-on-jit 失败 | **流程内正常事件**：重预热即可（~1-2 分钟），无代码改动 |
| 新模型加入 registry | 每版（notes 的模型表） | release notes 模型表 | 有客户需求才加 `MODELS_KEEP_PREFIXES`；闭包自动展开 |
| kernels/jit 或模块结构变化 | 低频 | `build_slim.py` AST 契约报警（file:line 列出违规 import） | 修 `prune_manifest.py`（把违规项移回保留或调整模式） |
| 上游行为变化（缓存 env 名、CRASH_ON_JIT 语义、launcher 约定） | 极低频 | smoke/e2e 失败 | 对照报错修 `make_bundle.sh` 的 launcher |

## 已知陷阱（历史教训，更新时优先检查）

1. **pip 解析漂移**：重装依赖时 pip 可能拉高版本（曾把 torch 2.13 拉到 2.14）——
   install_deps 后必须 `pip list` 核对与 freeze.txt 的一致性，发现漂移用 `==` pin 回
2. **闭包依赖的外部包**：白名单闭包只覆盖"文件"，不覆盖 pip 包
   （inkling → cutlass-dsl）；新模型跑不了时先查 serve log 的 "Ignore import error"
3. **flashinfer 缓存行为**：0.6.18 的 JIT 加载器每次启动 spawn ninja（bundle 自带）；
   flashinfer 大版本升级可能改变此行为——e2e_verify 的 env -i 会暴露
4. **toolchain 在场掩盖预热缺口**：冒烟必须跑 `--no-toolkit` / e2e 的 env -i 模式，
   否则 JIT miss 会被现场编译静默掩盖

## 可选：自动监控

```bash
# 每日检查新 release（cron 或手动）
gh api repos/sgl-project/sglang/releases/latest --jq .tag_name
# 或订阅 atom feed: https://github.com/sgl-project/sglang/releases.atom
```
