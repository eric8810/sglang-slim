# GPU 推理服务的 binary 交付形态调研

## 问题与用途

用户目标：做一个"精巧干练"的 GPU 推理服务 release 产品。
约束与偏好：
- 开发以 Python 为主（贴近 sglang 上游，保持上下游同步能力）
- 对外 release 形态最好是 **binary**（而非 wheel/docker 那种"开发者形态"）
- 场景：单机本地（PCI 互联）单卡/多卡推理（前期调研已确认 sglang 可精简 ~43% 代码 + 44% 仓库体积）

本次要回答：
1. 是否已有类似项目（binary/独立可执行形态的 LLM 推理产品）？
2. 别人是怎么做的（打包技术、CUDA 依赖处理、模型分发、上游同步策略）？
3. 对 sglang 这类重 Python + CUDA JIT + Rust 扩展的引擎，binary 化的可行路线是什么？

## 当前结论与建议

### 结论 1："Python 主开发 + sglang 上下游 + 严格单文件 binary"没有成功先例，工程上不可持续

（措辞已按复核修正：技术上 PyInstaller onefile 可打包 torch+CUDA 但实证脆弱，业界无任何"真编译成单文件"交付生产 LLM 服务的先例。）

业界 binary 形态推理产品分四类（详见 evidence-products.md）：

| 形态 | 代表 | 与本项目的兼容性 |
|---|---|---|
| 薄壳二进制 + 旁挂引擎 + 捆绑 CUDA 库 | ollama (Go+llama.cpp)、LocalAI、cortex.cpp | ❌ 前提是引擎本身是 C++，sglang 不是 |
| 单文件可执行 | llamafile、koboldcpp | ❌ llamafile 需运行时 nvcc JIT 编译 GPU 代码，服务端不可接受；且无法承载 Python 栈 |
| 桌面产品按 GPU 变体分发引擎 | Jan、LM Studio | ❌ 同样以 C++ 引擎为前提 |
| AOT 编译产物 + 极小 runner | MLC-LLM、ExecuTorch | ❌ 编译期定 kernel，无法套用到 Python 生态；但证明"运行时零 CUDA 库依赖"在理论上存在 |

保持 sglang（Python 引擎）时，**唯一可行形态是"自包含免安装发行包"**：嵌入式 Python 解释器 + 全部依赖轮子 + 捆绑 CUDA 运行库 + 预生成 JIT 缓存。这在 AI 产品界有官方先例（ComfyUI Portable 即此路线），但没有"真编译成单文件"的先例。

### 结论 2：体积上做不到"精巧"，体验上可以

- ollama 纯 C++ 引擎的 CUDA 发行包已达 **1.43GB**（GPU 运行库是绝对大头；macOS 无需捆绑仅 158MB）。Python 栈只会更大——估算 sglang 全家桶（torch + flashinfer + deep-gemm + CUDA 轮子）自包含包在 3-6GB 量级（未实测）。
- "精巧"应重新定义为**交付体验**：目标机只需 Linux + NVIDIA 驱动，零 Python/pip/nvcc/CUDA Toolkit 要求，解压即跑，模型旁路分发（参考 ollama 的 registry 模式）。

### 结论 3：sglang binary 化的工程路线（本地仓库已查证）

依赖栈中有 6 类 JIT/编译组件（详见 evidence-packaging.md 第 5 节表格）：

| 组件 | 编译模式 | binary 化处理 |
|---|---|---|
| sglang-kernel | AOT wheel（构建期） | ✅ 直接携带 |
| flashinfer | 预编译 + 首次使用 JIT | flashinfer-cubin 全架构包 + jit-cache 包 |
| sgl-deep-gemm | 运行时 JIT（启动预编译 3072 kernels） | 构建机预生成 DG_JIT 缓存随包分发 |
| triton / tilelang / cutlass-DSL / sglang 自有 JIT | 运行时 JIT (ninja+nvcc) | 构建机预生成 4 类缓存目录随包分发 |

关键事实（复核时源码逐一验证，`kernels/jit/utils/compile/cache.py`）：
- 缓存 key 指纹 = GPU arch + nvcc/c++ 版本 + 5 个包版本（torch/flashinfer/deep_gemm/nvidia-mathdx/tvm-ffi）；**绝对路径已显式归一化**（cache.py:109-145、267-268 源码注释明写 "a list written by one clone is readable from another"），跨机器不是障碍
- **必须一致的三要素：GPU arch、CUDA/编译器版本、包版本**；其余可跨机器
- **上游官方背书**：`SGLANG_CRASH_ON_JIT_COMPILE`（loader.py:131-137）命中缓存缺失即抛错，报错原文 "Seed the cache with prebuilt artifacts matching this environment"——这正是"构建机预热 → 随包分发 → 目标机只读"模式的官方支持，同时提供了验证手段：目标机开此开关冷启动不崩 = 命中集合完整

**建议路线**：
```
底座:   python-build-standalone（可重定位嵌入式 CPython）
引擎:   精简 sglang（构建期排除而非物理删除）
依赖:   全部 pip 轮子（CUDA 库走 nvidia-*-cuXX 轮子）
缓存:   构建机首启预热，收齐 4+1 类 JIT 缓存，随包分发
交付:   tar/AppImage 风格自包含目录（或薄壳 launcher 单入口）
验证:   目标机仅 NVIDIA 驱动，实测无 nvcc 环境冷启动
```

### 结论 4：上下游策略——把"删"变成"不启用"

- 业界五模式（evidence-upstream.md）：vendor+补丁（ollama）、长期 fork+rebase（Cohere←vLLM，AI 辅助后单次同步从数周降到数天）、组件化不 fork（NVIDIA NIM）、自研核心（TurboMind）、license 违规教训（ollama binary 漏 MIT 声明被追责）。
- **物理删除 43% 没有回流路径，冲突面比 Cohere 的"加法 fork"更不利**。建议：
  - 模型收窄用 `SGLANG_DISABLED_MODEL_ARCHS`（运行时白名单，零删码）
  - 跨节点模块删减改为构建期排除（打包时剔除文件，git 历史保留）
  - 必须改源码的部分尽量 PR 回上游（减小私有 diff）
  - 按 tag 固定节奏同步：上游 tag 密集可用（复核时 `git ls-remote --tags` 核实，v0.1.x→v0.5.9+ 含 patch/rc；本地仓库是浅克隆无 tag，勿以此推断），**pin 到某个稳定 tag** 再 rebase + git rerere + CI 全量回归
- license（已核对本地 sglang LICENSE）：Apache 2.0 条款 4 要求 binary 交付附带 LICENSE 全文、修改文件带声明、保留版权声明；不强制开源修改。建议产物携带 LICENSE + THIRD_PARTY_NOTICES。

## 覆盖范围

已覆盖：
- A. binary 形态产品盘点（10+ 项目，形态分类、CUDA 处理、模型分发）→ evidence-products.md
- B. Python→binary 打包技术（7 种打包器 + AOTInductor + JIT 组件清单本地查证）→ evidence-packaging.md
- C. 上下游实践（5 种模式、Cohere/ollama/NIM/TGI 一手来源、Apache 2.0 条款本地核对）→ evidence-upstream.md

未覆盖 / 不宣称：
- 未做实操打包实验（体积估算未实测）
- PyInstaller/Nuitka 交付 LLM 服务的商业案例未找到公开先例（可能存在但未公开）
- sglang JIT 缓存首次启动的完整命中集合需实测收集
- tilelang 预编译产物形态未确认

## 关键推理

1. "binary = 单文件"是误区：GPU 生态中静态链接 CUDA 驱动库不可行（NVIDIA 明确驱动必须动态），所以一切 GPU binary 本质都是"可执行 + 捆绑 .so + 依赖系统驱动"。ollama（最成功的同类产品）就是这么做的。
2. Python 栈不能被编译消除（Nuitka 只编译胶水，torch/flashinfer 的 .so 原样携带），但可以被**封装**（嵌入式解释器 + 自包含目录）。ComfyUI Portable 证明此路线对大型 GPU 应用成立。
3. sglang 的特殊障碍是运行时 JIT（6 类组件），而非 Python 本身；这些 JIT 全部有缓存机制，缓存可预生成——这是"目标机免工具链"的关键杠杆，也是本调研最重要的工程发现。
4. 上下游可持续性与删减方式直接相关：物理删除不可回流 → 应转为构建期排除/运行时白名单。Cohere 案例证明"小 diff + 固定节奏 + 自动化"可长期维持 fork。

## 关键未知

1. JIT 缓存的**完整命中集合**：预生成靠构建机首启预热，但惰性路径（特定模型/特性才触发的编译）可能漏项。验证机制已有：`SGLANG_CRASH_ON_JIT_COMPILE` + 按目标模型实测冷启动。**下一步最优先实测**。
2. `__has_include` 缺口（cache.py:35-39 自述）：依赖图可能因"文件是否存在"而非内容变化——极窄但真实的跨机器风险。
3. **DeepGEMM 预生成的硬件约束**：`compile_utils.py:338` 预热在 GPU 上分配 tensor，构建机需同 arch GPU + 足够显存（对多 arch 交付意味着多台构建机）。
4. 自包含包的真实体积（估算 3-6GB 未验证；注意 ollama 1.43GB 是压缩口径）。
5. sglang 是否存在 vLLM 式插件机制（若有，模式 C 可行性提升）。
6. Triton 在无 ptxas 环境的驱动回退行为一致性。

## 待用户决策

**已决策（2026-09-16 用户确认）**：容器是可选项，核心诉求是"方便安装与部署"（痛点：每台机器装 CUDA Toolkit 麻烦）。

由此确定交付形态：**自包含目录为核心产物 + 同目录包装 OCI 镜像为可选交付**：
- 关键事实澄清：驱动（不可逃避）≠ CUDA Toolkit（可完全避免）；sglang 需要 toolkit 仅因运行时 JIT，预生成缓存可替代
- pip 直装路线不解决痛点（无 toolkit 机器 JIT 会崩，vllm#49497 实证）
- 镜像内放自包含目录而非 pip install：Python/包版本/JIT 缓存全部构建期锁定，镜像即 immutable release artifact
- Go/No-Go 判据实验：无 toolkit 目标机 + `SGLANG_CRASH_ON_JIT_COMPILE=1` + 目标模型全链路冷启动，不崩即缓存命中完整

## 复用条件

本地仓库：/media/eric8810/fast-deliver/code/sglang（git bd45cd50，2026-09 快照）。
Web 调研：2026-09-16，来源以官方文档/GitHub release/工程博客为主（各 evidence 文档内附 URL）。
条件变化需重查：torch/flashinfer/deep-gemm 版本升级（JIT 缓存机制可能变化）、CUDA 大版本。

## 复核结果

2026-09-16 由独立复核 agent（未参与结论形成）复核：
- **判定：通过**。三问均有可操作结论，关键判断一手证据逐一验证为真（JIT 缓存机制源码、SGLANG_DISABLED_MODEL_ARCHS、pyproject 版本、上游 tag）。
- 修正已落地：结论 1 措辞软化（"不可兼得"→"无先例且不可持续"）；结论 3 补 SGLANG_CRASH_ON_JIT_COMPILE 官方背书与路径归一化事实；结论 4 补 tag 核实与"pin 到稳定 tag"；未知项收窄（跨机器障碍从"是否含机器信息"改为"arch/工具链/包版本三要素一致"+ 新增 __has_include 与 DeepGEMM 硬件约束）；新增"自包含目录 vs OCI 镜像"决策点。
- 复核确认不影响答案的支线（nix/Flatpak/体积口径）已记录于证据文档。
