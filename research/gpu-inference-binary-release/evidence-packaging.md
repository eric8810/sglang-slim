# GPU 推理服务 Python→binary 交付 — 可行性证据包

> 调研日期：2026-09-16。方法：WebSearch/WebFetch（官方文档、GitHub issue、release 说明优先）+ 本地仓库
> `/media/eric8810/fast-deliver/code/sglang` 源码查证（grep/read）。本文件只做技术与先例取证，未做打包实操实验。
> 结论用途：决定「Python 开发 → binary 交付」GPU 推理服务（基于 sglang）的技术路线。

---

## 0. 结论速览

| 方案/路线 | 对 PyTorch+CUDA 巨大依赖栈的可行性 | 证据强度 |
|---|---|---|
| PyInstaller（onedir） | 可行但脆：能收集 pip 轮子里的 CUDA .so，但版本冲突/漏收集是常见坑；onefile 不实用 | 高（有 issue 实证） |
| Nuitka（编译模式） | 只编译你写的 Python 胶水，torch/flashinfer 等 C 扩展仍按二进制整体携带；不消除 sglang 运行时 JIT | 中 |
| PyOxidizer | 已停维护，对 CUDA 栈不适用 | 高（作者自述） |
| python-build-standalone + pip 轮子 | **最现实的「免安装独立目录」路线**（ComfyUI 便携版同类）；CUDA 库靠 torch 的 nvidia-*-cu12 轮子携带 | 高（官方文档） |
| conda-pack | 可重定位 conda 环境 tarball，可含 CUDA 工具链；体积大（GB 级） | 中 |
| shiv / pex | 纯 Python zipapp；zip 内 .so 加载受限，**GPU 场景基本排除** | 中（机制推断，无 CUDA 先例） |
| staticx | 静态化 glibc 可行，但 CUDA 驱动库必须动态加载，无法完全静态；与 torch 栈兼容差 | 中 |
| torch.export + AOTInductor | 单模型可出 .so（无 Python）；但 sglang 这类服务框架（动态 batch、CUDA graph、自研 kernel、Python 调度）**不能全量导出** | 高（官方博客） |
| **嵌入 Python + 预构建 JIT 缓存随包分发**（自制） | 对 sglang 最对症：wheel 已预编译 sgl-kernel/flashinfer，DeepGEMM/Triton/sglang-JIT 的缓存目录构建期预生成后随包分发 | 本地证据充分 |

**核心判断**：对 sglang 这类「运行时 JIT 深度依赖」的引擎，**没有一种打包器能让你摆脱工具链依赖**；
binary 交付的正确形态是「独立目录 = 嵌入 Python + 全部 pip 轮子（含 CUDA 库）+ 预生成的各 JIT 缓存」，
目标机只要求 NVIDIA 驱动 + glibc 兼容，**不要求 CUDA Toolkit / nvcc**（前提：所有 JIT 路径都有预编译缓存或驱动回退）。

---

## 1. Python→binary 打包技术能力边界

### 1.1 PyInstaller

- 机制：打包 CPython 解释器 + 依赖 .so + 数据为 `onedir`/`onefile`（启动时解压）。不是编译。
- PyTorch+CUDA 可行性：可行，但全部依赖「正确收集 nvidia pip 包里的 .so」。实证坑：
  - pyinstaller/pyinstaller#8154（2023-12）：PyTorch 2.1.1+cu121 打包后 `Could not load library libcublasLt.so.11`，
    与随包携带的 libcublasLt.so.12 版本错乱，需手工 `ln -s` 修正。
    https://github.com/pyinstaller/pyinstaller/issues/8154
  - StackOverflow：PyInstaller 不会自动包含 CUDA toolkit（需显式收集 CUDA 库）。
    https://stackoverflow.com/questions/59074416/does-pyinstaller-include-cuda
  - discuss.pytorch.org：PyInstaller 产物跨机器部署时，目标机仍须有匹配的 NVIDIA 驱动；CUDA 运行库可随包，
    驱动库（libcuda）不能。
    https://discuss.pytorch.org/t/how-to-deploy-a-pytorch-detector-built-by-pyinstaller-in-ubuntu-to-another-host-with-cuda-of-different-version/192842
- 体积/启动：onefile 每次启动解压（秒级~十秒级），onedir 启动快；torch 栈包体 2–6 GB 级。
- 对 sglang 的适配问题：sglang 大量运行时 JIT（nvcc/Triton/DeepGEMM）不经过 PyInstaller 收集，
  且 PyInstaller 的收集对 flashinfer 的动态加载（dlopen cubin 路径）支持差。未找到 PyInstaller 交付大型 LLM 推理服务的公开案例。

### 1.2 Nuitka（编译模式）

- 机制：把**你自己的 Python 代码**编译为 C，第三方 C 扩展（torch 等）作为二进制依赖原样携带；`--mode=standalone/onefile`。
  https://nuitka.net/user-documentation/user-manual.html
- 社区证据：torch 数据科学部署到客户机场景存在（Nuitka#2218），可行但配置繁琐、体积大。
  https://github.com/Nuitka/Nuitka/issues/2218
- 对 sglang：只编译 Python 胶水层，**不消除** nvcc/Triton/DeepGEMM 运行时 JIT；收益低于成本。无 GPU LLM 先例。

### 1.3 PyOxidizer

- 维护者 Gregory Szorc 2024-03 博客宣布重心转移，项目事实上停更。
  https://gregoryszorc.com/blog/2024/03/17/my-shifting-open-source-priorities/
  https://github.com/indygreg/PyOxidizer
- 定位于「嵌入 Python 的单一可执行文件」，对 GB 级 CUDA 依赖栈 + 运行时 JIT 组件不适用。**排除**。

### 1.4 python-build-standalone（astral-sh）+ pip 轮子

- 官方定位：真正独立、可重定位的 CPython 发行版（解压即用），现由 Astral 维护。
  https://github.com/astral-sh/python-build-standalone
  https://astral.sh/blog/python-build-standalone
- 与 CUDA 的关系：自身不含 CUDA；CUDA 库由 PyTorch 的 pip 轮子（`nvidia-*-cu12/cu13` 全家桶：
  cudart/cublas/cublasLt/cudnn/cufft/nccl/nvrtc/nvtx/cusparse/cusolver 等）以 .so 形式带入，随环境整体分发。
- 这正是 ComfyUI Portable / Fooocus 一类的「嵌入 Python + pip 轮子」独立目录方案的技术底座。
  **结论：本路线是「免安装独立目录」的最现实实现方式。**

### 1.5 conda-pack

- 把 conda 环境打成可重定位 tarball，目标机解压即用；conda 渠道有 cuda-toolkit/cuda-* 包，可整包携带 CUDA 运行库。
  https://conda.github.io/conda-pack/
- 限制：体积 GB 级、需 libmamba/conda 生态、对 PyPI-only 组件（flashinfer、sgl-deep-gemm）要混装。
  可行性中；未被大型 GPU 推理产品公开采用。

### 1.6 shiv / pex

- shiv：zipapp 不落盘解压，.so 从 zip 直接 dlopen 受限 → CUDA 扩展基本不可行（未找到成功先例，标注未确认）。
- pex：支持 venv/解压模式，理论上可装 .so，但面向纯 Python 工具分发；无 CUDA 场景先例。
- **GPU 场景均排除**。

### 1.7 staticx

- 作用：把 PyInstaller 产物静态化 glibc，去除对目标机 libc 版本依赖；代价是启动解压更慢。
  https://github.com/JonathonReinhart/staticx
- CUDA 本质限制：NVIDIA 官方论坛确认 CUDA 驱动库（libcuda）必须由宿主驱动动态提供，无法完全静态化；
  静态链接 cudart 存在但驱动层永远动态。
  https://forums.developer.nvidia.com/t/is-there-a-way-to-create-a-completly-static-cuda-binary/313487
- 对 torch/flashinfer 大量 .so 的兼容性差。**排除**。

---

## 2. PyTorch 官方 AOT 路线：torch.export + AOTInductor

- 能力：`torch._export.aot_compile` 把导出图编译为**自包含共享库 .so**：
  「包含所有编译器生成的 Triton kernel 的预编译 cubin，保证不需要任何运行时编译；只依赖一个小型运行 ABI，无 CPython 依赖，
  可跨 libtorch 版本使用」。入口说明见：
  https://dev-discuss.pytorch.org/t/whats-the-difference-between-torch-export-torchserve-executorch-aotinductor/1642
  https://blog.ezyang.com/2024/12/ways-to-use-torch-export/
- 限制（同源证据，ezyang blog）：
  1. 必须可 `fullgraph` 导出（非 strict 可部分绕过）；输入/输出限定为 torch 支持类型（pytree of Tensors）。
  2. **禁止重编译**：动态 shape 必须显式声明；无 guard 分发机制，无法一张图吃任意形状组合。
  3. 必须是单图；多图需手工拆子网再胶合。
  4. 真实 LLM 应用中 tokenizer、sampling 等非张量逻辑不可导出，需自写 C++ 侧调度。
  5. 导出前有大量工作（几十个 workaround 是常态）与数值验证。
- CUDA graph 交叉证据：动态 shape 会禁用 CUDA graph 捕获；AOTInductor .so 在捕获场景有失败 issue。
  https://github.com/pytorch/pytorch/issues/158834
- **LLM 生产案例**：torchchat（PyTorch 官方）提供端到端 AOTInductor 服务器侧推理示例（`runner/run.cpp`）。
  https://github.com/pytorch/torchchat
  注意：torchchat 面向单模型/本地/小规模服务；**没有 sglang/vLLM 级（动态 batch + CUDA graph + 多模型 +
  自研 kernel 全栈）整体 AOT 导出的公开案例**（标注未确认）。
- **对 sglang 的判断**：整体 export 不可行；AOTInductor 只适用于把个别算子/子图预编译成 .so 的混合路线，
  与「预生成 JIT 缓存」路线相比收益低、工作量高。

---

## 3. GPU 应用 Python 打包的已知坑

### 3.1 CUDA 扩展模块（.so 内链接 libtorch_cuda/libcudart/libcublas 等）

- 打包时 CUDA 库版本错乱是头号坑（见 §1.1 pyinstaller#8154）。
- 正确做法（社区共识）：随包分发 PyTorch pip 轮子自带的 `nvidia-*-cuXX` 全家桶，而非依赖目标机系统 CUDA；
  目标机只保证 NVIDIA 驱动 ≥ torch 构建要求的最低版本。

### 3.2 flashinfer — 三包机制（关键）

官方文档明确 flashinfer 的分层交付：
- `flashinfer-python`：核心包，「首次使用时编译/下载 kernels」；
- `flashinfer-cubin`：**全部支持 GPU 架构的预编译 cubin 包**；
- `flashinfer-jit-cache`：按 CUDA 版本（cu129/cu130/cu134）的**预构建 kernel 缓存包**；
- 官方建议离线/快速初始化场景两者都装：「This eliminates compilation and downloading overhead at runtime」。
  https://docs.flashinfer.ai/installation.html
- 实证坑：vLLM#49497 — 默认 wheel 安装（无系统 CUDA Toolkit、无 nvcc）下，FlashInfer sampler 走
  `flashinfer/jit/cpp_ext.py` 的 JIT 路径，`RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`，
  引擎启动崩溃。说明**即使默认预编译安装，仍有部分模块会触发 JIT，需要 nvcc 或配套 cubin/jit-cache 包**。
  https://github.com/vllm-project/vllm/issues/49497

### 3.3 DeepGEMM（sgl-deep-gemm）— 全量运行时 JIT

- 官方 README：「All kernels are compiled at runtime through DeepJIT, requiring no CUDA compilation during installation」
  —— 安装无需编译，但**运行时全部 kernel 现场编译**。
  https://github.com/deepseek-ai/DeepGEMM
- DeepJIT 为独立库，缓存为共享目录，可跨机器复用编译产物（「Matching compilation inputs and cache tags allow users
  to reuse each other's compiled kernels」）。
  https://github.com/deepseek-ai/DeepJIT
- sglang 侧本地证据（见 §5）：`DG_JIT_CACHE_DIR`、`DG_JIT_USE_NVRTC`（默认 0 = 用 nvcc；
  注释明言 NVRTC 有性能损失且「NVCC JIT speed is also 9x faster」）、启动期预编译数千 kernels（fast-warmup 模式 3072 个）。
  → **binary 交付必须预生成 DeepGEMM JIT 缓存并随包分发；无 nvcc 目标机默认不可用（除非切 NVRTC）。**

### 3.4 Triton — 运行时编译链 + ptxas 依赖/回退

- 编译链：AST → Triton-IR → TTGIR → LLVM-IR → PTX → cubin（NVIDIA 路径）。
  https://pytorch.org/blog/triton-kernel-compilation-stages/
- 工具链依赖：cubin 生成需要 `ptxas`；常见错误 `RuntimeError: Cannot find ptxas`。
  https://discuss.pytorch.org/t/cannot-find-ptxas/195784
- 回退机制：ptxas 失败时可回退「由驱动 JIT 编译 PTX」（"Failed to compile generated ptx with ptxas, falling back to
  compilation by driver"）——即**无工具链目标机仍可跑，但走驱动回退**。具体版本行为标注部分未确认。
- Triton wheel 自带 ptxas 存在与 CUDA 版本匹配问题（pytorch#163801「Triton Wheel Missing CUDA13 ptxas」）。
  https://github.com/pytorch/pytorch/issues/163801
- torch.compile / torch._inductor 深度依赖 Triton，`~/.triton/cache` 为 kernel 缓存（可预生成后随包分发）。

### 3.5 tilelang / nvidia-cutlass-dsl（CuTe DSL）

- tilelang：`tilelang.jit` 运行时编译（sglang 用于 NSA/DSA 稀疏注意力等 kernel）。
- nvidia-cutlass-dsl：sglang 的 CuTe DSL JIT 有专用 AOT 持久缓存（`cute_aot_cache.py`，见 §5）。
- tilelang 是否有预编译 wheel 产物：**未确认**；默认视为运行时 JIT。

---

## 4. 实际先例（公开产品）

| 产品 | 交付形态 | 与 PyTorch/CUDA 的关系 | 来源 |
|---|---|---|---|
| **ComfyUI Portable**（官方） | Windows 便携 zip：`python_embeded`（嵌入 Python）+ 按 GPU 厂商拆分包（NVIDIA CUDA 13/Py3.13、CUDA 12.6/Py3.12 老卡、AMD ROCm、Intel），解压即用 | 嵌入解释器 + pip 轮子（torch cu 版）；非单文件、非编译 | https://docs.comfy.org/installation/comfyui_portable_windows |
| **Fooocus**（lllyasviel） | Windows 一键包：独立嵌入 Python 3.10 + 预装 torch 环境 | 同上 | https://github.com/lllyasviel/Fooocus （社区帖：https://github.com/lllyasviel/Fooocus/discussions/4002 ） |
| **AUTOMATIC1111 SD WebUI** | 便携版 = 嵌入 Python + 逐个 pip 安装包（官方自己承认体量 5 GB+，仍是「zip+脚本」，非二进制编译） | 同上 | https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1363 |
| **torchchat**（PyTorch 官方） | AOTInductor 把 LLM 编译成 .so，C++ runner 部署（服务器推理示例） | 真 AOT（无 Python 运行） | https://github.com/pytorch/torchchat |
| **SGLang 官方生产交付** | Docker 镜像（含 NVIDIA NGC 容器），生产按容器交付；RFC 讨论默认切 CUDA 13.0 | 容器 = 事实上的「binary 交付」载体 | https://docs.sglang.ai/get_started/install.html 、 https://catalog.ngc.nvidia.com/orgs/nvidia/containers/sglang 、 https://github.com/sgl-project/sglang/issues/20486 |

**未确认/未找到**：PyInstaller 或 Nuitka 交付大型 LLM 推理服务（vLLM/sglang 级）的公开商业案例；
SD/ComfyUI 生态的「便携版」均为嵌入解释器路线，无一走真编译或单文件。LM Studio/Ollama 等非 Python 栈，不计。

---

## 5. sglang 本地仓库 JIT 依赖清单（源码查证）

> 仓库：`/media/eric8810/fast-deliver/code/sglang`（commit 约 2026-09-12）。计数方法：`grep -rc` 遍历
> `python/sglang/**/*.py` 全量匹配，`occ` = 出现行数合计，`files` = 命中文件数。

| 组件 | 版本（python/pyproject.toml） | 计数 occ/files | 编译模式 | 本地证据（文件:行） |
|---|---|---|---|---|
| torch | `torch==2.13.0`（pyproject.toml:6,89） | — | torch.compile→Triton 运行时 JIT | — |
| sglang-kernel（import 名 `sgl_kernel`） | `sglang-kernel==0.4.6.post1`（:80） | 739/251 | **AOT**：CMake+nvcc 在 wheel 构建期编译，随 pip wheel 分发 | `python/sglang/kernels/aot/README.md`（「source tree lives under …/aot/，Python import path remains sgl_kernel」） |
| flashinfer_python | `flashinfer_python[cu13]==0.6.18`（:42） | 2263/259 | 预编译 wheel + 首次使用 JIT（cubin/jit-cache 可选包） | import 点如 `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` |
| sgl-deep-gemm（import 名 `deep_gemm`） | `sgl-deep-gemm==0.1.7`（:79） | 602/82 | **运行时 JIT**（DeepJIT，默认 nvcc） | `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:38-51`（`SGLANG_JIT_DEEPGEMM_PRECOMPILE`、`DG_JIT_CACHE_DIR`、`DG_JIT_USE_NVRTC`=0、fast-warmup 3072 kernels）；`entrypoint.py:18`（`ENABLE_JIT_DEEPGEMM`） |
| tilelang | `tilelang==0.1.12`（:84） | 233/23 | 运行时 JIT | `python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py`、`python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` |
| triton | 随 torch | import 632/327 | 运行时 JIT（ptxas 或驱动回退） | `python/sglang/srt/layers/moe/moe_runner/triton_kernels.py`、`python/sglang/kernels/ops/.../triton_*` 多个目录 |
| nvidia-cutlass-dsl（CuTe DSL） | `nvidia-cutlass-dsl[cu13]==4.6.2`（:55） | 「cutlass」4003 | **运行时 JIT + 专用 AOT 持久缓存** | `python/sglang/kernels/jit/cute_aot_cache.py:1-8` |
| sglang 自有 JIT CUDA kernels | 仓库自带（`kernels/jit/csrc/` 24 个 kernel 组） | `load_jit(` 164 处；import `kernels.jit` 179 文件 | **运行时 ninja+nvcc 编译**，内容寻址持久缓存 | `python/sglang/kernels/jit/utils/compile/loader.py:48`（`load_jit` 入口）、`toolchain.py:53-62`（device compiler = `$CUDA_HOME/bin/nvcc`）、`cache.py:301-308`（缓存根 `~/.cache/sglang/jit`） |
| nvshmem4py-cu13 | （:58） | — | 预编译 wheel（分布式，非 JIT） | — |
| tvm_ffi | 传递依赖 | 205 | 预编译 wheel（JIT .so 加载器） | `toolchain.py:99-110` |

### 5.1 sglang JIT 机制要点（源码事实）

1. **`sglang.kernels.jit`（RFC #29630 内部 JIT 树）**：`load_jit()` 生成 `build.ninja`，运行时调用
   `{CUDA_HOME}/bin/nvcc` + host `c++` 把 `csrc/**/*.cu` 编译成 .so，经 `tvm_ffi` dlopen 加载。
   `toolchain.py:53-62` 明确 device compiler 路径解析（`CUDA_HOME`→`nvcc`→`/usr/local/cuda`），
   **目标机必须有 nvcc（或 CUDA Toolkit）才能首次编译**。
2. **持久缓存**：内容寻址（`build_key`=源码+编译参数+环境指纹含 nvcc/c++ 版本与各包版本；`deps_key`=传递依赖文件内容），
   缓存根 `~/.cache/sglang/jit`（`SGLANG_JIT_CACHE_DIR` 可重定向）。缓存 key 包含 flashinfer/deep_gemm/nvidia/cutlass
   头文件与 CUDA toolkit 头文件 → **预生成缓存须与交付环境版本完全一致**（`cache.py:109-131, 200-256, 301-308`）。
3. **`cute_aot_cache.py`（CuTe DSL AOT 缓存）**：把 nvidia-cutlass-dsl 编译的 .so 持久化到
   `{SGLANG_CACHE_DIR}/cute_aot`（默认 `~/.cache/sglang/cute_aot`），带文件锁跨进程共享；
   指纹含 python 版本、平台、cutlass/cuda 版本、目标 arch（`cute_aot_cache.py:44-90`）。
4. **`python/sglang/kernels/aot/`**：这是 **sglang-kernel 独立包**（与 jit 树并列的 AOT 路径），wheel 构建期
   CMake+nvcc 编译，PyPI 分发 → 这是仓库内唯一「预编译产物随 wheel 走」的先例。
5. **kernel 命名空间**：`sglang.kernels.ops.*` 为薄包装层，按需惰性选择后端（`sgl_kernel` AOT / `kernels.jit` /
   Triton），有 `SGLANG_FORCE_FUSED_OP_BACKEND` 开关（`python/sglang/kernels/__init__.py`）。
6. **DeepGEMM 启动预编译**：`compile_utils.py:37-80` — 启动期按 `_BUILTIN_M_LIST`（fast-warmup = 3072 个 kernel）
   批量 JIT 编译，缓存目录强制对齐 `SGLANG_CACHE_DIR`。

### 5.2 对 binary 交付的含义（本地事实 → 工程结论）

- 全量 AOT 化 sglang 不可行（164 处 `load_jit` 调用 + Triton/DeepGEMM/tilelang/CuTe DSL 四处独立 JIT）。
- binary 交付需覆盖四类缓存：`~/.cache/sglang/jit`、`~/.cache/sglang/cute_aot`、`DG_JIT_CACHE_DIR`、
  `~/.triton/cache` + flashinfer `cubin/jit-cache` 包。全部可在**构建机（带 nvcc/CUDA Toolkit）上预生成**，
  随包分发并在目标机首次启动时复制到约定缓存位（目录可经环境变量重定向）。
- 目标机最终依赖清单：NVIDIA 驱动（≥ torch/flashinfer 要求）、glibc 兼容、`libcuda`（驱动自带）；
  **不需要 nvcc/CUDA Toolkit**（前提是上述缓存全部命中；未命中的极端路径仍会尝试 nvcc 并失败）。

---

## 6. 未确认项清单

1. PyInstaller/Nuitka 交付大型 LLM 推理服务（vLLM/sglang 级）的公开商业案例 — **未找到**。
2. shiv/pex 跑通 CUDA 扩展的案例 — **未找到**（按机制推断不可行）。
3. tilelang 是否有官方预编译 wheel/AOT 产物 — **未确认**（默认运行时 JIT）。
4. Triton 无工具链时「驱动回退」在各版本（3.2/3.4/3.5/3.6/3.7）的行为一致性 — **部分未确认**（存在回退机制，
   具体版本行为与 ptxas 自带情况见 pytorch#163801）。
5. DeepGEMM 在 `DG_JIT_USE_NVRTC=1` 下无 nvcc 目标机的可用性与性能损失幅度 — 官方注释承认性能损失 + 慢约 9x，
   精确数据未确认。
6. flashinfer 默认 wheel 安装下**具体哪些**模块会走 JIT（已知 sampler 等）— 完整清单未确认（vllm#49497 只暴露了 sampler 路径）。
7. sglang 各 JIT 缓存首次启动的完整命中集合（哪些 kernel 一定在预生成清单内）— 未确认，需在构建机上实测收集。
