# 以 binary / 独立可执行 / 单文件形态交付的 LLM（GPU）推理产品调研

> 调研目的：为"精巧干练的 GPU 推理服务 release 产品"（Python 开发、binary 交付）提供形态选型依据。
> 调研时间：2026-09-16。所有结论均来自官方文档 / GitHub release 页面 / GitHub 仓库，标注来源 URL；查不到的标注"未确认"。
> 背景：用户场景是单机 NVIDIA GPU 服务器上的本地推理服务（非端侧 app），最相关参照是服务端 binary 产品（ollama / LocalAI / cortex），端侧产品（MLC / ExecuTorch）作为 AOT 技术参照。

---

## 0. 一页总结（形态分类矩阵）

| 形态 | 代表产品 | 交付物 | GPU 依赖处理 |
|---|---|---|---|
| 单文件可执行（自解压/嵌入式多 arch） | llamafile、koboldcpp | 1 个文件（含引擎+权重，或引擎捆绑 CUDA DLL） | llamafile：运行时 JIT 编译 GPU 源码（需系统编译器+驱动）；koboldcpp：预捆绑 CUDA DLL |
| 预编译二进制包（多可执行+捆绑 so） | llama.cpp releases、ollama | tar.gz/zip：主可执行 + 引擎二进制 + 运行时库 | 捆绑 CUDA/ROCm 运行时库；依赖系统内核驱动；按 GPU 类型分包/按需下载附加库 |
| 安装包/桌面 app（内嵌或运行时下载引擎） | LM Studio、Jan、gpt4all、cortex.cpp、Ollama(macOS) | .exe/.dmg/.pkg/.deb/.AppImage | 引擎按硬件变体分版本随 app 分发或首次运行下载；CUDA 库多捆绑或要求系统安装 |
| 动态库 + 薄壳（服务端框架式） | LocalAI、cortex.llamacpp、onnxruntime-genai、MNN/ncnn | 核心可执行（Go/C++）+ 按需拉取的 backend/引擎 .so/.dll | LocalAI：backend 是独立 gRPC 进程按需下载；ORT：NUGet/PyPI 包内含 DLL，GPU EP 仍需系统 CUDA+cuDNN；MNN/ncnn：单库 libMNN.so/libncnn.so |
| AOT 编译产物（模型编译进产物） | MLC-LLM、ExecuTorch | 编译出的模型库（.tar 包 / .pte 文件）+ C++/Swift/Kotlin runner | GPU kernel 在编译期由 TVM/ExportedProgram 生成；运行时依赖平台 API（Metal/CUDA/Vulkan） |
| 非独立 binary（对照） | TensorRT-LLM | NGC Docker 镜像 + Python wheel | 依赖整套 TensorRT+CUDA 环境，随镜像分发 |

**核心结论（对 sglang-based 产品）：** 服务端 binary 产品的通行做法是"**薄壳 + 内嵌/旁挂引擎二进制**"：
- 主服务（Go/C++/Rust）负责 API、调度、模型管理、进程拉起；
- 推理引擎（llama.cpp 系 or 自研）编译成独立 runner 二进制或动态库；
- GPU 运行时库（CUDA/ROCm）**捆绑随包分发，但依赖系统内核驱动**；macOS 因 Metal 为系统组件所以包最小（ollama darwin.tgz 仅 158MB vs Linux 1.4GB）；
- 模型权重**旁路分发**（内置下载器或注册表），不打包进 binary。
- 真"单文件"（llamafile）以牺牲 GPU 便利性（需要系统编译器 JIT）为代价，服务端场景一般不值得。

---

## 1. llama.cpp（ggml-org）

**交付形态**：多平台预编译二进制 tar.gz/zip（非单文件，含多个可执行 + 配套库），由 GitHub Actions 自动构建发布。Linux 按后端分多个包：Ubuntu x64/arm64/s390x (CPU)、Vulkan、CUDA 12/13、ROCm、OpenVINO、SYCL；Windows 有 CPU/CUDA/Vulkan/OpenVINO/SYCL/ROCm 版 zip；另有 macOS arm64/x64、Android arm64、iOS XCFramework。
来源：https://github.com/ggml-org/llama.cpp/releases （b10991 资产清单）

**GPU/CUDA 依赖处理**：CUDA 版包**分离出独立的 CUDA 运行时库包**（如 `cudart-llama-b10991-bin-ubuntu-cuda-12.8-x64.tar.gz`、Windows 的 `CUDA 12.4 DLLs` 包），与主二进制包并列提供 —— 即"捆绑 so/DLL 但单独成包"。仍需系统 NVIDIA 内核驱动。GitHub README 明确引导用户"Download pre-built binaries from the releases page"。
来源：https://github.com/ggml-org/llama.cpp/releases ；https://github.com/ggml-org/llama.cpp

**模型分发**：旁路文件 —— 用户自行下载 GGUF 模型（Hugging Face 等），运行时传模型路径，无内置下载器。

**引擎与产品关系**：自研（ggml/GGUF 生态本体），被 ollama、LM Studio、Jan、LocalAI、koboldcpp、gpt4all 等大量内嵌。官方定位即"inference library + server"，非打包产品。

**发布渠道与体积**：GitHub Releases 自动发布，每个 build 约每日一次。单个 CPU 包约几十 MB（未精确确认体积）；CUDA 版约 100-200MB（含库，未精确确认）。**注意**：2024-04 曾有一段时间 releases 中断，后恢复（社区讨论），说明依赖 CI 流水线。

---

## 2. ollama

**交付形态**：Linux/macOS 为 tar.zst/tgz 包 + `install.sh`（curl 管道脚本）；Windows 为 zip；macOS 还有 .app（安装到 /Applications）。Linux 安装脚本将主二进制放 `$BINDIR`（/usr/local/bin 等），**其余组件装到 `$OLLAMA_INSTALL_DIR/lib/ollama`**（含 llama.cpp 编译的 runner/引擎二进制与运行时库）。安装包由 GitHub Actions 自动发布。
来源：https://raw.githubusercontent.com/ollama/ollama/v0.34.1/scripts/install.sh ；https://github.com/ollama/ollama/releases

**GPU/CUDA 依赖处理**（关键做法）：
- 主包即含 CUDA 运行时库（Linux amd64 包 1.43GB 压缩，远超纯引擎体积，说明捆绑了大量 CUDA 库）；
- AMD GPU 用户在安装时被检测（lspci/lshw/nvidia-smi 检测 GPU 厂商），**额外下载独立的 `ollama-linux-amd64-rocm.tar.zst` 包**（1.05GB）；
- Jetson（JetPack R35/R36）系统额外下载 `ollama-linux-arm64-jetpack5/6.tar.zst`（约 270-297MB）；
- 检测不到 GPU 则提示 CPU-only 模式。**仍依赖系统内核驱动**（libcublas.so 缺失是常见故障，见 NixOS issue：https://github.com/NixOS/nixpkgs/issues/342385）。
来源：https://raw.githubusercontent.com/ollama/ollama/v0.34.1/scripts/install.sh ；https://github.com/ollama/ollama/releases

**模型分发**：内置下载器 + 注册表。`ollama pull` 从 ollama.com 的 registry 拉取模型，存到 `~/.ollama/models`（Linux systemd 安装为 `/usr/share/ollama/.ollama/models`），可用 `OLLAMA_MODELS` 环境变量改位置。模型与 binary 完全分离。
来源：https://docs.ollama.com/windows ；https://github.com/ollama/ollama/issues/733

**引擎与产品关系**：Go 主服务（API/scheduler）+ 内嵌 llama.cpp（vendored fork，仓库内 `llama/` 目录 + 持续"llama.cpp updates"changelog；社区确认它是 fork 并裁剪了 server 端点为自有 API）。近年（2025+）新增自研 runner（`runner/` 目录、MLX runner），开始摆脱纯 llama.cpp；Apple Silicon 用 MLX 引擎。**形态：Go 薄壳 + 独立 runner 进程。**
来源：https://github.com/ollama/ollama （仓库结构）；https://www.reddit.com/r/LocalLLaMA/comments/1cjaybn/how_ollama_uses_llamacpp/

**发布渠道与体积**（v0.34.1 实测）：
- ollama-linux-amd64.tar.zst ≈ 1.43GB（1,429,323,296 B）
- ollama-linux-amd64-rocm.tar.zst ≈ 1.05GB
- ollama-linux-arm64.tar.zst ≈ 1.55GB
- ollama-windows-amd64-mlx.zip ≈ 1.40GB
- Ollama-darwin.zip ≈ 198MB / ollama-darwin.tgz ≈ 158MB（Metal 为系统组件，无需捆绑 CUDA → 显著更小）
来源：GitHub API https://api.github.com/repos/ollama/ollama/releases/latest （v0.34.1 assets）

---

## 3. llamafile（Mozilla AI，原 Mozilla Builders / Justine Tunney）

**交付形态**：真·单文件可执行 —— shell 脚本（带 MZ 前缀）+ APE（Actually Portable Executable）封装，**同一文件原生运行于 macOS/Windows/Linux/FreeBSD/OpenBSD/NetBSD**，模型权重直接拼接/内嵌在文件尾部（ZIP 容器，页面大小对齐以便 GPU 直接 mmap 指针）。用户下载 1 个文件、chmod +x、运行。项目现由 mozilla-ai 维护且活跃（内含 llama.cpp submodule + patches；还扩展到 diffusionfile/whisperfile 等"file"系列）。
来源：https://docs.mozilla.ai/llamafile/reference/technical_details ；https://github.com/mozilla-ai/llamafile

**技术原理**：
- Cosmopolitan libc 静态链接实现 6 OS 兼容；AMD64 + ARM64 两套 llama.cpp 都编进去，启动时 shell 脚本提取 APE loader 映射二进制；
- 权重以 **ZIP 容器**内嵌（自写 zipalign 保证 4KB 对齐 → Metal 等 GPU 可直接用 mmap 指针读权重）；
- 微架构分派：SSSE3/AVX/AVX2 多版本编译，运行时用 `X86_HAVE()` 分派。
来源：https://docs.mozilla.ai/llamafile/reference/technical_details

**GPU/CUDA 依赖处理**（最重要的取舍）：Cosmopolitan 是静态链接，**无法静态链接 GPU 库**，因此 llamafile 把 `ggml-cuda.cu` / `ggml-metal.m` 源码打包进 zip，**运行时要求系统装有编译器**（nvcc / Xcode），现场编译为适配当前 GPU 微架构的模块，再经特殊实现的 dlopen 链入（ELF 上用一个 helper 可执行映射平台 ELF 解释器，跨 ABI 用 __ms_abi__/TLS 技巧）。即：**GPU 依赖系统驱动 + 系统编译器**；无编译器则退化为 CPU。
来源：https://docs.mozilla.ai/llamafile/reference/technical_details

**模型分发**：模型直接打包在文件内（分发即分发权重），也可用 `llamafile -m model.gguf` 加载外部模型。可运行本地 HTTP server。
来源：https://docs.mozilla.ai/llamafile/reference/technical_details ；https://github.com/mozilla-ai/llamafile

**引擎与产品关系**：内嵌上游 llama.cpp（vendored + patches），非自研引擎。当前状态：活跃（Mozilla.ai 接手续维护，"llamafile Returns"）。
来源：https://blog.mozilla.ai/llamafile-returns/ ；https://github.com/mozilla-ai/llamafile

**发布渠道与体积**：GitHub Releases（llamafile 本体 + 预打包含权重的模型 llamafile，权重文件可达数 GB，由体积决定分发渠道）。`assimilate` 工具可把单文件转成宿主平台原生可执行（放弃跨平台换取原生格式）。体积：不含权重的 llamafile 本体约几十 MB（未确认具体数字）。

---

## 4. koboldcpp（LostRuins，单文件典型）

**交付形态**：llama.cpp fork，官方定位"single self-contained distributable / single file executable"。Windows 发布单文件 exe（**CUDA edition 约 300MB，把 CUDA DLL 直接捆绑进 exe**；另有 CPU/Vulkan/ROCm/OpenBLAS 等变体）；Linux 通过官方构建脚本编译或社区包分发。
来源：https://github.com/LostRuins/koboldcpp ；https://www.reddit.com/r/LocalLLaMA/comments/14faz1d/building_koboldcpp_cuda_on_linux/ ；https://www.promptquorum.com/power-local-llm/koboldcpp-review

**GPU 依赖处理**：把 CUDA/ROCm/Vulkan 运行时 DLL 静态合入 exe（Windows 上无需安装 CUDA 即可跑 GPU）；Linux 上需自己编译对应后端。模型仍为旁路 GGUF 文件。
**发布渠道**：GitHub Releases（每版大量变体资产）。许可证 AGPL-3.0。

---

## 5. LM Studio（闭源桌面产品）

**交付形态**：桌面安装包（Windows .exe / macOS .dmg / Linux AppImage），官方下载站分发。闭源。
**引擎**：官方确认 runtime 基于 **MLX 与 llama.cpp**（"Powered by the LM Studio runtime, with MLX and llama.cpp under the hood"）。引擎与 UI 打包在安装包内，GGUF 模型在 app 内直接下载（内置模型浏览器，从 HF 镜像站拉取）。
**GPU 依赖**：随安装包捆绑对应平台运行时（macOS 用 Metal/MLX 系统框架；Windows/Linux 捆绑 llama.cpp 编译产物与 CUDA 库——细节闭源未公开，标注"未确认"：是否要求系统预装 CUDA toolkit）。
来源：https://lmstudio.ai/ ；https://news.ycombinator.com/item?id=49267928 （社区：LM Studio 内部用 llama.cpp 跑 GGUF）

---

## 6. Jan（开源桌面产品，janhq/jan）

**交付形态**：开源 Electron 桌面 app + 各平台安装包。**推理引擎不硬编码进主包**：Jan 的 llama.cpp 后端（含 CUDA/Vulkan 等硬件变体）作为可下载组件，在 app 内"Settings → Llama.cpp → 后端选择"中按硬件版本下载/更新（`win-avx2-cuda-cu12.0-x64` 等命名变体），安装到用户数据目录。v0.8.0 起用 llama.cpp 的 router 模式（单 `llama-server` 进程多模型调度）。**安装后端后 Jan 会检查系统是否缺 CUDA/Vulkan/cuDNN 并提示安装** —— 即 GPU 库仍倾向依赖系统环境，而非完全捆绑。
来源：https://www.jan.ai/docs/desktop/local-engine/llama-cpp

**模型分发**：app 内 Hub 下载（jan.ai hub / HF），或本地 GGUF 文件 in-place 链接（不复制）。
**引擎与产品关系**：内嵌上游 llama.cpp；2025 年曾自研 Cortex 壳（janhq/cortex.cpp）作为推理后端，后于 2025-07 放弃并回退到直接用 llama.cpp（Jan issue #4941：deprecates Cortex）。
来源：https://github.com/janhq/jan/issues/4941

---

## 7. LocalAI（mudler，Go 服务端）

**交付形态**：Go 编写的一个主二进制（OpenAI 兼容 API 核心）+ **独立 gRPC backend 进程**。官方架构：核心 Go API 通过 gRPC 把请求路由到"按需拉取/管理"的 backend（llama.cpp 等 C++ 引擎编译为独立 backend 二进制）。用户可用 `local-ai backends install llama-cpp` 按需安装引擎。2025-08 起"Modular Backends"：llama.cpp、stablediffusion.cpp 等 backend 可独立更新（不再整体重建镜像）。
来源：https://localai.io/docs/reference/architecture/ ；https://www.reddit.com/r/LocalLLaMA/comments/1mo3j17/localai_major_update_modular_backends_update/

**GPU 依赖处理**：llama.cpp backend 编译时按 CUDA/CPU 等配置构建（backend 二进制含对应运行时）；后端数据缓存在 `/tmp/localai/backend_data`。细节（是否捆绑 CUDA so）未逐一确认。
**模型分发**：`local-ai models install`（从 HF/模型库下载）；模型目录与引擎分离。
**引擎与产品关系**：Go 薄壳 + 内嵌上游 C++ 引擎（llama.cpp 为主，多后端）。MIT 协议。
来源：https://localai.io/docs/reference/architecture/

---

## 8. cortex.cpp（menloresearch，前 janhq）

**交付形态**：服务端/本地 binary 产品。安装形态：Windows `cortex.exe`、macOS `cortex.pkg`、Linux Debian `cortex.deb`，其余发行版用 curl 安装脚本。`cortex start` 起本地 server（端口 39281），OpenAI 兼容 API。旧架构（cortex.llamacpp，2025-07 归档）：引擎编译为**动态库 `libengine.so/.dll/.dylib`**，约定放入 `engines/cortex.llamacpp/` 目录，由 server 运行时加载 —— "a dynamic library that can be loaded by any server at runtime"。
来源：https://github.com/menloresearch/cortex.cpp ；https://github.com/janhq/cortex.llamacpp （archived 通知，开发移往 menloresearch）

**GPU 依赖处理**：自动 GPU 检测（NVIDIA/AMD/Intel）。捆绑方式细节未确认（现为闭源商业产品形态，cortex.so）。
**模型分发**：`cortex pull` 从 Hugging Face hub 拉取模型（GGUF）。
**引擎与产品关系**：多引擎架构 —— 内嵌 llama.cpp 起步（menloresearch/llama.cpp fork），可加自定义引擎；Jan 时代其 C++ 引擎基于 llama.cpp fork + onnxruntime 等多后端。

---

## 9. MLC-LLM（mlc-ai，TVM 路线）

**交付形态**：模型被 TVM AOT 编译成**模型库（model library）**，用 `mlc_llm package` 打包成 `dist/lib/*.tar`（含编译出的平台专用二进制/库）+ `dist/bundle/`（权重），供 iOS/Android 桌面 app 集成；同一套 runtime 库（libmlc）嵌入宿主 app。提供 iOS Swift SDK / Android SDK / Web(WASM) / 桌面。
来源：https://llm.mlc.ai/docs/compilation/package_libraries_and_weights.html ；https://github.com/mlc-ai/mlc-llm

**GPU 依赖处理**：GPU kernel 在**编译期**由 TVM 针对目标后端生成（Metal/CUDA/Vulkan/OpenCL 等 TVM 后端），运行时链接平台 API。桌面/服务端有 CUDA 后端支持（LLM-CUDA 编译配置）。权重与库分离打包，`bundle_weight: true` 时把权重一并打进 app。
来源：https://llm.mlc.ai/docs/compilation/package_libraries_and_weights.html

**模型分发**：模型从 Hugging Face 上的 MLC 预转换仓库（HF://mlc-ai/...-MLC）下载，或在 package 阶段 bundle 进 app。引擎：自研（TVM 编译器 + MLC runtime）。
**发布渠道**：pip（mlc-llm、tvm）、模型库发布在 Hugging Face、iOS/Android SDK 集成进宿主 app —— 本身不发布独立可执行产品（官方 app 是 MLCChat）。

---

## 10. ExecuTorch（pytorch/executorch，Meta）

**交付形态**：AOT 导出 + C++/Swift/Kotlin runner。PyTorch 模型经 `torch.export()` + backend 分区编译为单个 **`.pte` 文件**（含图与权重的二进制程序），由极小的 C++ runtime 加载执行；官方提供 `executor_runner` C++ 示例 + LLM 专用 runner（extension/llm 目录）与 tokenizer/sampler 配套。target 覆盖 iOS/Android/嵌入式（Cortex-M、ESP32）/桌面 GPU（Vulkan Linux+Windows）。LLM 支持度：Llama 3.2/3.3、Qwen、SmolLM2、Qwen3-VL 等导出与量化（int4）流程，release 附带预导出 .pte 快速上手。
来源：https://docs.pytorch.org/executorch/stable/using-executorch-export.html ；https://github.com/pytorch/executorch/releases （v1.4.1）

**GPU 依赖处理**：backend kernel 编译进 .pte/运行时（Vulkan、XNNPack、QNN、CoreML/MLX 等 backend），宿主系统提供图形/驱动 API；CUDA 不是其主目标（桌面 Vulkan 是新增能力）。
**模型分发**：.pte 文件旁路分发（模型仓库/release 资产）。
**引擎与产品关系**：自研 runtime（非内嵌第三方引擎）；定位是部署工具链而非产品。

---

## 11. onnxruntime-genai（微软）

**交付形态**：ONNX Runtime 的生成式 AI 扩展（C++/C#/Python/Java API），**以库形态发布**：NuGet 包（`Microsoft.ML.OnnxRuntimeGenAI.Managed/.Cuda`）、PyPI（`onnxruntime-genai`，含 CPU 与 GPU wheel）、nightly feed；OS 覆盖 Linux/Windows/Mac/Android。模型为 ONNX 格式（量化 int4 等），从 Hugging Face 下载（如 `microsoft/Phi-3-mini-4k-instruct-onnx`）。
来源：https://github.com/microsoft/onnxruntime-genai ；https://www.nuget.org/packages/Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.2.0

**GPU 依赖处理**：GPU EP（CUDA）要求**系统安装匹配版本的 CUDA + cuDNN**（官方文档明确"required to install CUDA and cuDNN"）；1.21.0 起提供 `preload_dlls` 预加载 CUDA/cuDNN/MSVC DLL。支持 DirectML/OpenVINO/QNN/WebGPU/TRT-RTX 等多 EP。
来源：https://onnxruntime.ai/docs/install/ ；https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

**模型分发**：旁路（HF ONNX 模型目录，含 tokenizer 与 generation config）。
**引擎与产品关系**：自研引擎（ONNX Runtime）的官方扩展。**形态即"动态库+薄壳"**：产品用它嵌入（Foundry Local、Windows ML、VS Code AI Toolkit 是它的宿主）。

---

## 12. MNN-LLM / ncnn（阿里 / 腾讯，端侧单库）

**MNN（alibaba/mnn）**：C++ 推理引擎，单库（`libMNN.so`/静态库）+ 转换工具链，覆盖服务器/PC/手机/嵌入式。MNN-LLM 是基于 MNN 的 LLM 运行时（transformers 目录），模型经 LLM.py 转成 MNN 格式旁路分发，宿主 app 链接 libMNN 运行。支持多后端（CUDA/OpenCL/Vulkan/NPU）。形态：**动态库 + 宿主 app 集成，非独立可执行**。
来源：https://github.com/alibaba/mnn ；https://mnn-docs.readthedocs.io/en/latest/intro/about.html

**ncnn（Tencent/ncnn）**：腾讯端侧推理框架，"universal GPU acceleration with Vulkan，deploy on CPU & GPU across desktop & mobile"。ncnn-llm 为极简 LLM 推理实现（LLM 解码 C++，用 pnnx 转换模型，支持 Qwen 等），主要跑 Linux/Windows/macOS，可 Android 直接编译。形态同为**单库 + 可执行 demo/宿主集成**。
来源：https://www.khronos.org/developers/linkto/ncnn-universal-neural-network-inference-with-vulkan ；https://zhuanlan.zhihu.com/p/1983954636361709280 ；https://opensource.tencent.com/summer-of-code/project/103/issue

---

## 13. 其他同类对照

- **gpt4all（Nomic）**：开源桌面 app（安装包）+ Python 库；引擎 = llama.cpp backend + Nomic 自研 C backend；模型在 app 内模型目录下载（GGUF）。来源：https://github.com/nomic-ai/gpt4all ；https://docs.gpt4all.io/index.html
- **fastllm（ztxz16）**：纯 C++ 实现、**"后端无依赖"**的高性能推理引擎（不依赖 PyTorch），支持 CUDA/ROCm/CPU/NUMA/磁盘混合、多卡、OpenAI API server；Android 可直接编译 —— 面向"本地运行和服务部署"的引擎库+server 形态（直接可执行）。来源：https://github.com/ztxz16/fastllm
- **TensorRT-LLM（NVIDIA）**：**对照案例（非 binary 产品）**：以 NGC Docker 容器 + Python wheel 发布，依赖完整 TensorRT+CUDA 环境；不追求独立可执行。来源：https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation/installation-guide.md ；https://catalog.ngc.nvidia.com/orgs/nvidia/tensorrt-llm/containers/devel/-
- **LLM-rs / mistral.rs、vLLM（Python 系）**：vLLM 以 pip/uv wheel 发布（Python 包 + 编译扩展），非独立 binary —— 说明 Python 生态做服务端产品时"wheel 分发"是主流，binary 化反而是差异化（未逐一深入调研）。

---

## 14. 跨项目规律提炼

### 14.1 GPU/CUDA 依赖的五种处理策略
1. **捆绑 CUDA 运行时库，依赖系统内核驱动**（llama.cpp 的 cudart 包、ollama 主包/rocm 包、koboldcpp Windows exe）。Linux 上捆绑 .so、Windows 上捆绑 DLL 是常态；系统仍需 NVIDIA 驱动。
2. **按 GPU 类型拆包、安装时探测并额外下载**（ollama：lspci 探测 → rocm/jetpack 附加包）。
3. **运行时 JIT 编译 GPU 源码，依赖系统编译器**（llamafile 独有，代价是要求 nvcc/Xcode）。
4. **依赖系统预装 CUDA/cuDNN**（onnxruntime-genai GPU EP；Jan 安装后端后检查系统库并提示）。
5. **编译期 AOT 生成 kernel，运行时只依赖平台 API**（MLC-LLM/TVM、ExecuTorch 的 backend 编译进产物）。

### 14.2 模型分发
- 绝对主流：**旁路分发** —— 内置下载器（ollama pull / cortex pull / LocalAI models install / LM Studio·Jan·gpt4all 内置模型库）或用户自备文件（llama.cpp）。
- 例外：llamafile 把权重拼进文件（因为卖点就是"1 文件即模型"）。

### 14.3 引擎与产品关系
- 服务端产品清一色**薄壳 + 内嵌上游引擎**：ollama(Go) + llama.cpp fork；LocalAI(Go) + gRPC backends；cortex(C++) + llama.cpp/onnxruntime；Jan(Electron) + llama.cpp。自研引擎（MLC/ExecuTorch/ORT/MNN/ncnn/fastllm）大多是"引擎+SDK"而非终端产品。
- llama.cpp 事实上成为"推理内核标准件"，被所有主流本地推理产品内嵌。

### 14.4 体积与发布
- 捆绑 CUDA 后 Linux 服务端包普遍 **1GB+**（ollama amd64 1.43GB、rocm 1.05GB）；不含 GPU 库的 macOS 包仅 158MB（Metal 系统自带）—— GPU 库是体积大头。
- 发布渠道以 GitHub Releases（CI 自动构建）为主；桌面产品走官网安装包；服务端 binary 也常配 docker 镜像（LocalAI/TensorRT-LLM）。

---

## 15. 未确认项清单

- llama.cpp 各平台包的精确体积（release 页未显示体积；可通过 GitHub API 资产 size 字段确认）。
- LM Studio 是否在安装包内捆绑 CUDA 运行时库、是否要求系统预装 CUDA toolkit（闭源，官方未公开细节）。
- LocalAI llama.cpp backend 是否捆绑 CUDA so 还是依赖系统 CUDA（官方文档未细述）。
- cortex.cpp 现行版本（menloresearch）GPU 运行时库的捆绑方式（商业产品，文档未公开）。
- MLC-LLM 桌面/服务端 CUDA 后端的运行时依赖细节（文档侧重 iOS/Android）。
- ollama 新自研 runner（非 llama.cpp）在 v0.34 中的实际启用范围（release notes 提及 MLX runner / llama.cpp updates 并存，未逐一核实版本）。
- llamafile 本体（不含权重）的精确体积。
- 各桌面 app（LM Studio/Jan/gpt4all）完整安装包体积。
