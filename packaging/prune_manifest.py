"""Prune manifest for the slim single-node release (layer 0/2, zero sglang diff).

Layer 0: whole top-level dirs that never enter the product (handled by
build_slim.py's --from-repo mode; listed here for documentation and stats).

Layer 2: files/dirs excluded from the python package copy. Safety contract
for every entry: it is either (a) imported only lazily inside functions, or
(b) never imported on the single-node code path. build_slim.py verifies this
contract with an AST scan (no module-level import may reference an excluded
module); a violation fails the build rather than shipping a broken tree.

Static-imported candidates (anthropic/ollama/search entrypoints, mlx,
elastic_ep, kv_canary, speculative, lora, multimodal, ...) are deliberately
NOT listed here: they need patches first (layer 3, see
research/gpu-inference-binary-release/pruning-plan.md).
"""

# ---------------------------------------------------------------------------
# Layer 0: top-level repository dirs excluded from the product entirely.
# ---------------------------------------------------------------------------
TOPLEVEL_EXCLUDES = [
    "test",
    "docs",
    "benchmark",
    "scripts",
    "examples",
    "tools",
    "assets",
    "docker",
    "3rdparty",
    "sgl-model-gateway",
    "experimental",
]

# ---------------------------------------------------------------------------
# Layer 2: package-internal excludes, glob patterns relative to python/sglang/.
# Every pattern must be justified by the comment above it.
# ---------------------------------------------------------------------------

PACKAGE_EXCLUDES = [
    # --- Cross-node KV transfer backends (lazy import via get_kv_class;
    #     disaggregation_mode=null never triggers them). ---
    "srt/disaggregation/mooncake",
    "srt/disaggregation/nixl",
    "srt/disaggregation/mori",
    "srt/disaggregation/ascend",
    "srt/disaggregation/fake",
    # --- Remote KV storage backends (register_backend registers a lazy
    #     loader only; import happens on create_backend, never for
    #     file/mmap/shm). Verified: backend_factory.py:44-63. ---
    "srt/mem_cache/storage/nixl",
    "srt/mem_cache/storage/mooncake_store",
    "srt/mem_cache/storage/hf3fs",
    "srt/mem_cache/storage/umbp",
    "srt/mem_cache/storage/flexkv",
    "srt/mem_cache/storage/lmcache",
    "srt/mem_cache/storage/aibrix_kvcache",
    "srt/mem_cache/storage/eic",
    "srt/mem_cache/storage/simm",
    "srt/mem_cache/storage/npu_memcache",
    # --- Ray cluster mode (self-contained; only launch_server.py's
    #     use_ray branch imports it lazily). ---
    "srt/ray",
    # NOTE: hardware_backend/{npu,xpu,musa} look removable but have BARE
    # module-level imports from layers/ (e.g. cuda_graph_setup.py:16-17,
    # moe_runner/ascend.py, compressed_tensors schemes). Verified by the
    # AST checker on 2026-09-16; moved to layer 3 (needs patches).
    # --- Cross-node / other-vendor device communicators: NONE are safely
    #     removable at layer 2. mooncake_transfer_engine is imported at
    #     module level by model_runner.py:37; gated_launch by bootstrap.py:23;
    #     naive_distributed by utils/host_shared_memory.py:10 (all found by
    #     the AST checker, contradicting earlier survey notes). These move
    #     to layer 3. ---
    # --- Standalone tools / optional entrypoints (lazy or unused on the
    #     default path). grpc_server.py is the legacy --smg-grpc-mode route;
    #     the native --grpc-port rust route (grpc_bridge.py) is kept.
    #     checkpoint_engine is only imported inside a function in
    #     weight_updater.py. gated_launch/naive_distributed are KEPT
    #     (module-level imports from bootstrap.py / host_shared_memory.py). ---
    "srt/checkpoint_engine",
    "srt/entrypoints/grpc_server.py",
    # --- Debug tooling (comparator/simulator are standalone CLIs). ---
    "srt/debug_utils/comparator",
    "srt/debug_utils/schedule_simulator",
    # NOTE: sglang/lang (legacy DSL frontend) is KEPT: test/* imports it at
    # module level. 4.6K lines, harmless.
    # NOTE: sglang/test and sglang/benchmark are both KEPT: test contains
    # the scripted_runtime runtime component (imported at module level by
    # scheduler.py / request_receiver.py / ipc_channels.py / utils/common.py),
    # and test/* imports benchmark at module level. Found by the AST
    # checker; fine-grained trimming deferred.
    # NOTE: both dots3_common and inkling_common are KEPT: the former is
    # imported by multimodal/processors/dots_note_omni.py:21, the latter by
    # lora/trtllm_lora_temp/inkling_dense.py:8 (both found by the AST
    # checker). Model-family commons stay with the package.
]

# ---------------------------------------------------------------------------
# Model whitelist: keep llama/qwen/deepseek families + shared infrastructure.
# registry.py discovers models via pkgutil scanning, so missing files are
# simply not registered (SGLang upstream design, zero patch).
# ---------------------------------------------------------------------------
MODELS_KEEP_PREFIXES = (
    "llama", "qwen", "deepseek", "granite",
    "mixtral",  # granitemoe.py imports mixtral.py (dependency closure)
    "dbrx",     # deepseek_v4.py imports dbrx.py (dependency closure, found in e2e)
)
MODELS_KEEP_SHARED = {"__init__.py", "registry.py", "utils.py", "transformers.py"}
MODELS_KEEP_DIRS = {"deepseek_common"}

# JIT kernel sources under sglang/kernels are kept wholesale for now:
# trimming per-model kernel dirs is deferred until the target model set is
# frozen (risk of missing a kernel the warmup path needs).
