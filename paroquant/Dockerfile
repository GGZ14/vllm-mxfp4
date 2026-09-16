# syntax=docker/dockerfile:1.7
#
# Dockerfile (paroquant/) -- the ParoQuant prod-bake image.
#
# Same contract as the top recipe in Dockerfile.ggz14.top, differing only in
# the profile baked: the ggz14 image bakes the MXFP4 measured profile; this
# image bakes the ParoQuant measured profile (paroquant/run_paroquant.sh's
# "-e" list at launcher defaults), so a bare `docker run` reproduces the
# ParoQuant prod unit without an exporter.
#
# Everything else is shared, on purpose: both images ride the same published
# base (build.sh --paro pulls it and passes the live digest), bake the same
# common bake (patched libr4d, the 22-patch chain, the two kernels, the aiter JIT module .so,
# the
# dispatch registration, entrypoint + preamble) and carry the same universal
# quant plugin set -- dispatch stays purely by the checkpoint's
# quantization_config, no envvar "personality" switch. A quark_mxfp4
# checkpoint run on this image still lands on the MXFP4 kernel (under the
# PQ profile); a Querulf.paroquant checkpoint run on the ggz14 image still
# lands on the ParoQuant kernel (under the MXFP4 profile).
#
# The libr4d here is the b9e42ab + r4d_radiance_extras.patch build (the
# rx6 flavour the launcher docs call the "fresh box" default; the rx7 build
# only matters for the optional GDN_SSM_DTYPE narrow-state serves, which
# still go through the launcher's in-container build).
#
# Setup / serving are unchanged: IMAGE=ggz14/vllm-radiance-paroquant:latest
# in setup.sh / serve.sh routes the same flow to this image (serve.sh's
# per-QUANT profile exports are idempotent against the baked values), and
# build.sh --paro tags it $VERSION-$SHA + latest + v$VERSION beside the
# ggz14 image.

# NB: same base caveat as the top recipe -- the published base carries vLLM
# 0.27.1; the patch set carries both shapes.
ARG BASE_REPO=stilldeadcode/vllm-radiance
ARG BASE_TAG=0.9.3
# Same default pin as the top recipe. build.sh pulls the base and passes its
# live digest via --build-arg BASE_DIGEST=<hex> so a base repaint can't slip
# in; change BASE_TAG and BASE_DIGEST together if you move the base.
ARG BASE_DIGEST=45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25
ARG R4D_REPO=https://codeberg.org/StillDeadcode/libr4d.git
ARG R4D_PIN=b9e42ab
ARG GFX_ARCH=gfx1201

# ------------------------------------------------------------- stage 1: libr4d
# The upstream libr4d repo at the pinned commit, with this repo's extras patch
# applied, compiled by its own build.sh. Only the final .so is inherited.
FROM ${BASE_REPO}:${BASE_TAG}@sha256:${BASE_DIGEST} AS r4d_builder
ARG R4D_REPO
ARG R4D_PIN
COPY r4d_radiance_extras.patch /tmp/r4d_radiance_extras.patch
RUN set -e; \
  # The published base is a pruned shipping image -- no git. Get it if absent.
  command -v git >/dev/null 2>&1 || (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git); \
  git clone "$R4D_REPO" /src/libr4d; \
  git -C /src/libr4d checkout -q "$R4D_PIN"; \
  git -C /src/libr4d apply /tmp/r4d_radiance_extras.patch; \
  cd /src/libr4d; \
  GFX_ARCH="$GFX_ARCH" ./build.sh
# /src/libr4d/r4d.so is the deliverable of this stage.

# ------------------------------------------------------------- stage 2: main
FROM ${BASE_REPO}:${BASE_TAG}@sha256:${BASE_DIGEST}
ARG GFX_ARCH

# Build-time sources. Everything lands in /build (outside the venv, and never
# mounted at runtime) so a stale copy cannot shadow site-packages on sys.path
# -- the documented 17-hours-stale-kernel incident, where a `make`-left .so in
# the working directory silently served a kernel older than its own source.
COPY r4d_radiance_extras.patch        /build/r4d_radiance_extras.patch
COPY patch_*.py _patchlib.py install_radiance_hooks.py /build/
COPY radiance_mxfp4_fp8.hip           /build/
COPY radiance_preamble.py             /build/radiance_preamble.py
COPY radiance_entrypoint.sh           /build/radiance_entrypoint.sh
COPY mxfp4-configs/                  /build/mxfp4-configs/
COPY radiance_nvfp4.py \
     radiance_gemm.py \
     radiance_mxfp4.py \
     radiance_rmsquant.py \
     radiance_gdn.py \
     radiance_gdnmerge.py \
     radiance_drafthead.py \
     radiance_verifyhead.py \
     radiance_aroverlap.py \
     radiance_topk.py \
     radiance_arnq.py \
     radiance_tp3pad.py \
     radiance_dflash_capture.py       /build/
COPY paroquant/radiance_paroquant.hip     /build/
COPY paroquant/radiance_paroquant.py      /build/
COPY paroquant/radiance_paroquant_mxfp4.py /build/
COPY paroquant/par_kernels.h              /build/

# The patched libr4d.so from the builder stage.
COPY --from=r4d_builder /src/libr4d/r4d.so \
     /opt/vllm/lib/python3.12/site-packages/r4d.so

ENV SP=/opt/vllm/lib/python3.12/site-packages

# ------------------------------------------------------- the patch chain --------
# The union of the chains `serve-mxfp4.sh` (MXFP4) and `paroquant/run_paroquant.sh`
# (ParoQuant) run at container start, in launcher order, `patch_nvfp4_mxfp4` after
# `patch_quark_mxfp4` (the new NVFP4-at-load route, inert unless RADIANCE_NVFP4_MXFP4=1
# AND a NVFP4 checkpoint is loaded). Keep patch_qwen3_thinkoff.py non-fatal, exactly
# as the launchers do.
RUN set -e; cd /build; \
  python3 patch_quark_mxfp4.py; \
  python3 patch_nvfp4_mxfp4.py; \
  python3 patch_tp3_pad.py; \
  python3 patch_ar_maxbytes.py; \
  python3 patch_topk_triton_rows.py; \
  python3 patch_dflash_calib.py; \
  python3 patch_dflash_mxfp4_kv.py; \
  python3 patch_rmsquant_fusion.py; \
  python3 patch_verify_head.py; \
  python3 patch_kv_group_size.py; \
  python3 patch_topk_composite.py; \
  python3 patch_gdn_shared_build.py; \
  python3 patch_dflash_selector_topk.py; \
  python3 patch_gdn_merge_inproj.py; \
  python3 patch_dynwidth.py; \
  python3 patch_async_dynwidth.py; \
  python3 patch_step_trace.py; \
  python3 patch_ar_geometry.py; \
  python3 patch_ar_3rank.py; \
  python3 patch_gdn_glue.py; \
  python3 patch_skinny_gemm.py; \
  (python3 patch_qwen3_thinkoff.py \
   || echo "[radiance] WARNING: thinkoff patch did not apply")

# ------------------------------------------------------- ship the modules --------
# The same copy set the launchers do (mxfp4-configs into the triton gemm config
# dir, the radiance_*.py modules into site-packages, the dflash / paroquant /
# nvfp4 modules alongside).
RUN set -e; \
  cp -r /build/mxfp4-configs/* "$SP"/aiter/ops/triton/configs/gemm/; \
  cp /build/radiance_nvfp4.py \
     /build/radiance_mxfp4.py \
     /build/radiance_gemm.py \
     /build/radiance_rmsquant.py \
     /build/radiance_gdn.py \
     /build/radiance_gdnmerge.py \
     /build/radiance_drafthead.py \
     /build/radiance_verifyhead.py \
     /build/radiance_aroverlap.py \
     /build/radiance_topk.py \
     /build/radiance_arnq.py \
     /build/radiance_tp3pad.py \
     /build/radiance_dflash_capture.py \
     /build/radiance_paroquant.py \
     /build/radiance_paroquant_mxfp4.py \
     "$SP"/

# ------------------------------------------------------- compile the kernels -----
# Same flags as the launchers (`-O3 -w -std=c++17 -fPIC -shared --offload-arch`).
# The launcher's in-container ParoQuant build passes $RADIANCE_PQ_HIPCC_FLAGS, which
# is EMPTY in prod -- the header default PQ_HW_CVT=1 (device e4m3 convert) is the
# prod kernel, so an unflagged bake below IS the prod build. Build ONCE here instead
# of at every container start.
RUN set -e; cd /build; \
  hipcc -O3 -w -std=c++17 -fPIC -shared \
        --offload-arch=$GFX_ARCH \
        $(python3 -m pybind11 --includes) \
        radiance_mxfp4_fp8.hip -o "$SP"/radiance_mxfp4_fp8.so; \
  hipcc -O3 -w -std=c++17 -fPIC -shared \
        --offload-arch=$GFX_ARCH \
        $(python3 -m pybind11 --includes) \
        radiance_paroquant.hip -o "$SP"/radiance_paroquant_kernel.so

# ------------------------------------------------------- register the dispatch --
# Same guarded append the launchers write, to the STDLIB sitecustomize (a
# site-packages copy is silently shadowed on Ubuntu).
RUN set -e; \
  if ! grep -q radiance_paroquant /usr/lib/python3.12/sitecustomize.py; then \
    printf '%s\n' \
      "try:" \
      "    import radiance_paroquant  # registers the paroquant quantization config" \
      "    import radiance_paroquant_mxfp4  # and the MXFP4-weights variant (paroquant_mxfp4)" \
      "    import radiance_dflash_capture  # drafter training-data capture (inert unless RADIANCE_DFLASH_CAPTURE_DIR)" \
      "except Exception as e:" \
      "    import sys" \
      "    sys.stderr.write(\"[radiance.paroquant] registration failed: %r\\n\" % (e,))" \
      >> /usr/lib/python3.12/sitecustomize.py; \
  fi

# ------------------------------------------------------- entrypoint + preamble --
# The 0.9.3 image's entrypoint script loses its +x bit (the tracker keeps it
# 100644). Replace it with this repo's clean preamble + banner and chmod it in
# at build time so the runtime can never hit the "not executable" class again.
RUN install -m 755 /build/radiance_entrypoint.sh /opt/radiance_entrypoint.sh \
 && install -m 644 /build/radiance_preamble.py   /opt/radiance_preamble.py
RUN chmod +x /opt/radiance_entrypoint.sh

# --------------------------------------------------------- the measured profile --
# The in-image kernel modules default to STOCK when a RADIANCE_* env is unset,
# and the ParoQuant launcher's whole measured configuration arrives as `-e`
# flags. Bake paroquant/run_paroquant.sh's `-e` list at launcher defaults in
# so a bare `docker run` reproduces the ParoQuant prod unit:
#
#   int4-prod defaults (the unit's): RADIANCE_PAROQUANT=1, the rotation_supply
#   producers on (PG_PRODUCER=3, FUSED_TOKQ, ROT_STREAM/2=1, ROT_V2=1),
#   I8/PG/ZPE=0 (read off the checkpoint by the launcher and by serve.sh),
#   GDN_MERGE_INPROJ=0 (in_proj_a/b are fp16 in the PQ checkpoints; the mxfp4
#   image has it on -- the one profile divergence between the two images)
#   and GDN_FUSED_UPDATE=1 / the dynwidth stack / the drafter+verify profile,
#   all straight off the launcher.
#
# Skipped on purpose (empty / runtime-only at the launcher): RADIANCE_PQ_CHECKALL
# and its PQM sibling (empty in prod), RADIANCE_PQ_HIPCC_FLAGS (the in-container
# A/B build passthrough -- the kernel here is baked), the visible-GPU ids and
# the R4D_SO mount (this image BAKES the libr4d build instead).
#
# Cache dirs default to /cache (the launcher mounts $CACHE there) so a bare run
# with that mount behaves identically; without the mount the dirs land in the
# container layer and are dropped with --rm.
COPY qwen-fixed-v22.3.jinja /opt/qwen-fixed-v22.3.jinja
ENV HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_ROCM_USE_AITER=1 \
    VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
    VLLM_ROCM_USE_AITER_MHA=0 \
    VLLM_ROCM_USE_AITER_MLA=0 \
    VLLM_ROCM_USE_AITER_MOE=0 \
    VLLM_ROCM_USE_AITER_LINEAR=0 \
    VLLM_ROCM_USE_AITER_FP8BMM=0 \
    VLLM_ROCM_USE_AITER_FP4BMM=0 \
    VLLM_ROCM_USE_AITER_RMSNORM=0 \
    NCCL_PROTO=Simple \
    VLLM_CACHE_ROOT=/cache/vllm \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TRITON_CACHE_DIR=/cache/triton \
    AITER_ROOT_DIR=/cache/aiter \
    TRITON_CACHE_AUTOTUNING=1 \
    RADIANCE_USE_R4D=1 \
    RADIANCE_USE_R4D_AR=1 \
    RADIANCE_USE_R4D_AR_QUANT=1 \
    RADIANCE_R4D_REPORT=1 \
    R4D_ATTN_FP8=3 \
    RADIANCE_AR_MAX_KB=86016 \
    RADIANCE_PRESHUFFLE=1 \
    RADIANCE_FUSE_RMS_QUANT=1 \
    RADIANCE_GDN_FUSED_UPDATE=1 \
    RADIANCE_GDN_MERGE_INPROJ=0 \
    RADIANCE_GDN_FUSED_MAX_ITEMS=32 \
    RADIANCE_GDN_PATHS=both \
    RADIANCE_DYNAMIC_WIDTH=1 \
    RADIANCE_DYNW_ALPHA=0.35 \
    RADIANCE_DYNW_MARGIN=2 \
    RADIANCE_DYNW_MIN=2 \
    RADIANCE_DYNW_MIN_BATCH=3 \
    RADIANCE_AR_QNB=96 \
    RADIANCE_AR_QNT=1024 \
    RADIANCE_AR_OVERLAP=0 \
    RADIANCE_STEP_TRACE=0 \
    RADIANCE_SKINNY_GEMM=1 \
    RADIANCE_TOPK_TRITON_MIN_ROWS=1 \
    RADIANCE_KV_GROUP_OPT=1 \
    RADIANCE_FAST_DRAFT=1 \
    RADIANCE_DRAFT_TAU=0.20 \
    RADIANCE_DRAFT_RERANK=80 \
    RADIANCE_VERIFY_HEAD=1 \
    RADIANCE_VERIFY_HEAD_MAX_M=32 \
    RADIANCE_PAROQUANT=1 \
    RADIANCE_PQ_WPERM=1 \
    RADIANCE_PQ_DECODE_NT=1 \
    RADIANCE_PQ_ATILED=1 \
    RADIANCE_PQ_AT_LBK=128 \
    RADIANCE_PQ_AT_HOIST=1 \
    RADIANCE_PQ_PTOK=1 \
    RADIANCE_PQ_FUSED_TOKQ=1 \
    RADIANCE_PQ_PG_PRODUCER=3 \
    RADIANCE_PQ_ROT_STREAM=1 \
    RADIANCE_PQ_ROT_STREAM2=1 \
    RADIANCE_PQ_ROT_STREAM3=0 \
    RADIANCE_PQ_ROT_V2=1 \
    RADIANCE_PQ_AR_CHECK=0 \
    RADIANCE_PQ_AR_FALLBACK=0 \
    RADIANCE_PQ_I8=0 \
    RADIANCE_PQ_PG=0 \
    RADIANCE_PQ_ZPE=0 \
    RADIANCE_PQ_DECODE_MAX_M=64 \
    RADIANCE_PQ_CHECK_MAX_M=128 \
    RADIANCE_MXFP4_W4A8=1 \
    RADIANCE_MXFP4_WPERM=1 \
    RADIANCE_MXFP4_DECODE_NT=1 \
    RADIANCE_MXFP4_DECODE_MAX_M=64 \
    RADIANCE_MXFP4_EPIFAST=1 \
    RADIANCE_MXFP4_TN4_MIN_M=2048 \
    RADIANCE_MXFP4_A_TILED_MIN_M=513 \
    RADIANCE_NVFP4_MXFP4=1 \
    RADIANCE_NVFP4_EXP=mse \
    RADIANCE_NVFP4_FP8_LAYERS=mxfp4 \
    RADIANCE_NVFP4_BF16_LAYERS=in_proj_ba \
    RADIANCE_NVFP4_LMHEAD=bf16
# --------------------------------------------------------- aiter JIT pre-bake ---
# aiter builds its JIT .so lazily on first use, and two independent facts make
# the per-boot build unavoidable today, including on a persistent cache:
#   (a) the built .so lands in the writable container layer, which --rm drops
#       (the seen symptom: "start build [module_aiter_core]" on every boot);
#   (b) aiter's own get_module() staleness check probes
#       $SP/aiter/jit/module_aiter_core.so (get_user_jit_dir()), while
#       build_module() for this entry (not a python module, not standalone in
#       optCompilerConfig.json) copies the artifact to $SP/op_tests/ ... and
#       so nothing ever lands at the probed path: aiter re-compiles on every
#       boot even where a cache dir would be writable.
# The fix drives the same build at IMAGE build time (hipcc needs no GPU to
# compile, same as the mxfp4 / ParoQuant kernels above) and places the .so at
# the probed path. The runtime then sees a .so whose embedded arch markers
# match the device (--offload-arch strings in the binary; _needs_arch_rebuild
# is false for a matching device and for no device), adopts it, and skips the
# build. If it is ever missing or arch-mismatched, aiter rebuilds in place --
# same self-healing as today, so this cannot pin a stale binary.
#
# Fidelity: this is aiter's own core.build_module("module_aiter_core") with the
# exact per-module args its first-use runtime pass uses (get_args_of_build /
# optCompilerConfig.json: single TU aiter_core_pybind.cu, -DENABLE_CK=0,
# torch-excluded). Two env deltas, both equal to what the runtime resolves
# live on the serving card:
#   GPU_ARCHS=gfx1201  aiter's live-query path (no device -> ROCm info) lands
#                      here; the build has no GPU, so state it explicitly.
#   CU_NUM=64          64 CUs on the R9700; aiter needs it stated for a
#                      GPU-less build (no map entry) and resolves the same
#                      value live at runtime.
# MAX_JOBS stays unset (aiter self-limits by cores / free memory, psutil) and
# the assertion below pins the config identity so a future aiter source change
# forces a loud rebuild next time this block re-runs (any base bump runs it).
# Cost: tens of minutes added to a one-time image build; every boot after
# paid it before (and still pays for any of the ~113 other aiter modules this
# workload does not use -- those stay on the first-use lazy path, correct).
# Script (not heredoc): the classic builder (DOCKER_BUILDKIT=0, pinned in build.sh)
# discards a no-backslash in-file heredoc and runs python - clean on EOF (the first
# cut of this block failed exactly that way -- every test passed, no .so anywhere),
# so the bake runs from the repo's standalone bake_aiter_core.py, plain-COPY +
# plain-RUN safe under both classic and buildkit: the script's asserts / exceptions
# decide the step outcome -- a layout change or build failure is a loud failure,
# never a silently skipped bake.
COPY bake_aiter_core.py /opt/bake_aiter_core.py
RUN set -e; /opt/vllm/bin/python3 /opt/bake_aiter_core.py

# ------------------------------------------------------- build-time verification -
# Prove the whole kernel stack is import-clean BEFORE the image is declared
# built, so a broken kernel fails the build rather than the first container
# start. Use the venv's python (the system python3 does not see $SP). Includes
# the four .so files the runtime needs (the three baked kernels + the pre-baked aiter JIT module).
RUN set -e; \
  test -f "$SP"/r4d.so; \
  test -f "$SP"/radiance_mxfp4_fp8.so; \
  test -f "$SP"/radiance_paroquant_kernel.so; \
  test -f "$SP"/aiter/jit/module_aiter_core.so; \
  /opt/vllm/bin/python3 -c 'import radiance_nvfp4, \
      radiance_mxfp4, radiance_gemm, radiance_rmsquant, \
      radiance_gdn, radiance_gdnmerge, radiance_drafthead, radiance_verifyhead, \
      radiance_aroverlap, radiance_topk, radiance_arnq, radiance_tp3pad, \
      radiance_dflash_capture, radiance_paroquant, radiance_paroquant_mxfp4; \
      print("[parobake] all radiance modules import-clean in", "'"$SP"'")'

# ------------------------------------------------------- drop build sources -----
# Nothing at runtime needs the patches / source .py / .hip; remove them so a
# future stale copy cannot shadow the venv (the documented incident above).
RUN rm -rf /build /tmp/r4d_radiance_extras.patch

ENTRYPOINT ["/opt/radiance_entrypoint.sh"]
# No CMD: the consumer supplies the vllm-serve args (tamad `command:`, or
# `vllm serve ...`), which the entrypoint passes through verbatim.
