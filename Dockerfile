# vllm-radiance: vLLM/torch/triton/aiter stack for RDNA4 (gfx1201 / R9700), plus the radiance
# patches and kernels. Single multistage build on the official AMD ROCm image, in four stages:
#   1. builder    compile torch/triton/torchvision/aiter/vLLM from source into /wheels
#   2. rocmprune  cut the 19 GB ROCm tree down to this one GPU architecture
#   3. assemble   install the wheels, apply the patches, build the R4D kernel library
#   4. final      the release image: a clean Ubuntu with only the pruned ROCm and the venv
# No prebuilt component wheels and no checked-in binaries. The release image carries neither the
# build toolchain nor the wheels, which is most of the reason it is far smaller than the base.
#
# stack: torch 2.11.0, triton 3.6.0, torchvision 0.24.1, aiter v0.1.17, vLLM v0.27.1,
# all compiled for PYTORCH_ROCM_ARCH=gfx1201 against the base image's ROCm 7.14.
ARG ROCM_BASE=rocm/dev-ubuntu-24.04:7.14.0-full@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7
ARG GFX_ARCH=gfx1201
# The release stage starts from a clean distro image rather than the ROCm base, and COPYs in only
# the pruned ROCm tree plus the venv. Same Ubuntu release as the ROCm base (24.04), so the venv's
# interpreter (python 3.12.3) matches.
ARG RELEASE_BASE=ubuntu:24.04@sha256:a08e551cb33850e4740772b38217fc1796a66da2506d312abe51acda354ff061

# Component pins, in one place. Each is both the git tag that gets compiled and the version the
# resulting wheel reports, so `pip show`, `importlib.metadata`, and the startup banner all agree
# with what was actually built.
# torch/triton/torchvision are NOT free choices, and the number to read is not the one in
# pyproject.toml. vLLM 0.27.1's build-system asks for `torch == 2.13.0`, but that is the CUDA
# build: upstream's own ROCm image (docker/Dockerfile.rocm_base) builds PYTORCH_BRANCH=release/2.11
# with torchvision v0.24.1, and requirements/rocm.txt pins no torch at all. release/2.11 is
# therefore the combination upstream actually tests on ROCm, unchanged from 0.26.0. torch 2.11.0
# pins triton 3.6.0. Building against newer ones means running a combination upstream never tests:
# 0.5.0-0.5.4 did exactly that (torch 2.13 / triton 3.7.1 / torchvision 0.28) because
# `use_existing_torch.py` strips the pin, and those builds hang the GPU under load where 0.4.0 --
# which used this sanctioned trio -- does not.
ARG TORCH_VERSION=2.11.0
ARG TRITON_VERSION=3.6.0
ARG TORCHVISION_VERSION=0.24.1
ARG AITER_VERSION=0.1.17
ARG VLLM_VERSION=0.27.1
# transformers is pinned here because vLLM does not pin it: requirements/common.txt asks only for
# `transformers >= 5.5.3`, so an unpinned rebuild silently picks up whatever is newest and the
# stack changes underneath the build. 5.15.0 made Gemma-4's head_dim a per-layer attribute and
# turned the global read into AmbiguousGlobalPerLayerAttributeError, which no released vLLM config
# convertor handles -- a Gemma-4 checkpoint then fails during argument parsing, before a model or
# an attention backend exists. 5.14.1 is the last release before that change and loads every
# architecture this image serves.
ARG TRANSFORMERS_VERSION=5.14.1
# rocm-bandwidth-test for the startup topology/bandwidth sweep. Pinned to the NEWEST tag that still
# has a plain CMakeLists: the rocm-7.x tags moved to a cmake framework that demands clang>=19 on PATH
# plus vendored boost/fmt/curl submodules, none of which this tool needs.
ARG RBT_VERSION=rocm-6.4.4
# R4D: the HIP kernel library for this GPU -- attention, gated delta net, all-reduce and a
# skinny bf16 GEMM. It is a library of gfx1201 kernels rather than a part of this image, so it
# lives in its own repository and is pinned here like any other component; R4D_REPO exists so a
# fork or a local mirror can be substituted without editing the build. The tag is asserted against
# the version the built library reports, so a stale clone fails the build instead of shipping.
ARG R4D_REPO=https://codeberg.org/StillDeadcode/libr4d.git
ARG R4D_VERSION=v0.5.0

# =====================================================================================
# STAGE 1 builder: compile the stack from source into /wheels
# =====================================================================================
FROM ${ROCM_BASE} AS builder
ARG GFX_ARCH
ARG TORCH_VERSION
ARG TRITON_VERSION
ARG TORCHVISION_VERSION
ARG AITER_VERSION
# Build parallelism, as an ARG so a build can be told to leave the box some headroom:
#   docker build --build-arg MAX_JOBS=16 .
ARG MAX_JOBS=32
ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ROCM_ARCH=${GFX_ARCH} \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
    USE_ROCM=1 USE_CUDA=0 MAX_JOBS=${MAX_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

# Build tooling the base dev image lacks (git/venv/pkg-config + the -dev packages torch's cmake
# probes: libdrm for rocm_smi, libnuma, libelf).
RUN apt-get update && apt-get install -y --no-install-recommends \
      git python3.12-venv build-essential ccache pkg-config \
      libdrm-dev libnuma-dev libelf-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/py
ENV PATH=/opt/py/bin:$PATH
# setuptools-scm is a build requirement of aiter and vLLM (both take their version from the git
# tag). --no-build-isolation means it is NOT auto-installed: without it here setuptools silently
# ignores `use_scm_version` and the wheel is stamped 0.0.0.
RUN pip install -U pip wheel setuptools "setuptools-scm>=8.0" "cmake<4" ninja pybind11 numpy \
      pyyaml typing_extensions cffi requests
RUN mkdir -p /wheels

# --- torch (AOTriton off: its gfx1201 source-configure fails and vLLM never uses torch
#     SDPA-flash; USE_MAGMA=0: base has no magma) ---
RUN git clone --depth 1 -b v${TORCH_VERSION} --recurse-submodules --shallow-submodules \
        https://github.com/pytorch/pytorch.git /src/pytorch \
    && cd /src/pytorch \
    && pip install -r requirements.txt \
    && python tools/amd_build/build_amd.py \
    && USE_MAGMA=0 USE_MKLDNN=1 BUILD_TEST=0 USE_NCCL=1 USE_RCCL=1 \
       USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_AOTRITON=0 \
       PYTORCH_BUILD_VERSION=${TORCH_VERSION}+rocm7.14 PYTORCH_BUILD_NUMBER=1 \
       python setup.py bdist_wheel \
    && cp dist/*.whl /wheels/ && pip install dist/*.whl && rm -rf /src/pytorch

# --- triton ---
RUN git clone --depth 1 -b v${TRITON_VERSION} https://github.com/triton-lang/triton.git /src/triton \
    && cd /src/triton && pip wheel --no-build-isolation --no-deps . -w /wheels \
    && pip install /wheels/triton-*.whl && rm -rf /src/triton

# --- torchvision ---
# FORCE_CUDA=1 is REQUIRED: torchvision's BUILD_CUDA_SOURCES gates on torch.cuda.is_available(),
# which is false in `docker build` (no GPU) -> it picks CppExtension, where torch's build-time hipify
# double-compiles vision.cpp + vision_hip.cpp -> "multiple definition of vision::cuda_version()".
# FORCE_CUDA=1 forces CUDAExtension (correct hipify source replacement); hipcc needs no GPU to compile.
RUN git clone --depth 1 -b v${TORCHVISION_VERSION} https://github.com/pytorch/vision.git /src/vision \
    && cd /src/vision && FORCE_CUDA=1 USE_ROCM=1 pip wheel --no-build-isolation --no-deps . -w /wheels \
    && rm -rf /src/vision

# --- aiter (gfx1201; kernels JIT at runtime, PREBUILD_KERNELS=0) ---
# PRETEND_VERSION: the checkout is shallow, so setuptools-scm cannot describe the tag and would fall
# back to a placeholder version; pin it to the tag being built.
RUN git clone --recursive --shallow-submodules -b v${AITER_VERSION} https://github.com/ROCm/aiter.git /src/aiter \
    && cd /src/aiter && GPU_ARCHS=${GFX_ARCH} PREBUILD_KERNELS=0 \
       SETUPTOOLS_SCM_PRETEND_VERSION=${AITER_VERSION} \
       pip wheel --no-build-isolation --no-deps . -w /wheels \
    && pip install /wheels/*aiter-*.whl && rm -rf /src/aiter

# --- vLLM, built against the torch above. use_existing_torch strips the torch/torchvision pins so
#     pip does not try to fetch them; the versions built above ARE the pinned ones, so this is now
#     just "use what was compiled above", not an override.
#     setuptools-rust is a pyproject build requirement that --no-build-isolation does not install.
#     VLLM_VERSION_OVERRIDE pins the reported version to the tag: the tree is dirty (use_existing_torch
#     rewrites the requirements files) and shallow, so setuptools-scm would otherwise stamp the wheel
#     with a guessed next-release dev version plus the build date. ---
# ARG at the point of use, not at the top of the stage: an ARG line is a cache-key instruction, so
# declaring it up there would make a vLLM bump rebuild torch, triton, torchvision and aiter too.
ARG VLLM_VERSION
RUN git clone --depth 1 -b v${VLLM_VERSION} https://github.com/vllm-project/vllm.git /src/vllm \
    && cd /src/vllm && python use_existing_torch.py \
    && pip install "setuptools-rust>=1.9.0" \
    && VLLM_TARGET_DEVICE=rocm VLLM_VERSION_OVERRIDE=${VLLM_VERSION} \
       pip wheel --no-build-isolation --no-deps . -w /wheels \
    && rm -rf /src/vllm
RUN ls -la /wheels

# --- rocm-bandwidth-test (the startup topology + bandwidth sweep) ---
# ARG is declared HERE, not in the stage's opening block: an ARG line is a cache-key instruction, so
# putting it up there would invalidate every layer below it -- including the PyTorch compile.
ARG RBT_VERSION
# Plain cmake against the HSA headers/libs the base image already ships; no extra build deps (the
# builder venv's pip cmake and the apt build-essential above cover it). The RUNPATH is REQUIRED:
# ROCm 7.14 keeps libhsa-runtime64.so under the versioned component dir (/opt/rocm/core-<ver>/lib,
# reachable via the `core` alternatives symlink), which is not on the loader's default search path,
# so an un-rpathed binary dies with "libhsa-runtime64.so.1: cannot open shared object file".
RUN git clone --depth 1 -b ${RBT_VERSION} https://github.com/ROCm/rocm_bandwidth_test.git /src/rbt \
    && cmake -S /src/rbt -B /src/rbt/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm \
         -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/opt/rocm/core/lib:/opt/rocm/lib" \
    && cmake --build /src/rbt/build -j 16 \
    && mkdir -p /artifacts && cp /src/rbt/build/rocm-bandwidth-test /artifacts/ \
    && rm -rf /src/rbt

# =====================================================================================
# STAGE 2 rocmprune: cut the ROCm tree down to this image's single GPU architecture
# =====================================================================================
# ~19 GB of the base is device code for GPUs this image cannot run on, plus link-time-only
# archives. Pruning has to happen in a stage that the release stage COPYs FROM: deleting files
# in a layer stacked on the base reclaims nothing, it only writes whiteouts. See prune_rocm.sh
# for what is kept and why (the runtime still has to compile: AITER JITs kernels on first use).
FROM ${ROCM_BASE} AS rocmprune
ARG GFX_ARCH
COPY prune_rocm.sh /tmp/prune_rocm.sh
RUN bash /tmp/prune_rocm.sh ${GFX_ARCH} && rm -f /tmp/prune_rocm.sh

# =====================================================================================
# STAGE 3 assemble: install the wheels and apply the patches and kernels
# =====================================================================================
# Runs on the FULL base because it needs the toolchain (hipcc, headers, static archives) to
# compile the HIP kernels. Only the resulting /opt/vllm venv is carried into the release image.
FROM ${ROCM_BASE} AS assemble
ARG GFX_ARCH
ARG AITER_VERSION
ARG VLLM_VERSION
ARG TRANSFORMERS_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12-venv git \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/vllm
RUN python3 -m venv /opt/vllm
ENV PATH=/opt/vllm/bin:$PATH
ENV SP=/opt/vllm/lib/python3.12/site-packages

# --- install the wheels ---
# torch/triton/vision/aiter with --no-deps so pip does not replace them; vLLM with its pure-python
# dependencies. amdsmi (the ROCm python bindings the base image ships) is required for vLLM's ROCm
# platform detection. The transformers pin goes in the SAME pip invocation as the vLLM wheel so the
# resolver sees it as a constraint -- installing it afterwards would first pull the newest release
# and then downgrade it, leaving both in the layer.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir -U pip wheel setuptools \
 && pip install --no-cache-dir --no-deps \
      /wheels/torch-*.whl /wheels/triton-*.whl /wheels/torchvision-*.whl /wheels/*aiter-*.whl \
 && pip install --no-cache-dir /wheels/vllm-*.whl "transformers==${TRANSFORMERS_VERSION}" \
 && pip install --no-cache-dir /opt/rocm/share/amd_smi pillow pybind11 \
 && rm -rf /wheels /root/.cache

# RADIANCE_GFX_ARCH is what the gfx1201 patch and the banner read for the target arch (amdsmi's
# asic_info reports it empty on this card). It used to be called VLLM_ROCM_GCN_ARCH, which vLLM
# 0.26 flags as an unknown VLLM_* variable at startup; the old name is still honored.
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=${GFX_ARCH} RADIANCE_GFX_ARCH=${GFX_ARCH} \
    HIP_ARCHITECTURES=${GFX_ARCH} AMDGPU_TARGETS=${GFX_ARCH} GPU_ARCHS=${GFX_ARCH} \
    SAFETENSORS_FAST_GPU=1 TOKENIZERS_PARALLELISM=false TRITON_CACHE_AUTOTUNING=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- runtime modules and configs ---
# radiance_amdsmi.py and .pth: amdsmi init-order fix. amdsmi must init before HIP at site-init in
# every process, otherwise it enumerates 0 devices and platform detection fails.
COPY radiance_amdsmi.py radiance_amdsmi.pth \
     radiance_kernels.py radiance_vit_attn.py radiance_allreduce.py \
     radiance_draft.py radiance_draft_gpu.py radiance_drafthead.py radiance_gemm.py \
     radiance_r4d_attn.py radiance_gdn.py radiance_w4.py radiance_mxfp4.py ${SP}/
COPY fp8-configs/ ${SP}/vllm/model_executor/layers/quantization/utils/configs/
COPY moe-configs/ ${SP}/vllm/model_executor/layers/fused_moe/configs/
# mxfp4-configs: aiter ships GEMM-AFP4WFP4 tiles for gfx950/gfx1250 only, and its gfx1250
# bands set matrix_instr_nonkdim=32 for M>=64, which gfx1201 (WMMA 16x16x16 only) cannot
# lower. These pin 16 across every band. Used only when RADIANCE_MXFP4=1.
COPY mxfp4-configs/ ${SP}/aiter/ops/triton/configs/gemm/

# --- gfx1201 fixes and tuned-kernel patches ---
# Each patch edits a vLLM (or aiter/triton) source file in place and checks for source drift before
# writing. patch_gdn_wmma covers the solve_tril triangular block-inverse only; the gated-delta-net
# gram cast is handled upstream since 0.26.0.
# patch_conv1d_blockn widens the gated-delta-net prefill conv1d channel block to a 16-byte-per-lane
# access; bit-identical, and it defuses a 2**14-byte row pitch the caller's split() view creates.
# patch_r4d is the whole libr4d integration in one patch, switchable at run time with
# patch_dflash2 backports DFlash2 speculative decoding (vllm-project/vllm#52816, merged ten days
# after 0.27.1 was tagged): a block-diffusion drafter that proposes a whole block of positions in
# one backbone pass and walks a candidate path through the target head's top-K per position. It
# carries two new upstream modules, installed from /opt/patches/dflash2/. patch_dflash_base
# must run FIRST: DFlash2 subclasses the DFlash speculator, and 0.27.1's copy of that base
# predates three correctness fixes it relies on. Without them the drafter runs, reports a
# healthy acceptance curve, and emits garbled text -- the rejected suffix of the previous
# step is loaded back as accepted context.
# patch_dflash_fused_kv_fp8 lets that drafter be an fp8 checkpoint: the context-KV precompute
# fuses every layer's K/V projection by slicing the raw parameter, which is neither the right
# dtype nor the right row layout once the weights are quantized and preshuffled.
# patch_gdn_metadata cuts the per-step Python cost of building the gated-delta-net attention
# metadata: the per-request bookkeeping runs as one numpy pass over the same buffers, the arange
# and empty index become slices of cached buffers, and the block table is sliced rather than
# gathered when every sequence is a spec decode. Byte-identical output; RADIANCE_GDN_META=0 falls
# back to the stock path.
# patch_dflash_w4 marks the drafter's weight load so radiance_w4 can pack it to 4 bits -- the
# drafter's linears and the target's are indistinguishable inside a quant method's callback, and
# the drafter is loaded by exactly one call, so bracketing that call is the whole discriminator.
# Inert unless RADIANCE_FAST_DRAFT=1, the one switch over the whole tuned drafter stack: the draft
# pass falls 9.1% at a drafter batch of 64, and the acceptance question is settled -- acceptance on
# this stack is bimodal and the mode is drawn per compile, in the control arm too, so a single
# sample per arm reads a coin flip as a tax.
# RADIANCE_USE_R4D: the R4D attention backend enum, plus the gated-delta-net layer, where a whole
# step runs in five hand-written kernels (conv+prep+gating+cumsum, the K-gram with its triangular
# inverse, and the chunked scan on the prefill path; the conv update and the recurrent state
# update on the decode path), and the FLA chunk path still gets the fused scan for any step shape
# the layer hook declines. The two patches above tune that FLA path, which is what runs when
# RADIANCE_USE_R4D=0.
COPY patch_*.py install_radiance_hooks.py _patchlib.py /opt/patches/
COPY dflash2/ /opt/patches/dflash2/
RUN set -eu; cd /opt/patches; \
    for p in patch_gfx1201 patch_radiance_dispatch patch_skinny_gemm patch_unified_attention_lds \
             patch_gdn_wmma patch_preshuffle patch_radiance_fusion install_radiance_hooks \
             patch_unpad patch_mtp_mm_mask patch_mtp_loopbreak patch_qwen3_toolparse patch_from_json_filter \
             patch_dynamo_metrics patch_conv1d_blockn patch_r4d patch_dflash_base patch_dflash2 \
             patch_dflash_fused_kv_fp8 patch_dflash_w4 patch_gdn_metadata \
             patch_quark_mxfp4 patch_ar_maxbytes patch_topk_triton_rows patch_qwen3_thinkoff; do \
      echo "== applying $p =="; python "$p.py"; \
    done; \
    python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('${SP}/radiance_*.py')]; print('radiance modules parse OK')"

# --- R4D: the gfx1201 kernel library, cloned and compiled from source ---
# One shared object holding every hand-written kernel this image runs: paged attention (prefill and
# decode, fp8 or bf16 KV), the fused gated-delta-net prefill scan, the TP=2 P2P all-reduce in both
# its exact and its 6-bit-packed form, and the skinny bf16 GEMM. Built here rather than in the
# builder stage because it has to be compiled by the same hipcc the venv loads it against.
# ARGs are declared at the point of use: they are cache-key instructions, so putting them at the top
# of the stage would invalidate the wheel install above on every kernel bump.
ARG R4D_REPO
ARG R4D_VERSION
RUN git clone --depth 1 -b ${R4D_VERSION} ${R4D_REPO} /src/libr4d \
 && cd /src/libr4d && GFX_ARCH=${GFX_ARCH} OUT=${SP}/r4d.so ./build.sh \
 && WANT=$(echo "${R4D_VERSION}" | sed 's/^v//') \
 && python -c "import sys, torch, r4d; \
assert r4d.__version__ == sys.argv[1], 'r4d reports ' + r4d.__version__ + ', pinned tag is ' + sys.argv[1]; \
print('r4d', r4d.__version__, 'built:'); \
[print('   ', k['family'], k['name']) for k in r4d.kernels()]" "$WANT" \
 && rm -rf /src/libr4d

# --- the MXFP4 W4A8 GEMM, compiled from source ---
# radiance_mxfp4_fp8.hip: hand-written fp8-WMMA GEMM for MXFP4 weights against fp8 activations
# (W4A8). It lives here rather than in libr4d because it is specific to this fork; built in this
# stage for the same reason R4D is -- it must be compiled by the same hipcc the venv loads it
# against. Active only when RADIANCE_MXFP4_W4A8=1; radiance_mxfp4.py soft-disables if it is absent,
# so the import is asserted below rather than left to fail at serve time.
COPY radiance_mxfp4_fp8.hip /opt/patches/
RUN INC=$(python -m pybind11 --includes); \
    hipcc -O3 -std=c++17 -fPIC -shared --offload-arch=${GFX_ARCH} -Wno-unused-result \
      $INC /opt/patches/radiance_mxfp4_fp8.hip -o ${SP}/radiance_mxfp4_fp8.so \
 && python -c "import torch, radiance_mxfp4_fp8 as m; assert hasattr(m, 'launch'); print('radiance_mxfp4_fp8 built')"

# --- strip debug symbols from the installed extensions (worth ~1 GB) ---
# These are release builds, but they still carry .debug_* sections that nothing reads at runtime.
# R4D and the MXFP4 W4A8 GEMM are excluded: both are tiny and carry device fatbins.
RUN find /opt/vllm -type f -name '*.so*' ! -name 'r4d.so' ! -name 'radiance_mxfp4_fp8.so' \
      -exec strip --strip-unneeded {} + 2>/dev/null || true; \
    find /opt/vllm -name '__pycache__' -type d -prune -exec rm -rf {} + || true; \
    echo "extensions stripped"

# =====================================================================================
# STAGE 4 final: the release image -- a clean Ubuntu with only what is needed to serve
# =====================================================================================
# Built by COPYing an allowlist rather than by inheriting the ROCm base, which is what makes the
# prune above pay: the release image never contains the 19 GB tree, the build toolchain, the
# wheels, or the patch sources. Only the pruned ROCm, the venv, and the entrypoint come across.
FROM ${RELEASE_BASE} AS final
ARG GFX_ARCH
ARG AITER_VERSION
ARG VLLM_VERSION
ARG TRANSFORMERS_VERSION
ENV DEBIAN_FRONTEND=noninteractive
# ROCm 7.14 vendors its own libdrm / numa / elf / sqlite / zlib / zstd (the librocm_sysdeps_* set),
# so the release image needs very little from the distro:
#   python3.12     the interpreter the /opt/vllm venv was built against (Ubuntu 24.04 ships 3.12.3)
#   libnuma-dev    rocSHMEM dlopen()s the UNVERSIONED libnuma.so, which only the -dev package ships
#   numactl        optional --numa-bind;  curl  the compose healthcheck runs it inside the container
#   g++            NOT optional: AITER JIT-compiles its kernels on FIRST USE, inside this image, and
#                  hipcc needs the C++ standard headers (and the same g++ major torch was built
#                  with). Without it every JIT build dies with "Could not find standard C++ header
#                  'cmath'", aiter's flag probes all fail (including --offload-arch), and the engine
#                  crashes. The build-time probe below is what keeps this honest.
#   python3.12-dev Python.h, for the pybind11 modules aiter JIT-builds
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3.12 python3.12-dev g++ libnuma-dev numactl curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# /opt/rocm is a symlink farm pointing through /etc/alternatives into core-<ver>, so both have to
# come across or nothing resolves. The pruned tree is stable across radiance releases, which keeps
# it a cached layer users do not re-download for every version bump.
COPY --from=rocmprune /opt/rocm /opt/rocm
COPY --from=rocmprune /etc/alternatives /etc/alternatives
COPY --from=assemble /opt/vllm /opt/vllm
COPY --from=builder /artifacts/rocm-bandwidth-test /usr/local/bin/rocm-bandwidth-test

# RADIANCE_GFX_ARCH is what the gfx1201 patch and the banner read for the target arch (amdsmi's
# asic_info reports it empty on this card). It used to be called VLLM_ROCM_GCN_ARCH, which vLLM
# 0.26 flags as an unknown VLLM_* variable at startup; the old name is still honored.
ENV VIRTUAL_ENV=/opt/vllm \
    PATH=/opt/vllm/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=${GFX_ARCH} RADIANCE_GFX_ARCH=${GFX_ARCH} \
    HIP_ARCHITECTURES=${GFX_ARCH} AMDGPU_TARGETS=${GFX_ARCH} GPU_ARCHS=${GFX_ARCH} \
    SAFETENSORS_FAST_GPU=1 TOKENIZERS_PARALLELISM=false TRITON_CACHE_AUTOTUNING=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- radiance feature flags (set any to 0 to fall back to stock). RADIANCE_USE_R4D is the master
#     switch for the hand-written gfx1201 kernel library: 0 takes it out of the picture entirely
#     (attention, the gated delta net, vision attention, the all-reduce and the skinny GEMM all
#     revert to the stock path) without a rebuild. RADIANCE_USE_R4D_AR and its _QUANT variant are
#     the two all-reduce behaviours worth switching independently, since one is bit-identical to
#     RCCL and the other is not. RADIANCE_RUN_BWTEST runs the bandwidth sweep at startup;
#     it is backgrounded and takes about a second, so it never delays the serve. ---
ENV RADIANCE_USE_R4D=1 \
    RADIANCE_USE_R4D_AR=1 RADIANCE_USE_R4D_AR_QUANT=1 \
    RADIANCE_PRESHUFFLE=1 RADIANCE_FUSE_RMS_QUANT=1 \
    RADIANCE_DYNAMIC_DRAFT=1 RADIANCE_DRAFT_SCHEDULE=1:8,2:7,4:6,8:5,16:4 RADIANCE_DRAFT_TAU=0.35 \
    RADIANCE_RUN_BWTEST=1

# Fail the build if the native stack does not import, or if a wheel reports a version that does not
# match the source it was built from (a silently mis-stamped wheel is how "aiter 0.0.0" shipped).
# Running this in the RELEASE stage also proves the allowlist above is complete: a library left
# behind by the prune or by the slim base shows up here as an ImportError, not in production.
# Kept GPU-free: no `import aiter` (it runs rocminfo) and no full `import vllm`; versions come from
# package metadata. R4D and the MXFP4 W4A8 GEMM are imported after torch, which is what loads
# libamdhip64. The MXFP4 kernel soft-disables at runtime if it is broken, so assert it here instead.
RUN WANT_VLLM=${VLLM_VERSION} WANT_AITER=${AITER_VERSION} WANT_TF=${TRANSFORMERS_VERSION} \
    python -c 'import os, torch, vllm._C, amdsmi, importlib.metadata as m; \
import r4d, radiance_mxfp4_fp8; \
assert hasattr(radiance_mxfp4_fp8, "launch"), "radiance_mxfp4_fp8.so is missing its launch entry point"; \
v, a, t = m.version("vllm"), m.version("amd-aiter"), m.version("transformers"); \
assert v.startswith(os.environ["WANT_VLLM"]), "vllm wheel reports " + v + ", built tag is " + os.environ["WANT_VLLM"]; \
assert a.startswith(os.environ["WANT_AITER"]), "aiter wheel reports " + a + ", built tag is " + os.environ["WANT_AITER"]; \
assert t == os.environ["WANT_TF"], "transformers is " + t + ", pinned is " + os.environ["WANT_TF"]; \
print("stack OK | vllm", v, "| torch", torch.__version__, "| aiter", a, \
      "| torchvision", m.version("torchvision"), "| triton", m.version("triton"), \
      "| transformers", t, "| r4d", r4d.__version__, "| mxfp4-w4a8 ok")'

# The release image must still be able to COMPILE. AITER JIT-builds its kernels on first use, as a
# pybind11 HIP extension, so the shipped image needs hipcc AND the C++ standard headers AND Python.h.
# `hipcc --version` does not prove any of that -- it passes on an image whose JIT is broken, which is
# exactly how a slim release stage shipped with no libstdc++ headers. This mirrors aiter's real
# compile: build a pybind11 HIP module that includes <cmath>, then import it and call into it.
# torch is imported first because that is what pulls libamdhip64 into the process -- a bare HIP
# extension cannot resolve it on its own (no ROCm entry in ld.so.conf), here or in any prior release.
RUN printf '%s\n' \
      '#include <hip/hip_runtime.h>' \
      '#include <cmath>' \
      '#include <pybind11/pybind11.h>' \
      '__global__ void k(float* o) { o[threadIdx.x] = 1.0f; }' \
      'PYBIND11_MODULE(_jit_probe, m) { m.def("f", [](double x) { return std::sqrt(x); }); }' \
      > /tmp/_jit_probe.hip \
 && hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=${GFX_ARCH} \
      $(python -m pybind11 --includes) /tmp/_jit_probe.hip -o /tmp/_jit_probe.so \
 && python -c "import torch, sys; sys.path.insert(0, '/tmp'); import _jit_probe; assert _jit_probe.f(4.0) == 2.0" \
 && rm -f /tmp/_jit_probe.hip /tmp/_jit_probe.so \
 && echo "runtime JIT toolchain OK (hipcc + libstdc++ headers + Python.h + pybind11)"

ARG RADIANCE_VERSION=0.6.2
ENV RADIANCE_VERSION=${RADIANCE_VERSION}
# The banner reads this file first: one source of truth for the version, so it reports what was
# built even when the image is built without --build-arg.
COPY VERSION /opt/radiance_version
COPY radiance_preamble.py /opt/radiance_preamble.py
COPY radiance_entrypoint.sh /opt/radiance_entrypoint.sh
RUN chmod +x /opt/radiance_entrypoint.sh
ENTRYPOINT ["/opt/radiance_entrypoint.sh"]
