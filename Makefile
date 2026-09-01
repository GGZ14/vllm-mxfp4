# Development helper for the R4D kernel library and this fork's own HIP kernel.
#
# The kernels are NOT in this repo: they live in libr4d and the image build clones and compiles the
# tag pinned as R4D_VERSION in the Dockerfile. This target does the same thing here, so a locally
# built r4d.so matches the one the image ships and can be dropped into a running container:
#
#     make r4d                            # clone the pinned tag into ./libr4d and build it
#     make r4d R4D_VERSION=v0.2.0         # a different tag
#     make r4d IMAGE=vllm-radiance:dev    # compile against a different image's toolchain
#     make verify
#     make radiance_mxfp4_fp8.so       # this fork's MXFP4 W4A8 GEMM
#     make clean
#
# The library must be compiled with the same ROCm/hipcc it loads against at runtime, i.e. the
# toolchain inside the vllm-radiance image, so hipcc runs in a throwaway container.

GFX_ARCH ?= gfx1201
IMAGE    ?= vllm-radiance:$(shell cat VERSION 2>/dev/null || echo latest)
PYTHON   ?= python3
# Defaults read out of the Dockerfile so there is one pin, not two.
R4D_REPO    ?= $(shell sed -n 's/^ARG R4D_REPO=//p' Dockerfile)
R4D_VERSION ?= $(shell sed -n 's/^ARG R4D_VERSION=//p' Dockerfile)
R4D_DIR     ?= libr4d

# podman where it exists, docker otherwise -- same as the launcher's auto-detection.
RUNTIME ?= $(shell command -v podman >/dev/null 2>&1 && echo podman || echo docker)
RUN     = $(RUNTIME) run --rm --entrypoint bash -v "$(CURDIR)/$(R4D_DIR):/work" -w /work $(IMAGE) -c
# The MXFP4 kernel is a single .hip at the repo root, so it mounts the root rather than $(R4D_DIR).
RUN_ROOT = $(RUNTIME) run --rm --entrypoint bash -v "$(CURDIR):/work" -w /work $(IMAGE) -c
HIPCC_FLAGS = -O3 -std=c++17 -fPIC -shared --offload-arch=$(GFX_ARCH) -Wno-unused-result

.DEFAULT_GOAL := r4d
.PHONY: r4d verify clean

$(R4D_DIR):
	git clone --depth 1 -b $(R4D_VERSION) $(R4D_REPO) $(R4D_DIR)

r4d: $(R4D_DIR)
	@cd $(R4D_DIR) && git fetch --depth 1 origin tag $(R4D_VERSION) 2>/dev/null; \
	  git -C $(R4D_DIR) checkout -q $(R4D_VERSION)
	@$(RUN) 'GFX_ARCH=$(GFX_ARCH) PYTHON=$(PYTHON) ./build.sh'
	@echo "[make] built $(R4D_DIR)/r4d.so from $(R4D_VERSION)"

verify:
	@$(RUN) 'PYTHONPATH=$$PWD $(PYTHON) -c "import torch, r4d; \
	  print(\"[verify] r4d\", r4d.__version__, \"OK:\", [n for n in dir(r4d) if not n.startswith(\"_\")])"'

# This fork's own kernel: the MXFP4 W4A8 fp8-WMMA GEMM. It is specific to this fork rather than
# general to gfx1201, so it stays here instead of in libr4d, and the image build compiles it too.
%.so: %.hip
	@$(RUN_ROOT) 'INC=$$($(PYTHON) -m pybind11 --includes); \
	  hipcc $(HIPCC_FLAGS) $$INC $< -o $@ && echo "[make] built $@"'

clean:
	rm -rf $(R4D_DIR) radiance_mxfp4_fp8.so
