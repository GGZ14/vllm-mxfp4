# Development helper for the R4D kernel library.
#
# The kernels are NOT in this repo: they live in libr4d and the image build clones and compiles the
# tag pinned as R4D_VERSION in the Dockerfile. This target does the same thing here, so a locally
# built r4d.so matches the one the image ships and can be dropped into a running container:
#
#     make r4d                            # clone the pinned tag into ./libr4d and build it
#     make r4d R4D_VERSION=v0.2.0         # a different tag
#     make r4d IMAGE=vllm-radiance:dev    # compile against a different image's toolchain
#     make verify
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

RUN = docker run --rm --entrypoint bash -v "$(CURDIR)/$(R4D_DIR):/work" -w /work $(IMAGE) -c

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

clean:
	rm -rf $(R4D_DIR)
