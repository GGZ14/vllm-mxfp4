#!/bin/bash
# serve-tp1.sh -- serve on 1 card(s). A thin wrapper: everything is serve-mxfp4.sh with TP=1 pinned.
#   ./serve-tp1.sh                      # first 1 usable card(s) (gpu-detect.sh order)
#   GPUS=1 ./serve-tp1.sh    # name the card(s)
#   ./serve-tp1.sh --enforce-eager      # extra args go straight to vllm serve
# All serve-mxfp4.sh knobs apply (MAXSEQS, MAXLEN, CHUNK, SPEC, DETACH, PORT, ...). See README.md.
# TP=1 turns on the single-GPU profile automatically: fp16 ssm cache, MAXSEQS 3, MAXLEN 220000,
# CHUNK 2560 (the long-context shape -- ~220k KV tokens off the pin in kv-profiles.tsv; pass
# MAXSEQS=8 MAXLEN=65536 CHUNK=4096 for the measured throughput shape instead),
# libr4d rx9 (narrow-state GDN kernels) and the fp8 stream at TP=1. SINGLE_GPU_PROFILE=0 turns it
# off. Lazy GDN snapshots (RADIANCE_GDN_LAZY, rx10) are OFF by default -- they corrupt multi-turn
# chat; see the note at serve-mxfp4.sh's RADIANCE_GDN_LAZY block.
exec env TP=1 "$(cd "$(dirname "$0")" && pwd)/serve-mxfp4.sh" "$@"
