#!/bin/bash
# serve-tp2.sh -- serve on 2 card(s). A thin wrapper: everything is serve-mxfp4.sh with TP=2 pinned.
#   ./serve-tp2.sh                      # first 2 usable card(s) (gpu-detect.sh order)
#   GPUS=0,1 ./serve-tp2.sh    # name the card(s)
#   ./serve-tp2.sh --enforce-eager      # extra args go straight to vllm serve
# All serve-mxfp4.sh knobs apply (MAXSEQS, MAXLEN, CHUNK, SPEC, DETACH, PORT, ...). See README.md.
# TP=2 is the auto-detected default on a two-card host; this pins it on hosts with more cards.
exec env TP=2 "$(cd "$(dirname "$0")" && pwd)/serve-mxfp4.sh" "$@"
