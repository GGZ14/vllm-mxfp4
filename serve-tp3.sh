#!/bin/bash
# serve-tp3.sh -- serve on 3 card(s). A thin wrapper: everything is serve-mxfp4.sh with TP=3 pinned.
#   ./serve-tp3.sh                      # first 3 usable card(s) (gpu-detect.sh order)
#   GPUS=0,1,2 ./serve-tp3.sh    # name the card(s)
#   ./serve-tp3.sh --enforce-eager      # extra args go straight to vllm serve
# All serve-mxfp4.sh knobs apply (MAXSEQS, MAXLEN, CHUNK, SPEC, DETACH, PORT, ...). See README.md.
# TP=3 is explicit only (zero-weight dummy heads, own cache dir; see README "TP=3").
exec env TP=3 "$(cd "$(dirname "$0")" && pwd)/serve-mxfp4.sh" "$@"
