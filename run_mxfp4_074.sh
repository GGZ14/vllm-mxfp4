#!/bin/bash
# Compatibility shim. This launcher is now ./serve-mxfp4.sh -- the old name tracked the 0.7.4
# image it was written for, which it stopped launching two image versions ago.
#
# Two defaults changed with the rename, both toward what production actually serves:
#   SPEC_METHOD   mtp -> dflash    (needs Qwen3.8-27B-DFlash2-FP8; SPEC_METHOD=mtp restores it)
#   chat template a host-local ~/.cache/huggingface file -> this repo's qwen3.8-enhanced.jinja
#                 (CHAT_TEMPLATE=<path> restores any other one)
echo "[run_mxfp4_074.sh] renamed to serve-mxfp4.sh -- forwarding (see the header for two default changes)" >&2
exec "$(cd "$(dirname "$0")" && pwd)/serve-mxfp4.sh" "$@"
