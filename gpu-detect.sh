#!/bin/bash
# gpu-detect.sh -- work out which AMD GPUs this host can serve on, pick a tensor-parallel size
# that fits both the hardware and the model's head counts, and look up a KV cache pin measured
# for that hardware.
#
#   ./gpu-detect.sh          print what was found and the values serve-mxfp4.sh will use
#   . ./gpu-detect.sh        source it: sets RAD_* in the calling shell
#
# Everything here comes from sysfs. That is deliberate: rocm-smi is not installed on a stock
# host (it is inside the image, not outside it), and `podman run`-ing the image just to count
# cards costs several seconds and takes a GPU lock that a running server may still hold.
#
# WHY A VRAM FLOOR AND NOT A COUNT OF CARDS
# Counting amdgpu render nodes is wrong on any desktop part with integrated graphics. The
# reference box reports THREE: two R9700 (0x7551, 32624 MiB) and the Granite Ridge iGPU
# (0x13c0, 2048 MiB). Using the count picks --tensor-parallel-size 3, which fails the head
# divisibility rule below and would shard 4-bit weights onto a 2 GiB display adapter.
# MIN_GPU_MIB excludes it. Raise it to also skip a small discrete card you do not want used.

RAD_MIN_GPU_MIB=${MIN_GPU_MIB:-8192}

# Tensor-parallel sizes this checkpoint supports. Qwen3.8-27B has num_attention_heads=24,
# linear_num_key_heads=16 and linear_num_value_heads=48, and TP must divide all three. That
# rules out 3, 6 and 12 even though they divide 24 -- the GDN linear-attention heads are what
# reject them, which is exactly what TP3_PADDING_PLAN.md exists to fix. 16 is out because it
# does not divide 24. Override only if you are serving a different checkpoint.
RAD_TP_ALLOWED=${TP_ALLOWED:-"8 4 2 1"}

# Marketing names for the parts this has actually been run on. Anything else falls back to
# lspci if it is installed, and to the raw PCI id if it is not -- a name is for the log line,
# nothing branches on it.
rad_gpu_name() {
  case "$1" in
    7551) echo "Radeon AI PRO R9700" ;;
    744c) echo "Radeon RX 7900 XTX/XT" ;;
    7448) echo "Radeon PRO W7900" ;;
    73a5) echo "Radeon RX 6950 XT" ;;
    *)
      local n=""
      if command -v lspci >/dev/null 2>&1; then
        n=$(lspci -d "1002:$1" 2>/dev/null | sed -n '1s/.*\[\([^]]*\)\].*/\1/p')
      fi
      if [ -n "$n" ]; then echo "$n"; else echo "AMD 0x$1"; fi ;;
  esac
}

# Fills RAD_GPU_INDICES / _COUNT / _MIB / _ID / _NAME / _SIG / _SKIPPED and RAD_TP.
#
# The HIP index of a card is its position among the amdgpu render nodes in PCI probe order,
# which is the order the kernel allocates renderD numbers in. Verified on the reference box:
# renderD128 = 03:00.0, renderD129 = 06:00.0, renderD130 = 7e:00.0 (the iGPU) -> HIP 0, 1, 2.
# Non-AMD render nodes are skipped WITHOUT advancing the counter, because HIP never enumerates
# them. If your host disagrees, name the indices yourself: GPUS=0,1 ./serve-mxfp4.sh
rad_detect_gpus() {
  local d idx=0 pciid mib cand="" t
  RAD_GPU_SKIPPED=""
  # idx -> "pciid:mib" for every amdgpu node, so the figures can be re-read per index later.
  declare -A _mib=() _pid=()

  for d in $(ls -d /sys/class/drm/renderD* 2>/dev/null | sort -V); do
    [ "$(cat "$d/device/vendor" 2>/dev/null || true)" = "0x1002" ] || continue
    pciid=$(cat "$d/device/device" 2>/dev/null || echo 0x0000); pciid=${pciid#0x}
    mib=$(( $(cat "$d/device/mem_info_vram_total" 2>/dev/null || echo 0) / 1048576 ))
    _pid[$idx]=$pciid; _mib[$idx]=$mib
    if [ "$mib" -lt "$RAD_MIN_GPU_MIB" ]; then
      RAD_GPU_SKIPPED="$RAD_GPU_SKIPPED $idx:0x$pciid:${mib}MiB"
    else
      cand="$cand $idx"
    fi
    idx=$((idx + 1))
  done

  # GPUS= names HIP indices directly and overrides the VRAM floor.
  if [ -n "${GPUS:-}" ]; then cand=$(echo "$GPUS" | tr ',' ' '); RAD_GPU_SKIPPED=""; fi

  set -- $cand
  RAD_GPU_COUNT=$#

  # Largest supported TP the candidates can fill. A 3-card host serves on 2 and leaves one
  # idle: correct, not a bug, until the TP3 padding work lands.
  RAD_TP=1
  for t in $RAD_TP_ALLOWED; do
    if [ "$RAD_GPU_COUNT" -ge "$t" ]; then RAD_TP=$t; break; fi
  done

  # Truncate to TP FIRST, then read the card figures back off only the cards that survived.
  # Deriving them during the scan instead reports a card the run will never touch: with the
  # VRAM floor lowered far enough to admit the reference box's 2 GiB iGPU, the minimum came
  # from that iGPU while the run still served on the two R9700, and the signature it produced
  # (2x7551-2048) matched no measured row.
  cand=$(echo "$cand" | tr -s ' ' | sed 's/^ //;s/ /,/g')
  RAD_GPU_INDICES=$(echo "$cand" | cut -d, -f1-"$RAD_TP")

  RAD_GPU_MIB=0; RAD_GPU_ID=0000
  local i first=1
  for i in $(echo "$RAD_GPU_INDICES" | tr ',' ' '); do
    [ -n "${_mib[$i]:-}" ] || continue
    # The KV cache is sized per rank, so the SMALLEST card binds the whole run.
    if [ "$first" = 1 ] || [ "${_mib[$i]}" -lt "$RAD_GPU_MIB" ]; then RAD_GPU_MIB=${_mib[$i]}; fi
    if [ "$first" = 1 ]; then RAD_GPU_ID=${_pid[$i]}; first=0; fi
  done

  RAD_GPU_NAME=$(rad_gpu_name "$RAD_GPU_ID")
  RAD_GPU_SIG="${RAD_TP}x${RAD_GPU_ID}-${RAD_GPU_MIB}"
}

# ---------------------------------------------------------------- KV cache pin lookup
# A pin (--kv-cache-memory) is worth having because vLLM's own profiling is deliberately
# conservative: non_kv_cache_memory carries the profile run's TRANSIENT activation peak plus
# the cudagraph estimate, both of which sit above what steady-state serving actually needs.
# On the reference box that conservatism is ~0.93 GiB per rank, which is 5.7% of the KV cache.
#
# The size of that margin is a property of the hardware AND the batch shape, and it is not
# derivable from anything vLLM logs -- calibrate-kv.sh finds it by measurement. Rows measured
# that way live in the two tables below; anything not in them falls back to profiling, which
# is always safe and merely leaves the margin on the table.
RAD_KV_TABLE=${KV_TABLE:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/kv-profiles.tsv}
RAD_KV_TABLE_LOCAL=${KV_TABLE_LOCAL:-${XDG_CACHE_HOME:-$HOME/.cache}/radiance-mxfp4/kv-profiles.local.tsv}

# rad_kv_lookup <sig> <maxseqs> <chunk> <maxlen> <spec_method>
# Echoes the pin in bytes, or nothing. The local table is read LAST so a locally measured row
# beats a shipped one for the same key -- your own hardware outranks our table on your host.
rad_kv_lookup() {
  local sig=$1 seqs=$2 chunk=$3 maxlen=$4 spec=$5 f hit=""
  for f in "$RAD_KV_TABLE" "$RAD_KV_TABLE_LOCAL"; do
    [ -r "$f" ] || continue
    local row
    row=$(awk -F'\t' -v s="$sig" -v q="$seqs" -v c="$chunk" -v l="$maxlen" -v m="$spec" \
      '$1 !~ /^#/ && $1==s && $2==q && $3==c && $4==l && $5==m {print $6}' "$f" | tail -1)
    if [ -n "$row" ]; then hit=$row; fi
  done
  echo "$hit"
}

rad_detect_gpus

# Run directly (not sourced) -> report. `return` fails outside a function in a sourced file
# only on very old bash, so guard on BASH_SOURCE instead of trusting its exit status.
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then
  echo "AMD GPUs usable:  $RAD_GPU_COUNT x $RAD_GPU_NAME (0x$RAD_GPU_ID, $RAD_GPU_MIB MiB each)"
  echo "HIP indices:      ${RAD_GPU_INDICES:-<none>}"
  [ -n "$RAD_GPU_SKIPPED" ] && echo "skipped (<${RAD_MIN_GPU_MIB} MiB):$RAD_GPU_SKIPPED"
  echo "tensor parallel:  $RAD_TP   (supported: $RAD_TP_ALLOWED)"
  echo "hardware sig:     $RAD_GPU_SIG"
  kv=$(rad_kv_lookup "$RAD_GPU_SIG" "${MAXSEQS:-8}" "${CHUNK:-8192}" "${MAXLEN:-262144}" "${SPEC_METHOD:-dflash}")
  if [ -n "$kv" ]; then
    echo "KV cache pin:     $kv bytes ($(awk -v b="$kv" 'BEGIN{printf "%.2f", b/1073741824}') GiB/GPU, measured)"
  else
    echo "KV cache pin:     none for this signature -- vLLM will profile (safe)"
    echo "                  ./calibrate-kv.sh measures one for this host"
  fi
fi
