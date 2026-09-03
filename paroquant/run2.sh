#!/bin/bash
# Build the harness once in the radiance image, print kernel resource usage, then run the given
# shell snippet (default: the correctness gate). e.g.  ./run2.sh '/tmp/par && /tmp/par --bench2 pre'
set -e
SNIP=${1:-/tmp/par}
exec podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -e PQ_SHAPE="${PQ_SHAPE:-}" -e PQ_MS="${PQ_MS:-}" \
  -v /home/brian/mxfp4_work/paro:/work:z \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
cd /work
hipcc -O3 -w -std=c++17 --offload-arch=gfx1201 \
  -Rpass-analysis=kernel-resource-usage par_harness.hip -o /tmp/par > /work/build2.log 2>&1 || {
    echo BUILD FAILED; grep -v remark /work/build2.log | head -60; exit 1; }
python3 - <<'PY'
import re
txt=open('/work/build2.log').read()
seen=set()
for m in re.finditer(r'Function Name: (\S+).*?VGPRs: (\d+).*?Occupancy \[waves/SIMD\]: (\d+).*?LDS Size \[bytes/block\]: (\d+)', txt, re.S):
    name=m.group(1)
    if name in seen: continue
    seen.add(name)
    if 'atiled' in name or 'quant_tiled' in name or 'gemm_decode' in name or 'gemm_prefill' in name:
        spill = 'SPILL' if re.search(r'Function Name: '+re.escape(name)+r'.*?ScratchSize \[bytes/lane\]: ([1-9]\d*)', txt, re.S) else ''
        print(f'  {name[:90]:90s} vgpr={m.group(2):>3} occ={m.group(3)} lds={m.group(4)} {spill}')
PY
echo
$SNIP"
