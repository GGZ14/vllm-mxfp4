#!/bin/bash
set -e
SP=/home/brian/mxfp4_work/ar
exec podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -v $SP:/work:z --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
cd /work
hipcc -O3 -w -std=c++17 --offload-arch=gfx1201 -Rpass-analysis=kernel-resource-usage \
  decopt.hip -o /tmp/decopt > /work/decopt_build.log 2>&1 || { echo BUILD FAILED; tail -40 /work/decopt_build.log; exit 1; }
echo '--- prefill variant resources (VGPR / occ / LDS / spill) ---'
python3 - <<'PY'
import re
cur=None
for ln in open('/work/decopt_build.log'):
    m=re.search(r'Function Name: (\S+)',ln)
    if m: cur=m.group(1); vals={}
    for k in ('VGPRs:','Occupancy','LDS Size','VGPRs Spill'):
        if k in ln and cur:
            v=ln.split(':')[-1].split('[')[0].strip(); vals[k]=v
            if k=='VGPRs Spill' and 'prefill_opt' in cur or (k=='VGPRs Spill' and 'gemm_prefill' in cur):
                import subprocess
                o=re.search(r'ar_prefill_optILi2ELi(\d+)',cur)
                name=('opt%s'%o.group(1)) if o else ('shipped' if 'gemm_prefill' in cur else cur[:24])
                print('  %-10s VGPR=%-4s occ=%-3s LDS=%-6s spill=%s'%(name,vals.get('VGPRs:','?'),vals.get('Occupancy','?'),vals.get('LDS Size','?'),vals.get('VGPRs Spill','?')))
PY
echo
/tmp/decopt $*"
