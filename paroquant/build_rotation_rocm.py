import io, shutil, subprocess, sys, sysconfig, time
from pathlib import Path
import torch
from torch.utils.hipify import hipify_python

TORCH = Path(torch.__file__).parent
PYINC = sysconfig.get_paths()["include"]
src = Path("/src/paroquant/kernels/cuda")
work = Path("/tmp/pqsrc"); shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
for f in src.iterdir():
    if f.suffix in (".cu", ".cuh"):
        shutil.copy(f, work / f.name)
hipify_python.hipify(project_directory=str(work), output_directory=str(work),
                     includes=[str(work / "*")], is_pytorch_extension=True, show_detailed=False)

# ROCm compat: HIP's bf16 header has no __floats2bfloat162_rn. Patch the hipified header only,
# so the upstream source stays pristine and can be re-pulled.
h = work / "rotation_hip.cuh"
s = io.open(h).read()
old = "return __floats2bfloat162_rn(a, b);"
new = ("__hip_bfloat162 r; r.x = __float2bfloat16(a); r.y = __float2bfloat16(b); return r;")
assert s.count(old) == 1, f"compat patch did not match ({s.count(old)})"
io.open(h, "w").write(s.replace(old, new))
print("compat patch applied")

so = work / "paroquant_rotation.so"
cmd = ["/opt/rocm/bin/hipcc", "-O3", "-std=c++17", "-fPIC", "-shared",
       "--offload-arch=gfx1201", "-ffast-math", "-w",
       f"-I{TORCH}/include", f"-I{TORCH}/include/torch/csrc/api/include",
       f"-I{PYINC}", "-I/opt/rocm/include",
       "-D__HIP_PLATFORM_AMD__=1", "-DUSE_ROCM=1",
       str(work / "rotation.hip"), "-o", str(so),
       f"-L{TORCH}/lib", "-ltorch", "-ltorch_cpu", "-ltorch_hip", "-lc10", "-lc10_hip"]
print("building...")
t0 = time.time()
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("=== BUILD FAILED ==="); print(r.stdout[-2000:]); print(r.stderr[-6000:]); sys.exit(1)
print(f"=== BUILD OK in {time.time()-t0:.0f}s -> {so.stat().st_size/1024:.0f} KiB ===")
torch.ops.load_library(str(so))
print("registered:", torch.ops.rotation.rotate)
shutil.copy(so, "/src/paroquant_rotation_rocm.so")
