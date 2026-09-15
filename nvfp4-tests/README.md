# nvfp4-tests

Offline checks and studies behind `radiance_nvfp4.py` (NVFP4 -> MXFP4 requantization at load).
All run inside the radiance image via `run_ct.sh` (mounts this repo at /patches and $MODELS at /models):

    ./run_ct.sh 'python3 test_requant.py'        # e2m1 encoder == argmin reference; NVFP4 dequant == vLLM's
    ./run_ct.sh 'python3 study_real.py 0,7,20,40,55'   # requant error on real tensors vs NVFP4 and vs the bf16 original (CPU)
    ./run_ct.sh 'python3 test_drafthead_fp8.py'  # int2 draft head with an FP8 per-channel lm_head: rerank + packing
    ./run_ct.sh 'python3 bench_head_fp8.py'      # head path timing, fp8 vs bf16 lm_head

study_real.py needs both `Qwen3.8-27B-NVFP4` and `Qwen3.8-27B-bf16` under $MODELS.
