# TP3 support via zero-padded dummy heads

Status: planned 2026-08-31, ahead of third-R9700 arrival. Nothing implemented yet.
Rev 2: added the phased 3-rank R4D all-reduce plan (supersedes the earlier "TP3 AR is RCCL-only" decision).

## Context

The stack runs Qwen3.8-27B-MXFP4 at hardcoded TP=2 on two R9700s. A third R9700 is incoming; goal is TP=3 support ahead of the hardware by padding heads with zero-weight dummies (Megatron-style) so every sharded dimension divides by 3. Upstream vLLM rejected this feature (vllm-project/vllm#11797, closed not-planned), so it's a local patch.

Decisions:
- **Runtime patch** (new patcher + helper module), no offline checkpoint rewrite. TP1/TP2 stays byte-identical: everything gated on `RADIANCE_TP_PAD`.
- **Pad Q 24→36 as well as KV 4→6** so per-rank TP3 is 12Q/2KV = GQA 6, keeping the R4D fp8 attention kernels (compiled for head_dim=256/GQA=6/block16, refuse other GQA).
- **TP3 all-reduces: phased custom 3-rank R4D AR** (see the "3-rank R4D all-reduce" section). Phase 1 ships a 3-rank `exact` kernel for decode-size ARs; prefill rides RCCL until the third card's link is measured. Verified: `radiance_allreduce.py` self-disables cleanly at ws≠2 today (`install_custom_ar` gates on ws==2 at `:264`; `r4d.select("allreduce", world_size=3)` returns None), so TP3 is functional on RCCL from day one regardless of AR-kernel progress.
- **Third card attachment**: CPU-attached M.2 Gen5 x4 riser (~15 GB/s, no chipset hop); both existing cards are on the CPU x16 bifurcated x8/x8 Gen5 (~28 GB/s). The slow link is rank 2's.

**Critical finding**: vLLM 0.27.1 does NOT handle vocab at TP3 — `VocabParallelEmbedding.__init__` does `divide(pad_vocab_size(248320, 64), tp_size)` (`vocab_parallel_embedding.py:296-298`) and asserts. Vocab padding must be patched too (→ 248448, per-rank 82816).

## Geometry

### Target model (`text_config`), active when `RADIANCE_TP_PAD=3`

| Quantity | Stock | Padded | Per-rank TP3 | Why |
|---|---|---|---|---|
| num_attention_heads | 24 | **36** | 12 | keeps GQA=6 for R4D |
| num_key_value_heads | 4 | **6** | 2 | 4∤3; KV bytes/token/GPU unchanged vs TP2 |
| linear_num_key_heads | 16 | **18** | 6 (Hg) | 16∤3 |
| linear_num_value_heads | 48 | **54** | 18 (H) | V/K ratio 3 is baked into the GDN layout/code |
| intermediate_size | 17408 | **17472** | 5824 | multiple of 192 ⇒ per-rank down_proj K%64==0 |
| vocab (runtime pad) | 248320 | **248448** | 82816 | fixes the divide() assert; round to 64·3 |

Dummy heads appended at the **global end** of each head axis: contiguous sharding puts all dummies on rank 2, real q heads only attend real KV heads (q h→kv h//6, v j→k-group j//3 — consistent), ranks 0/1 all-real. GDN kernels take H/Hg as runtime args (`radiance_gdn.py:456`), only HEAD_K=HEAD_V=128/CHUNK=64/CONV_WIDTH=4 are compiled in — no kernel change.

### DFlash2 drafter (different geometry: 32Q/8KV/head128, FP8 with 128×128 blockwise `weight_scale_inv`)

| Quantity | Stock | Padded | Per-rank TP3 |
|---|---|---|---|
| num_attention_heads | 32 | **36** | 12 (GQA stays 4) |
| num_key_value_heads | 8 | **9** | 3 |
| intermediate_size | 17408 | **17664** (multiple of 384=3·128, whole scale-blocks per rank) | 5888 |

`fc`/`attention_conv`/`mlp_conv`/`candidate_selector` are ReplicatedLinear — untouched. Note 9 KV heads ∤ 2 → padded drafter validates only at TP1 (Gate A); use `SPEC_METHOD=mtp` for TP3 bring-up.

### Per-rank MXFP4 kernel audit at TP3 (all functional; two soft notes)

All layers pass K%64 (`radiance_mxfp4.py:572`). Notes: down_proj K=5824 → split-K auto-degrades to sk2 (perf note only); **in_proj_ba per-rank N=36 fails %16**, so (a) `RADIANCE_MXFP4_WPERM` must stay 0 when padding is active (WPERM raises at `:713`) — already the launcher default; (b) `radiance_gdnmerge` skips all 48 GDN merges at TP3 (`:126` gate) — functional, costs the known ~3-6.5% decode; optional phase-2 relaxes the gate for WPERM=0.

## Weight-padding design

A generator wraps the loader's weights iterable and pads each named tensor to the padded-config shape **before** vLLM's sharding weight_loaders run. Fills:
- MXFP4 packed `weight` (u8): `0x00` (two e2m1 +0); `weight_scale` (e8m0 u8): **`0x7F`** (=2^0; never 0xFF NaN)
- FP8 weights: `0x00`; fp8 scales (F32 per-channel / BF16 blockwise scale_inv): **1.0**
- BF16 value tensors (conv1d, A_log, dt_bias): 0.0 (A_log=0 ⇒ finite decay; zero-weight GDN head state provably stays 0)

Target/MTP spec table (suffix-driven; MTP layers are FP8 with F32 per-N-channel scales, so K-pads touch no scale):

| Tensor suffix | Op |
|---|---|
| `.self_attn.q_proj.weight`/`_scale` | rows 12288→18432 — append 12 zero **512-row q/gate units** (verified layout: per-head rows [512h,512h+256)=q, [+256,+512)=gate, `qwen3_next.py:294-302`) |
| `.self_attn.{k,v}_proj.*` | rows 1024→1536 |
| `.self_attn.o_proj.weight` | K-cols: packed 3072→4608 (mtp fp8 6144→9216); u8 scale cols 192→288 |
| `.mlp.{gate,up}_proj.*` | rows 17408→17472 |
| `.mlp.down_proj.weight` | K-cols: packed 8704→8736 (mtp 17408→17472); u8 scale cols 544→546 |
| `.linear_attn.in_proj_qkv.*` | **section pad** [2048→2304 q, 2048→2304 k, 6144→6912 v] (interior zero blocks) |
| `.linear_attn.in_proj_z.*` | rows 6144→6912 |
| `.linear_attn.in_proj_{b,a}.*` | rows 48→54 |
| `.linear_attn.out_proj.*` | K-cols 3072→3456; scale 192→216 |
| `.linear_attn.conv1d.weight` | dim-0 section pad [2304,2304,6912] on [10240,1,4] |
| `.linear_attn.A_log`, `.dt_bias` | 48→54, fill 0.0 |
| untouched | `q_norm`/`k_norm` (per-head_dim [256], shared), `linear_attn.norm`, layernorms, `embed_tokens`, `lm_head`, `mtp.fc` |

Drafter spec (keyed on `architectures==["DFlash2DraftModel"]`): q rows 4096→4608 (scale block-rows 32→36, fill 1.0), k/v 1024→1152 (8→9), o_proj K 4096→4608 (32→36), gate/up rows 17408→17664 (136→138), down K same. All whole 128-blocks; per-rank shards stay whole blocks.

All K-pads land at global K-end and every scale-group boundary is 32-aligned — no e8m0 group straddles real/dummy columns. Numerical safety traced: zero q/k → RMSNorm(0)=0 → RoPE(0)=0 → uniform softmax over V=0 → 0; gate=sigmoid(0)·0=0; GDN dummy heads are independent and stay 0 from 0-init; no NaN source.

## Files to create/modify (padding work)

1. **New `radiance_tp3pad.py`** — helper module (added to the launcher `cp` list):
   - `maybe_pad_config(hf_config, hf_text_config)`: no-op unless `RADIANCE_TP_PAD` set; guards on exact stock values + model identity; sets `_radiance_tp_padded` marker (idempotent). Knobs for the gates: `RADIANCE_TP_PAD_INTERMEDIATE` (default 17472), `RADIANCE_TP_PAD_DRAFTER=0` to skip drafter padding.
   - `pad_weights(weights_iter, model_config)`: the streaming padder; picks target/drafter/none spec from `hf_config`; logs a per-pattern tally (`[radiance.tp3pad] padded q_proj 16x [12288->18432]`) so the serve log proves coverage.
   - `vocab_pad_multiple()`: 64·3 when enabled, else 64.
2. **New `patch_tp3_pad.py`** — patcher in the `_patchlib.apply` style (sentinels, ast.parse, rerun-safe; pattern on `patch_ar_maxbytes.py`), three hunks:
   - `vllm/config/model.py`: after anchor `self.hf_text_config = get_hf_text_config(self.hf_config)` (line 566; `get_hf_text_config` returns the nested object, so mutation propagates everywhere — one hook covers target, MTP, and drafter ModelConfigs) insert the `maybe_pad_config` call.
   - `vllm/model_executor/model_loader/default_loader.py`: wrap anchor `model.load_weights(self.get_all_weights(model_config, model))` (line 394) with `pad_weights(...)`.
   - `vllm/model_executor/layers/vocab_parallel_embedding.py`: make the `pad_vocab_size` pad-to multiple `vocab_pad_multiple()`-aware in `VocabParallelEmbedding.__init__` (env off ⇒ arithmetic identical). At implementation, confirm the vocab weight_loader zero-fills the padded tail and LogitsProcessor masks padded logits (standard machinery; verify against the image's copy).
3. **New `tp3pad_selftest.py`** — GPU-free self-test: read both checkpoints' safetensors headers, run every spec against real shapes, assert the padded-shape/divisibility tables above (catches checkpoint/name drift).
4. **`run_mxfp4_074.sh`**:
   - `TP=${TP:-2}`; derive device list (`0`/`0,1`/`0,1,2`) for `ROCR/HIP_VISIBLE_DEVICES` (line 432); `--tensor-parallel-size "$TP"` (line 545); auto-set `RADIANCE_TP_PAD=3` at TP=3; pass the three env knobs into the container.
   - KV pin (line 169): condition `KV_MEM=18563072000` on `TP=2` — at TP3 re-derive per the header procedure (weights shrink ~1/3, pin moves up).
   - Cache dir suffix `-tp$TP` when TP≠2, `-pad` when padding forced at TP≠3 (padded shapes change every traced graph; stale-cache burns are on record).
   - When padding active: force `RADIANCE_MXFP4_WPERM=0` (would hard-raise), print notice; `AR_OVERLAP/FP8S/NQF` stay at their off defaults at ws≠2.
   - Add `python3 /patches/patch_tp3_pad.py` to the chain (after `patch_quark_mxfp4.py`) and `radiance_tp3pad.py` to the `cp` list (lines 505-527).
5. **Step 0 before any of the above**: diff the container's `qwen3_5.py`, `qwen3_next.py`, `qwen_gdn_linear_attn.py`, and the three hunk-target files against `/home/brian/_newvllm/` (podman run --rm + diff). `_patchlib`'s unique-anchor rule makes drift fatal-loud anyway.

## Verification (works on today's 2 GPUs)

Gates are GSM8K 500q paired + greedy snippet compare + `RADIANCE_MXFP4_CHECKALL`, per project convention — not perplexity byte-compares.

- **Gate A — TP1 at full TP3 geometry** (36/6/18/54/17472 all divide 1; GQA stays 6 → R4D binds both sides). Baseline `TP=1 SPEC_METHOD=mtp` vs same + `RADIANCE_TP_PAD=3`, fresh caches. Check: GSM8K paired; tp3pad tally matches the spec table exactly; R4D report identical kernel bindings; mxfp4 layer census shows no new aiter fallbacks; one run with `RADIANCE_GDN_NANTRACE=1`. Repeat once with `SPEC_METHOD=dflash` — acceptance/draft should match unpadded (drafter padding validates only here).
- **Gate B — TP2, heads-only** (validates sharding across ranks, which TP1 can't): `RADIANCE_TP_PAD=3 RADIANCE_TP_PAD_INTERMEDIATE=17408 RADIANCE_TP_PAD_DRAFTER=0 SPEC_METHOD=mtp TP=2`. 36/6/18/54 all divide 2; intermediate left stock (17472/2=8736 fails K%64 → would hit the known-wrong aiter path; 8704 passes). GQA/rank=6 → R4D live; ws=2 → R4D AR live. Run `GDN_MERGE=0` both sides. Note this exercises odd GDN H=27/Hg=9 which TP3 won't (H=18) — triage odd-H failures against Gate A before treating as blockers.
- **Gate C — deferred TP3 checklist** (when the card lands): smoke boot; confirm R4D "12 q / 2 kv" + gdnmerge skip notice + tally lines + expected AR path per phase; verify P2P across all 3 slots (Gate M below) and profile AR; re-derive `--kv-cache-memory` (never carry the TP2 literal; subtract AR scratch first); GSM8K + CHECKALL vs TP2 prod; SPEC re-sweep (re-sweep after any step-time change) and re-check `DECODE_MAX_M` vs MAXSEQS×(SPEC+1); only then trial NQF/FP8S/AR_OVERLAP one at a time with own cache suffixes.

Padding phase 2 (optional, post-TP3): relax the gdnmerge %16 gate for WPERM=0 (concat of row-major [N, K/2] needs no alignment) to recover the 96-launch merge saving at TP3; A/B with GSM8K.

## 3-rank R4D all-reduce (avoiding the RCCL fallback — phased)

Feasibility verdict: **buildable and well-scoped**. libr4d dispatch is a runtime constraint table (`r4d_registry.hip:80-85`, `C_EQ("world_size", 2)`), not templates — a 3-rank kernel is one new `.hip` + `r4d.h` decl + registry row + `m.def` + `build.sh` UNITS entry, and the Python side already asks `select("allreduce", world_size=...)`; the only hard Python gate is the literal `world_size == 2` at `radiance_allreduce.py:264`.

**Key design finding**: under full-duplex PCIe, the symmetric 3-rank one-shot (each rank pushes its full message to both peers) **ties** a hub/tree on wire time at every size — one-shot moves 2S each way in parallel on rank 2's link; a hub does S up + S down but sequentially. So the existing algorithm generalizes; no topology-aware tree needed unless duplex efficiency measures < ~0.6. Minimum slow-link traffic for any all-reduce is S each direction, so that floor is topology tax, not kernel inefficiency.

### Kernel designs (all additive; the shipping 2-rank units stay byte-identical)

1. **`r4d_ar_oneshot_3rank_exact.hip`** (new file, templated on the 2-rank kernel at `r4d_ar_oneshot_2rank_exact.hip:65-123`): scratch becomes 2 receive regions × 2 seq-parity slots × max_bytes (region of sender i in receiver j = `i - (i > j)`, computed host-side); flags become 2 regions × `AR_MAX_BLOCKS`; push loop stores each chunk to both peers interleaved (one `s_wait_storecnt` drains both); thread-0 releases both peer flags then spins on both in one loop; reduce is a **canonical rank-ascending 3-term fp32 sum** (`vadd3`: `((x0+x1)+x2)` on every rank, `my_pos` selects operand mapping) — this restores the cross-rank bit-identity that 2-term commutativity gave for free. Per-block device-resident seq counters, drain=3, and the double-buffer-reuse argument carry over unchanged per peer pair. New shared `r4d_ar_handshake.h` used **only** by the 3-rank family.
2. **`r4d_ar_oneshot_3rank_exact_nq`** (same file, phase 2): port of the fused AR+residual+RMSNorm+fp8-quant kernel (`exact.hip:140-254`) — only the handshake block and the single `vadd` → `vadd3` change; the epilogue (`:187-253`) is identical. Shape-only dispatch discipline (both ranks branch identically, `radiance_arnq.py:74-79`) keeps seq counters in lockstep.
3. **`r4d_ar_oneshot_3rank_wht6`** (phase 3, conditional): same 6-bit WHT wire format; push stores packed streams to both peers + `loc_pack`; reduce unpacks **three** streams and folds in canonical rank order (single quantization round — no hub double-quant; `-ffp-contract=off` is load-bearing for the 3-product fold). +12 VGPRs in the reduce loop — far under the budget the wht6 perf block ruled infeasible. Re-sweep the block cap (start 96).

### Wire arithmetic (slow link = M.2 Gen5 x4 ≈ 15 GB/s; both directions in parallel)

- Decode `exact`: M=8 (80 KiB) ≈ 11 µs/call wire; M=128 (1.25 MiB) ≈ 167 µs. ×128 sites/step, custom stays ~1.5–2× ahead of RCCL (launch-latency-bound at these sizes) and stays cudagraph-native.
- **Cutoff policy at ws=3: `RADIANCE_AR_MAX_KB=2048`** (covers decode M≤~200); `_quant_ok` returns False at ws=3 in phase 1, so 80 MiB prefill ARs ride RCCL (~7.5 ms/call est.) until phase 3 one-shot wht6 (~5 ms est., ~1.5× win). For scale: TP2's same AR is 1.47 ms — TP3 prefill AR is several× slower under any scheme; that's the link, not the kernel.

### File-level changes (AR work)

- libr4d (in the pinned checkout, carried via `r4d_radiance_extras.patch`): new `r4d_ar_handshake.h`, `r4d_ar_oneshot_3rank_exact.hip` (+`_nq`), phase-3 `r4d_ar_oneshot_3rank_wht6.h/.hip`; append decls to `r4d.h`, constraint rows `C_EQ("world_size",3)` + kernel rows to `r4d_registry.hip` (after `:166`), `m.def`s to `r4d_module.hip` (after `:642`), UNITS + link line in `build.sh`. **Patch-regen trap**: `git add -A` in the checkout before regenerating the patch (the new files are exactly what `git diff` silently drops); gate with a fresh-checkout build.
- `radiance_allreduce.py`: `peer = 1 - self.rank` (`:102`) → sorted peer list with per-peer `ar_ipc_open` + region offsets; scratch `(ws-1)*2*max_bytes`, flags `(ws-1)*maxb*4`; ws-branched dispatch; per-ws `max_bytes` default (ws=3 → 2 MiB unless env-set, coordinated with `patch_ar_maxbytes.py`); the `:264` gate → `in (2, 3)`.
- `radiance_arnq.py`: phase 1 adds `world_size == 2` to the ok-check (its fallback arm is already TP-agnostic → functional TP3); phase 2 adds the ws=3 arm calling `_3rank_exact_nq`.
- `run_mxfp4_074.sh`: at TP=3 export `RADIANCE_AR_MAX_KB=2048`; quant AR off until phase 3.
- New microbench `p2p3_bench.hip` + script on the `mxfp4_work/tier7/ab.sh` pattern: per GPU pair — push bandwidth into fine-grained peer memory, duplex efficiency, flag ping-pong RTT, and the real 2S-out/2S-in one-shot pattern.
- Phase-3 memory note: wht6 scratch at max_bytes=86016 KB is 2×2×84 MiB + loc_pack ≈ **381 MiB/rank** — size it **before** the Gate C `--kv-cache-memory` re-derivation.

### AR phasing & gates

- **Phase 0 (now, pre-hardware)**: write exact(+registry+python+bench); fresh-checkout compile; `select(world_size=3)` returns the new row; validate the bench on the existing pair against known ~28 GB/s; optional 3-process/2-GPU logic smoke (ranks 0,1 on GPU0 via same-device IPC) — validates peer-list/region math, both-flag spin, canonical-order bit-identity; **not** fabric ordering or perf.
- **Gate M (hardware day 1)**: run the microbench across all three pairs before enabling anything. Thresholds: no P2P to rank 2 → park, TP3 stays RCCL (functional baseline); duplex eff < 0.6 → revisit hub for prefill; B_slow ≥ ~13 GB/s (expected with the M.2 riser) → raise the exact cutoff toward 4 MiB and green-light phase 3.
- **Phase 1**: 3-rank exact for decode + gate loosening. Gate: GSM8K 500q paired vs RCCL-only TP3, cross-rank state-identity soak (also covers the drain=3 ordering assumption on the new path; drain=1 is the retreat), cudagraph capture/replay soak, step-time A/B.
- **Phase 2**: 3-rank `_nq` + ws=3 arm. Gate: bit-compare vs (3-rank exact + standalone `launch_add_rms_quant`) on production shapes, then GSM8K.
- **Phase 3 (conditional on Gate M)**: 3-rank one-shot wht6 for prefill; raise max_bytes, re-derive KV pin, block-cap re-sweep, GSM8K paired (checks the 3-term quantized fold) + per-call AR timing vs phase-1 RCCL numbers.

### AR-specific risks

- Verify-width wire scaling: AR bytes scale with M on the slow link (M=128 → ~170 µs/call) — the planned Gate C SPEC re-sweep will land narrower at TP3; don't debug that as a kernel regression.
- SPIN_MAX bailout silently reduces with stale data if the slow peer lags past ~4e9 spins — add a bailout counter to the microbench/soak.
- The M.2 riser slot must actually deliver Gen5 x4 with P2P (`hipDeviceCanAccessPeer` + KFD p2p_links) — Gate M is benchmark-first for exactly this.

## Open risks

1. Rank 2's link quality is the dominant perf unknown, handled by Gate M's benchmark-first thresholds; TP3 prefill may still trail TP2 until AR phase 3 lands.
2. libr4d GDN at H=18/Hg=6 has never run — Gates A (H=54) and B (H=27) give coverage, TP3-only quirk possible until Gate C.
3. gdnmerge/NQF disabled when padded (~3-6.5% decode) until the respective phase-2 items.
4. Padded-vocab interaction with `radiance_drafthead` int2 packing / `radiance_verifyhead` at per-rank vocab 82816 — shape-agnostic in principle, unexercised; check the int2 head arms in Gate A logs.
5. Padded drafter untestable at TP2 (9 KV heads ∤ 2) — dflash at TP3 rests on Gate A only; bring up TP3 with `SPEC_METHOD=mtp`.
