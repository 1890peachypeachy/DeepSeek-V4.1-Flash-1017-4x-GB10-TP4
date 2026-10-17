# DeepSeek-V4.1-Flash-1017 on 4 GB10 nodes

Measured 2026-10-17 by Adapt AI Systems on top of knapcio's v2.4 (`27616be`). Every number below comes
from the raw files in [`results/ds1017/`](results/ds1017/) or from the [hairpin-ring repo](https://github.com/adapt-ai-systems/gb10-hairpin-ring-switch-hop)
named in the last section. Where a number was not measured, this page says so.

## What changed in the weights (V4.1-Flash -> V4.1-Flash-1017)

Compared through the Hugging Face tree API (per-file LFS hashes) on 2026-10-17: `deepseek-ai/DeepSeek-V4.1-Flash-1017` @`1bae0e10`
(full revision `1bae0e1088993b9b82de08f59de5be5a34e42170`) against two revisions of `deepseek-ai/DeepSeek-V4.1-Flash`:

- `fb2764a5`: the revision knapcio's recipe pinned before this fork (`HF_REVISION` in `.env.tp4.example` and `start.sh` at `27616be`). This is the comparison that matters for a user of the recipe.
- `314113f8`: the head of the V4.1-Flash repository on that day (last modified 2026-10-16), shown because it is what someone downloading V4.1-Flash today gets.

| item | 1017 vs `fb2764a5` (recipe pin) | 1017 vs `314113f8` (repo head) |
|---|---|---|
| `config.json` | byte-identical (same architecture `DeepseekV41ForCausalLM`, 40 layers, 384 experts, Engram layers 1 and 14, DSpark with 3 MTP layers) | byte-identical |
| `model.safetensors.index.json` | identical: same 96,085 tensor names, same tensor-to-shard map, `total_size` 510,286,023,000 | identical |
| `tokenizer.json`, `tokenizer_config.json` | identical | identical |
| the 48 shards | all 48 differ in content, all 48 have the same byte size | same |
| `encoding/` and `inference/` files | `encoding/encoding.py`, `encoding/test_encoding.py`, `encoding/README.md`, `encoding/tests/test_output_1.txt`, `inference/model.py` and `inference/requirements.txt` differ (sizes differ); the rest are identical. We did not read these diffs. | identical |
| files only in the older revision | `DeepSeek_V41_Tech_Report.pdf`, `evaluation/`, `assets/` | the same, plus `chat_template.jinja` and `.gitignore` (these two are not in `fb2764a5`) |

Against the repository head the `encoding/` and `inference/` files are identical, so the differences against `fb2764a5` in those folders
were made in V4.1-Flash before 1017.

The weights are a drop-in swap for this recipe: same config, same tensor names and shard map, same shard sizes. `chat_template.jinja` is
not read by `boot.py`, `start.sh` or `adapter/` (they only use `chat_template_kwargs`), and the recipe booted and answered without it.
We did not test whether the `encoding/` differences change anything for this recipe.

## What this fork changes

- `start.sh` and `.env.tp4.example`: `HF_REPO` / `HF_REVISION` now default to the 1017 repository and revision. The original
  values are kept as a comment in `.env.tp4.example`. We did not run the `download` step itself; we fetched the files with
  `hf` and checked every LFS file with `sha256sum` against the Hugging Face file list (49 entries, 0 bad, on both machines that hold a full copy).
- [`.env.tp4.1017-ring.example`](../.env.tp4.1017-ring.example): the environment we booted from, as a ring example with placeholder addresses.
- This page and the raw results.

Nothing else in knapcio's code was changed for the numbers on this page. (The ring boot ran with a local, un-upstreamed
edit to `files/nccl.sh`, the worker settings and `doctor` that accepts `NCCL_IB_GID_AUTO=1`; see the note in the env example.)

## Moving from V4.1-Flash to 1017

1. Fetch the new weights to a new directory (`HF_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017`, `HF_REVISION=1bae0e1088993b9b82de08f59de5be5a34e42170`, 510.3 GB, 48 shards + metadata).
2. **Repack the Engram rows. Old packs do not work with the new weights.** The packs are derived from the weights (`scripts/pack_engram.py`,
   one pack per rank). On our nodes each rank wrote two files of 25.3 GB each (about 50.7 GB, 47.2 GiB, per rank) in 57 to 183 s;
   see [`results/ds1017/engram-pack.txt`](results/ds1017/engram-pack.txt).
3. Point `MODEL_DIR`, `COMMON_MODEL`, `ENGRAM_DIR` and `WORKER_ENGRAM_DIR` at the new directories.
4. The DSpark timing tables in `state-tp4/` are timing data, not weights. We reused them and did not re-profile. We did not check whether a re-profile changes the accept rate.

## Our 1017 boot

- Hardware: 2x NVIDIA DGX Spark + 2x Acer GN100 (GB10), TP4, EP 1, DSpark k=5, chunked prefill 8192, mem fraction 0.78, 1,048,576-token context.
- Image: v2.4 `Dockerfile.canary-roce`, SGLang `f80c91a4b` (image digest begins `779f12be`).
- Fabric for the table below: a **plain switchless ring** (patched NCCL 2.30.7 with sparkring's switchless-cycle change, RoCEnante off), first PCI domain only (2 HCAs).
  One of the four ring links ran through a switch instead of a cable; NCCL does not see the difference.
- Boot: 543 s from `serve` to Ready. KV pool `max_total_num_tokens=4,198,912`. Smoke test (19+23) returned 42.
- Clocks: the sparkDash runs used `-lgc 0,2200`. The H0 run used default clocks.

## Results on the plain ring (before any hairpin work)

Harness H1 is sparkDash 1.8.8 with knapcio's README protocol: 256 new tokens, temperature 0, thinking off, two discarded prose c1 warm-ups,
c1 = median of 3 runs, c4 = median of 2, c8 and c16 one run each, 0 failed streams. Raw JSON: [`results/ds1017/sparkdash-plain-ring/`](results/ds1017/sparkdash-plain-ring/).

| prompt type | c1 (median of 3) | c4 aggregate | c8 aggregate | c16 aggregate | v2.4 README, switched fabric, old weights (c1 / c8 / c16) |
|---|---:|---:|---:|---:|---|
| prose | 68.5 | 154.3 | 225.3 | 333.7 | 89.9 / 237.2 / 341.3 |
| code | 122.3 | 235.0 | 313.7 | 449.4 | 131.0 / 306.8 / 440.3 |
| structured | 138.8 | 164.3 | 236.1 | 377.6 | c1 152.8 |
| json | 116.4 | 267.3 | 423.0 | 653.9 | n/a |

The last column is knapcio's published result for different weights on a switched fabric. It is a reference, not a same-weights comparison.
Prose c1 is one greedy prompt; knapcio's README warns that a different greedy text moves it by about 10 %, and 1017 produces a different greedy text,
so code and structured c1 (6 to 9 % below the README) are the better speed signal. The c8 and c16 aggregates are at or near the README values.

Prefill, tok/s, cold. The sparkDash filler is one repeated token, so the Engram row cache inflates it. The real-text rows use unique salted
prompts built from documentation and source code (`scripts/realprefill.py` in the hairpin repo is the script; you must supply your own corpus):

| context | 4k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| sparkDash filler, this ring | 4,577 | 5,060 | 5,153 | 5,169 | 5,043 |
| sparkDash filler, v2.4 README (switched, old weights) | 4,756 | 5,790 | 5,941 | 5,969 | 5,896 |
| real text, this ring (prompt tokens 3,440 / 13,999 / 28,622 / 53,163 / 106,767) | 3,580 | 4,475 | 4,676 | 4,727 | 4,747 |

H0 (our older harness: non-streaming, `completion_tokens` / wall time, thinking off, 700-token essay, default clocks):
single stream 55.29 / 55.25 / 55.61 tok/s; c4 aggregate 119.4 tok/s; cold prefill of random words 4,180 tok/s at 9,008 tokens, 3,935 tok/s at 36,002 tokens, 4,293 tok/s at 143,977 tokens.
Raw output: [`results/ds1017/h0-1017.out`](results/ds1017/h0-1017.out).

Quality, `scripts/qeval.py` (75 auto-scored tasks, c1, temperature 0): **72/75 (96.0 %)**, median 77.03 tok/s, one task hit the token cap.
Misses: `code_interval_intersect`, `json_escape`, `math_m9` (got 4.0, want 38.0). knapcio's v2.4 README reports 72/75 too. One run only.
Raw: [`results/ds1017/qeval-plain-ring.txt`](results/ds1017/qeval-plain-ring.txt).

TX check on the GN100 nodes (kernel `7.0.0-1019-nvidia`, `kho=off`): a report in MiaAI-Lab's repo says `kho=off` caps ConnectX-7 TX at 12.7 Gb/s on a newer 6.17 kernel.
We saw up to 31.9 Gb/s per RDMA function during prefill, and `dmesg` showed no `enomem` / `reg_mr` lines on any of the four nodes before the boot, after H0 and after H1.

## What we did not measure

- No same-weights comparison against a switched fabric (the old V4.1 weights were not on the ring, and 1017 was not on a switch).
- No needle or long-context test above 144K tokens. No thinking-on test.
- The table above is a single boot.

## Optional: hairpin mesh with one switch hop in the ring

The ring above is a plain ring. The same four nodes also ran knapcio's RoCEnante-over-hairpin-mesh option
(sparkring's design, see `docs/switchless-ring.md`) in a ring where **one link goes through a switch instead of a cable**, with one node's
two ring ports used in reversed roles through a small `plan.py` patch, and a GID fix for the mixed Spark/GN100 fleet. Same weights, same day,
same harness, against the plain ring above:

| sparkDash, tok/s (median of 3) | plain ring | hairpin mesh, one switch hop | change |
|---|---:|---:|---:|
| code c1 | 122.9 | 133.7 | +8.8 % |
| structured c1 | 138.5 | 152.0 | +9.7 % |
| prose c1 | 68.2 | 84.3 | +23.6 % |
| code c4 aggregate | 235.2 | 247.0 | +5.0 % |
| prose c4 aggregate | 154.2 | 174.3 | +13.0 % |
| qeval median (75 tasks, one run each) | 77.03 | 84.99 | +10.3 %, pass 72/75 -> 71/75 |

Prefill did not improve (8k real text 4,030 -> 3,877 tok/s; 32k real text 4,634 -> 4,650 tok/s). Setup, patches, the switch-hop test, results
and known issues (settings that reset at reboot) are in
**[adapt-ai-systems/gb10-hairpin-ring-switch-hop](https://github.com/adapt-ai-systems/gb10-hairpin-ring-switch-hop)**.
The mesh itself is [FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring) (Apache-2.0), integrated into this recipe by knapcio's `scripts/ring_mesh/`.

## Serve config we adopted: dual-PCI NCCL plus hairpin mesh

After the mesh-only test above we added a second change: NCCL routing across both PCIe domains of each node, so all four RDMA functions carry traffic.
Same weights, same ring (one hop through a switch), same day. Harness H1 subset (sparkDash 1.8.8, 256 new tokens, temperature 0, thinking off, `-lgc 0,2200`, median of 3).
Raw runs: [`results/ab-runs/baseline` and `results/ab-runs/armAB`](https://github.com/adapt-ai-systems/gb10-hairpin-ring-switch-hop/tree/main/results/ab-runs) in the switch-hop repository.

| row | plain ring (`.env.tp4.1017-ring.example`) | dual-PCI + hairpin | change |
|---|---|---|---|
| decode c1: prose / code / structured, tok/s | 68.2 / 122.9 / 138.5 | 84.3 / 137.5 / 154.9 | +24 / +12 / +12 % |
| decode c4 aggregate: prose / code / structured, tok/s | 154.2 / 235.2 / 164.3 | 175.9 / 248.7 / 203.0 | +14 / +6 / +24 % |
| filler prefill 8K / 32K, tok/s | 4,671 / 5,140 | 4,712 / 5,618 | +1 / +9 % |
| real-text prefill 8K / 32K, tok/s | 4,030 / 4,634 | 4,450 / 5,103 | +10 / +10 % |
| KV pool, tokens | 4,198,912 | 4,327,424 | |
| `qeval.py`, 75 tasks, c1 | 72/75, median 77.03 tok/s (1 run) | 71/75 in all 3 runs, median of the 3 runs' medians 84.6 (84.6 / 85.47 / 83.0) | |
| draft tokens accepted per step, c1, 256 tokens | not measured | prose 2.78, code 5.12, structured 5.22 | |

The plain ring misses `code_interval_intersect`, `json_escape` and `math_m9`. The dual-PCI + hairpin config misses the same three plus `json_count`.
Prose c1 is one greedy prompt and carries the acceptance caveat given in the switch-hop repository. Which of the two changes does what:
mesh only (the table above) moves decode and leaves prefill flat; dual-PCI NCCL only did not move decode (within 1.5 %) and moved real-text prefill +11 % / +9 %;
the two together keep the mesh decode gain and add +6 % / +12 % filler and +15 % / +10 % real-text prefill over mesh only. That the gains add
because the two paths carry different traffic (small all-reduces on the mesh, large prefill collectives on NCCL) is our inference; we did not trace it.

### Environment changes against `.env.tp4.1017-ring.example`

```
IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1        # was rocep1s0f0,rocep1s0f1
NCCL_HOST_DIR=$HOME/nccl-2.30.7-dual                          # was $HOME/nccl-2.30.7-ring
NCCL_IB_EXTENDED_IPV4_GIDS=1                                  # new
NCCL_IB_PRESERVE_PCI_DOMAIN=1                                 # new
NCCL_IB_ROUTE_DIAGNOSTICS=1                                   # new
NCCL_IB_QPS_PER_CONNECTION=1                                  # new
```

In `EXTRA_CONTAINER_ENV`, remove `SGLANG_ROCE_ALLREDUCE=0` and append the line that `scripts/ring_mesh/plan.py` writes to its `env.txt`:

```
DSV41_ROCE_RING=1 SGLANG_ROCE_ALLREDUCE=1 SGLANG_ROCE_MAX_SIZE=262144 DSV41_ROCE_GATHER=262144
B12X_ROCE_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
B12X_ROCE_PEER_HCA_MAPS=<from the planner; depends on the cabling and on which node is reversed>
B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES=0
```

One of our nodes had its ring ports used in reversed roles (`--reversed`, [knapcio PR #12](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/12)). Our `B12X_ROCE_PEER_HCA_MAPS` value was
`1=1/3,2=0/3,3=0/2;0=1/3,2=0/2,3=0/3;0=1/2,1=0/2,3=1/3;0=1/3,1=0/3,2=0/2`. It is valid only for that wiring. Generate yours with the planner.

### The dual-PCI NCCL library

We did not build it. We used sparkring's prebuilt Route A files (sparkring release `native-runtime-sm121-aa8fa11831af`):

| item | value |
|---|---|
| archive | `native-runtime-files-20260908.tar`, sha256 `aa8fa11831afaa4539e0b74442fd1ea25dd8cf49ad5edb6fd95e7f05fdf5ce86` |
| library inside | `native/libnccl.so.2.30.7`, 63,651,200 bytes, sha256 `768a450b5eb84bf3d1191795350e43c96de75aeba4783ec314d47672fe6e1fc6` |
| feature markers found in it with `grep -a` | `SWITCHLESS_RING_ONLY`, `EXTENDED_IPV4_GIDS`, `PRESERVE_PCI_DOMAIN` |
| NCCL base revision | `73cf112295c33aee2b895f329f592f2a9b4b0f97` (NVIDIA/nccl) |
| patch | sparkring `runtime/sparkring/source_image/patches/nccl.patch` at `17dafa8b`, sha256 `8e2b8715d62d2b07a74caca3778da0eff2e7b54caf8a184bb728f179a5d1eba4` |

The archive's own `provenance.json` says the files were exported unchanged from an image that sparkring qualified for a different model, and that the evidence does not qualify other uses.
sparkring labels the dual-PCI-domain patch "research-only". We measured it on DeepSeek-V4.1-Flash-1017; that is our measurement, not sparkring's qualification.

### Host prerequisites (engine stopped, all four nodes)

1. The RoCEv2 GID at index 3 must be the IPv4 GID on all four fabric functions of every node. Our fix: `ipv6.method disabled` on the fabric NetworkManager connections, bring them up again, reload `mlx5_ib`
   ([knapcio PR #15](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/15)). Whether this survives a reboot was not tested by us as a separate step.
2. Install the mesh as described in `docs/switchless-ring.md`, "RoCEnante on the ring": marker binary, `hairpin.sh 8192`, the `dsv41-mesh` units. We start the units by hand and do not enable them.
3. The up and down scripts we adapted are `ring-up.sh` and `ring-down.sh` in [the switch-hop repository](https://github.com/adapt-ai-systems/gb10-hairpin-ring-switch-hop/tree/main/scripts). They are generalised and were not re-run in that form.

### Boot caveat

One boot of this config took 16 minutes: weight load 807 s, with all ranks waiting on rank 3 in the CPU-side `_dequant_fp8_wo_a_streaming`. The other boots of the mesh configs took 7 to 9 minutes. We do not know the cause.
The one node waited on had 16 GB of free memory during the load; we did not test whether that matters.
The KV pool also varied by boot: one dual-PCI-only boot gave 3,240,448 tokens, and the dual-PCI + hairpin boot above gave 4,327,424. We did not find the cause of the low boot.
