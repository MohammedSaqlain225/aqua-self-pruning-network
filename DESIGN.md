# DESIGN

## 1. Importance criterion: why it approximates "loss increase if removed"

Let `w_i` be one weight, `g_i = dL/dw_i`, `H_ii = d²L/dw_i²`. Removing it sets `w_i -> 0`, i.e. a step `delta = -w_i` along coordinate i. Second-order Taylor expansion:

    dL_i = L(w with w_i=0) - L(w)  ≈  g_i * (-w_i) + 1/2 * H_ii * w_i²  =  -g_i w_i + 1/2 H_ii w_i²

* **magnitude** `|w_i|` ignores both `g` and `H` - it equals dL only if all curvatures are equal and the first-order term vanishes. It also cannot be compared across layers, whose weights have different scales.
* **taylor** `|w_i * gbar_i|` keeps only the first-order term, with `gbar` an EMA of minibatch gradients. Weakness: near a minimum `E[g] ≈ 0`, so the term is dominated by minibatch noise and its sign is irrelevant to the true change.
* **fisher** `1/2 * F_i * w_i²` keeps the second-order term, using the diagonal empirical Fisher `F_i = EMA(g_i²)` as the Hessian proxy. For cross-entropy the Fisher equals the Gauss-Newton matrix, which is a good Hessian approximation near a minimum and is positive by construction. This is Optimal-Brain-Damage-style saliency with a Fisher diagonal. Because it is in units of *loss*, scores are comparable across layers, so a single global ranking is meaningful.

Caveats (honest): we use *minibatch* gradients, so `E[ĝ²] = F/B + (E g)²` - a constant factor that does not change rankings at fixed batch size. Diagonal ignores weight interactions (removing many weights at once is not additive).

**Empirical check** (`python -m prune.validate_criterion`, one trained dense net, 600 random single-weight deletions, Spearman rank correlation with the *measured* loss increase): fisher 0.65, taylor 0.55, magnitude 0.51. This supports the derivation ranking, but note it did **not** translate into a statistically significant accuracy gain in the sweep (see README).

**Schedule:** cubic ramp (Zhu & Gupta 2017), `s_t = s_f (1 - (1 - (t-t0)/(t1-t0))³)`. It prunes aggressively early, when the network is over-parameterised and redundancy is high, and slowly near the target when each removal hurts more; the final 30% of steps are fine-tuning with a frozen mask. One-shot pruning at the end is cheaper to implement but gives the network no chance to recover between removals.

**Regrowth:** at each update a fraction of the keep-budget (decaying linearly to 0 at `t1`, so the mask settles before fine-tuning) is swapped: prune to `K - r`, then revive the `r` previously-pruned slots with the largest `|EMA dL/dW_eff|` (first-order loss decrease available if revived at 0, RigL-style). Revived weights start at exactly 0 with clean optimizer state. Stability risk: a revived weight has score ≈ 0 and could be pruned again at the next update (churn); we measure cumulative churn (`results/part4_raw.jsonl`).

## 2. What is "the gradient of a masked weight"?

Forward: `W_eff = W ⊙ M`. Chain rule gives two exact quantities:

* `dL/dW = (dL/dW_eff) ⊙ M` - zero at pruned slots. This is `Tensor.grad`; it is what the optimizer sees. It is the true derivative of the loss w.r.t. the *stored* parameter (a pruned stored weight genuinely has no effect on the loss), and finite-difference-verified in `tests/test_masking.py`.
* `dL/dW_eff` - nonzero at pruned slots. Stored as `Tensor.dense_grad`; it answers "what would reviving this slot at 0 do", the correct regrowth signal. Also finite-difference-verified.

**Why zeroing the gradient is not enough** (the bug most implementations ship): with momentum/Adam, `m`, `v` are nonzero when a weight is pruned mid-training, so `-lr * m̂/(√v̂+ε)` keeps moving the raw weight ("ghost updates"); on revival the weight and its moments are stale; and Adam's single global step counter under-corrects bias for a slot with no history. Fix (`nn/optim.py`): on any mask change (pruned or revived) zero that slot's weight, `m`, `v` and per-slot `t`; each step multiply gradient and update by the mask; bias-correct with the per-slot `t`. A revived slot's first step has magnitude exactly `lr`, as for a fresh parameter (tested), whereas the naive implementation does not (tested as a negative control).

## 3. Where the engine bottlenecks, and how to fix it

* Python overhead per op: each op allocates a Tensor, a closure and temporaries; at these sizes (batch 64, 50k weights) interpretive overhead and memory allocation dominate over BLAS. Fix: operator fusion (e.g. a fused Linear+bias+ReLU node), preallocated buffers, in-place gradient accumulation, or compile the graph once and replay it.
* Masked op does `W*M` on the full dense matrix every forward/backward and keeps a dense `dense_grad`: O(in×out) regardless of sparsity. Fix: cache `W_eff` and only recompute after optimizer steps; compute `dense_grad` only on update steps (every 10 here), not every step.
* Pruner scoring concatenates all layers each update. Cheap at `every=10`; fix with per-layer `argpartition`/histogram thresholds.
* Gradient accumulation allocates (`t.grad + g`); in-place `+=` on owned buffers would remove copies.
* Float64 everywhere (needed for gradcheck); training in float32 would halve memory traffic.

## 4. Serving a self-pruned model in a multi-tenant service at scale

* **Don't serve zeros.** Measured here: the multiply-accumulate count falls 20x at 95% sparsity, yet my NumPy CSC kernel is slower than dense BLAS up to ~98% because unstructured gather is memory-bound and defeats SIMD/BLAS. Dense-times-zero gives no speedup at all. What did help in NumPy was *structural compaction*: dead neurons removed exactly, giving smaller dense matrices (4.5x faster at 99%, batch 64; only ~1.4x at 90%).
* Production path: (a) prefer **structured / N:M sparsity** (e.g. 2:4 on Ampere+ GPUs) or prune whole neurons/heads, which map to dense hardware kernels; unstructured sparsity needs a real sparse kernel (cuSPARSE/Triton/CPU AVX-512 sparse GEMM) and typically only pays above ~90%. (b) Export compacted dense weights; quantise (int8) on top.
* Multi-tenant concerns: dynamic batching across tenants (sparse models have smaller per-request compute, so batch to fill the device); per-tenant model versions behind a registry with canary rollout and an accuracy gate (sparse models can regress on tails - evaluate per-slice, not just average accuracy); weights are small after compaction, so many tenants' models can be co-resident in memory and hot-swapped; isolate noisy neighbours with per-tenant concurrency limits and queue-based admission control; track p50/p99 latency and cost per 1k requests as the real success metric, not FLOPs.
