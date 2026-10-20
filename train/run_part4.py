"""Part 4: sparsity sweep, baselines, seed variance, real-cost measurement, claim.

    python -m train.run_part4            # full sweep (resumes from results/part4_raw.jsonl)
    python -m train.run_part4 --fresh    # delete cache and recompute everything
    python -m train.run_part4 --analyze  # only re-do tables/plots from the cached raw numbers

Every run is deterministic given (method, sparsity, seed); the raw per-run
numbers are committed in results/part4_raw.jsonl.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prune.sparse_infer import cost_report
from train.data import load_data
from train.loop import Config, train

OUT = "results"
RAW = f"{OUT}/part4_raw.jsonl"
SPARSITIES = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99]
SEEDS = list(range(8))
METHODS = {                                       # name -> (criterion, regrow_frac)
    "magnitude": ("magnitude", 0.0),
    "taylor": ("taylor", 0.0),
    "fisher": ("fisher", 0.0),
    "magnitude+regrow": ("magnitude", 0.1),
    "fisher+regrow": ("fisher", 0.1),
}


def load_raw():
    if not os.path.exists(RAW):
        return {}
    rows = [json.loads(l) for l in open(RAW)]
    return {(r["method"], r["target"], r["seed"]): r for r in rows}


def sweep(data):
    done = load_raw()
    jobs = [("dense", 0.0, s) for s in SEEDS] + \
           [(m, sp, s) for sp in SPARSITIES for m in METHODS for s in SEEDS]
    with open(RAW, "a") as f:
        for n, (m, sp, s) in enumerate(jobs, 1):
            if (m, sp, s) in done:
                continue
            crit, rg = METHODS.get(m, ("magnitude", 0.0))
            cfg = Config(target_sparsity=sp, criterion=crit, regrow_frac=rg, seed=s)
            net, r = train(cfg, data)
            row = {"method": m, "target": sp, "seed": s, "achieved_sparsity": r["sparsity"],
                   "test_acc": r["test_acc"], "val_acc": r["val_acc"], "test_loss": r["test_loss"],
                   "churn": r["history"]["churn"][-1]}
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"[{n}/{len(jobs)}] {m:18s} sp={sp:.2f} seed={s} test={r['test_acc']:.4f} val={r['val_acc']:.4f}", flush=True)


def compute_cost(data):
    """Retrain the seed-0 dense and fisher+regrow models (deterministic) and measure real cost."""
    cost = {}
    for m, sp in [("dense", 0.0)] + [("fisher+regrow", sp) for sp in SPARSITIES]:
        crit, rg = METHODS.get(m, ("magnitude", 0.0))
        net, _ = train(Config(target_sparsity=sp, criterion=crit, regrow_frac=rg, seed=0), data)
        cost[(m, sp)] = cost_report(net, data[2][0])
        print(f"cost measured: {m} sp={sp}", flush=True)
    return cost


def paired_bootstrap(a, b, n=10000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)])
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())


def analyze(cost):
    rows = list(load_raw().values())
    get = lambda m, sp, key="test_acc": np.array([r[key] for r in sorted(rows, key=lambda r: r["seed"])
                                                  if r["method"] == m and r["target"] == sp])
    dense = get("dense", 0.0)
    lines = ["| target sparsity | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    lines.insert(0, f"Dense baseline (0% sparsity): test acc {100*dense.mean():.2f} +/- {100*dense.std(ddof=1):.2f} (mean +/- sd, n={len(dense)})\n")
    table = {}
    for sp in SPARSITIES:
        cells = []
        for m in METHODS:
            a = get(m, sp); table[(m, sp)] = a
            ach = np.mean(get(m, sp, "achieved_sparsity"))
            cells.append(f"{100*a.mean():.2f} +/- {100*a.std(ddof=1):.2f}")
        lines.append(f"| {sp:.2f} (achieved {ach:.4f}) | " + " | ".join(cells) + " |")
    md = [f"# Part 4 results (test accuracy in percent, mean +/- sd over {len(SEEDS)} seeds)\n"] + lines

    md += ["\n## Paired comparison vs magnitude (same seeds => same init & batch order)\n",
           "diff = acc(method) - acc(magnitude) in percentage points; 95% paired bootstrap CI; W/L/T = seeds where method is better/worse/tied\n",
           "| sparsity | method | mean diff (pp) | 95% CI | W/L/T | CI excludes 0? |", "|---|---|---|---|---|---|"]
    stats = {}
    for sp in SPARSITIES:
        for m in ["taylor", "fisher", "fisher+regrow", "magnitude+regrow"]:
            d, lo, hi, w, l, t = paired_bootstrap(table[(m, sp)], table[("magnitude", sp)])
            stats[(m, sp)] = (100*d, 100*lo, 100*hi, w, l, t)
            md.append(f"| {sp:.2f} | {m} | {100*d:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {w}/{l}/{t} | {'YES' if lo > 0 or hi < 0 else 'no'} |")

    # cost table -----------------------------------------------------------------
    if cost:
        json.dump({f"{m}@{sp}": v for (m, sp), v in cost.items()}, open(f"{OUT}/part4_cost.json", "w"), indent=1, default=str)
    if os.path.exists(f"{OUT}/part4_cost.json"):
        c = json.load(open(f"{OUT}/part4_cost.json"))
        md += ["\n## Real cost (seed 0, fisher+regrow models; MACs/sample; latency = best-of-5 mean, this machine, single thread)\n",
               "| model | MACs dense-times-zero | MACs nnz | MACs compacted | FLOP red. (nnz) | alive params | compacted shapes | lat B=1 dense / csc / compact (us) | lat B=64 (us) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for k, v in c.items():
            L = v["latency_seconds"]
            f_ = lambda B: " / ".join(f"{1e6*L[B][x]:.0f}" for x in ["dense_times_zero", "csc_sparse", "compacted_dense"])
            md.append(f"| {k} | {v['macs_dense_model']} | {v['macs_sparse_nnz']} | {v['macs_compacted_dense']} | {v['flop_reduction_nnz']:.1f}x | {v['params_alive']} | {v['compacted_shapes']} | {f_('1')} | {f_('64')} |")

    # falsifiable claim ----------------------------------------------------------
    best_sp = 0.99
    a_f, a_m = table[("fisher", best_sp)], table[("magnitude", best_sp)]
    d, lo, hi, w, l, t = paired_bootstrap(a_f, a_m)
    claim = (f"At {best_sp:.0%} sparsity (achieved {np.mean(get('fisher', best_sp, 'achieved_sparsity')):.4f}), Fisher-saliency pruning "
             f"(no regrowth) reaches {100*a_f.mean():.2f}% test accuracy vs {100*a_m.mean():.2f}% for magnitude pruning, "
             f"mean over {len(SEEDS)} paired seeds (diff {100*d:+.2f} pp, 95% paired-bootstrap CI [{100*lo:+.2f}, {100*hi:+.2f}], "
             f"fisher better/worse/tied on {w}/{l}/{t} seeds); dense reference {100*dense.mean():.2f}%.")
    md += ["\n## Falsifiable claim\n", claim]
    open(f"{OUT}/part4_summary.md", "w").write("\n".join(md) + "\n")
    open(f"{OUT}/claim.md", "w").write(claim + "\n")
    print("\n".join(md))

    # Pareto plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, zoom in zip(axes, [False, True]):
        xs0 = [0.0] + SPARSITIES
        for m in METHODS:
            mu = [100*dense.mean()] + [100*table[(m, sp)].mean() for sp in SPARSITIES]
            sd = [100*dense.std(ddof=1)] + [100*table[(m, sp)].std(ddof=1) for sp in SPARSITIES]
            ax.errorbar(xs0, mu, yerr=sd, marker="o", capsize=3, label=m, alpha=.85)
        ax.set_xlabel("achieved sparsity (fraction of weights removed)"); ax.set_ylabel("test accuracy (%)")
        ax.grid(alpha=.3)
        if zoom:
            ax.set_ylim(94, 100); ax.set_title("zoom (mean +/- sd, %d seeds)" % len(SEEDS))
        else:
            ax.set_title("Sparsity-accuracy Pareto curve"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/part4_pareto.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true"); ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.fresh and os.path.exists(RAW):
        os.remove(RAW)
    cost = {}
    if not a.analyze:
        data = load_data()
        sweep(data)
        cost = compute_cost(data)
    analyze(cost)
