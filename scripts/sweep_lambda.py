"""Select the margin-loss weight lambda on held-out tune frames, then evaluate it once on test.

A system is one (molecule, model) pair with its own Stage A checkpoint. Datasets, systems, the
grid and the selection rule are all set in the sweep config (configs/sweep/lambda.yml).
Use pretrained_dir with {seed} to pair each seed's independent Stage A checkpoint across
the control, margin candidates and baseline. The supplied configs use fixed Stage B budgets.

Protocol
  1. Stage B for every (system, lambda, seed); lambda = 0 is the fine-tuning control.
  2. Evaluate each run from `tune` frames (outside train/val/test); force MAE on tune frames.
  3. Select, per system or one shared lambda:
       budget:    force MAE <= (1 + acc_tol) * that of lambda = 0 (on every system if shared),
                  with no accuracy-guardrail aborts; an aborted control blocks selection
       best:      in-budget lambda with the lowest failure fraction (mean over systems if shared)
       selected:  smallest in-budget lambda within one standard error of the best
     SE = max(binomial over replicas, between-seed), so training noise counts.
  4. `final`: selected lambda, control and Stage A, on the official test frames, once.
     final/lock.json records the selection; a different one is refused without --force.

Usage
    python scripts/sweep_lambda.py all   --config configs/sweep/lambda.yml   # tune + select
    python scripts/sweep_lambda.py final --config configs/sweep/lambda.yml
    modal run --detach cloud/modal/app.py::sweep --config ... --phase tune|final   # parallel
Steps skip finished work, so any command can be rerun after a failure.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlipstab.config import _merge, load_config  # noqa: E402

CRITERIA = {"combined": "observed", "bond_only": "observed_bond_only"}  # keys in stability.npz
EVAL_DIR = "results/eval"  # where scripts/evaluate.py writes <tag>/


# ------------------------------------------------------------------ config
def load_sweep(path):
    """Read a sweep config; resolve each system as datasets[<dataset>] <- defaults <- system entry."""
    path = Path(path).resolve()
    cfg = load_config(path)
    defaults = cfg.get("defaults") or {}
    systems = {}
    for entry in cfg["systems"]:
        dataset = entry.get("dataset", defaults.get("dataset"))
        if dataset not in cfg["datasets"]:
            raise ValueError(f"system {entry}: dataset {dataset!r} is not in `datasets`")
        s = _merge(_merge(cfg["datasets"][dataset], defaults), entry)
        s["dataset"] = dataset
        s["id"] = s.get("id") or f"{s['molecule']}_{s['model']}"
        for section in ("train", "select_eval", "final_eval"):
            s[section]["config"] = str(ROOT / s[section]["config"])
        if s["id"] in systems:
            raise ValueError(f"duplicate system id {s['id']}; set `id` on one of them")
        systems[s["id"]] = s
    cfg["systems"] = systems
    cfg["_path"] = str(path)
    if 0 not in [float(x) for x in cfg["grid"]["lam"]]:
        raise ValueError("grid.lam must include 0, the fine-tuning control")
    if cfg["select"]["mode"] not in ("shared", "per_system"):
        raise ValueError("select.mode must be shared or per_system")
    if cfg["select"]["criterion"] not in CRITERIA:
        raise ValueError(f"select.criterion must be one of {sorted(CRITERIA)}")
    return cfg


def out_dir(cfg):
    return ROOT / cfg.get("results_dir", "results/sweeps") / cfg["name"]


def lam_tag(lam):
    return "stageA" if lam is None else f"lam{float(lam):g}"


def ckpt_path(cfg, s, lam, seed):
    # train_stability.py writes <results_dir>/<molecule>_<model>_seed<seed>/ckpt.pt
    return out_dir(cfg) / s["id"] / "train" / lam_tag(lam) / f"{s['molecule']}_{s['model']}_seed{seed}" / "ckpt.pt"


def eval_tag(cfg, s, stage, lam, seed):
    return f"{out_dir(cfg).relative_to(ROOT / 'results')}/{s['id']}/{stage}/{lam_tag(lam)}_seed{seed}"


def system_overrides(s, seed):
    """Merged into every generated config for this system: its data location and Stage A model."""
    out = {"data": s["data"]}
    if s.get("pretrained_dir"):
        out["pretrained"] = {"dir": s["pretrained_dir"].format(seed=seed)}
        if "{seed}" in s["pretrained_dir"]:
            out["pretrained"]["seed"] = seed
    return out


def write_yaml(path, include, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"includes": [include], **body}, sort_keys=False))
    return path


def sh(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


# ------------------------------------------------------------------ tune split
def split_dir(s):
    d = s["data"]
    return ROOT / d["src"] / d["name"] / s["molecule"] / d["size"]


def load_raw(backfill, molecule):
    """(z, R, E, F) in eV and eV/Å from the raw file named by a `tune_split.backfill` spec."""
    from mlipstab.units import EV_TO_KCAL_MOL

    path = ROOT / backfill["path"].format(molecule=molecule)
    if not path.exists() and "url" in backfill:
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(backfill["url"].format(molecule=molecule), path)
    raw = np.load(path)
    k = backfill["keys"]
    scale = {"kcal/mol": 1 / EV_TO_KCAL_MOL, "eV": 1.0}[backfill["unit"]]
    return (raw[k["z"]].astype(np.int64), raw[k["R"]],
            raw[k["E"]].reshape(-1) * scale, raw[k["F"]] * scale)


def prepare_tune(s, seed=0):
    """Ensure <split dir>/tune exists. Prep scripts write it; `backfill` builds it for older data."""
    from mlipstab.data.tune import tune_pool, used_by_positions, write_tune

    base, spec = split_dir(s), s["tune_split"]
    if (base / "tune" / "nequip_npz.npz").exists():
        print(f"{base / 'tune'} exists")
        return
    if "backfill" not in spec:
        raise FileNotFoundError(f"{base / 'tune'} is missing; rerun the dataset's prep script "
                                "or set tune_split.backfill for this dataset")
    z, R, E, F = load_raw(spec["backfill"], s["molecule"])
    if (base / "splits.npy").exists():
        splits = np.load(base / "splits.npy", allow_pickle=True).item()
        used = np.concatenate([splits["train"], splits["val"], splits["test"]])
    else:  # splits without raw indices (Fu et al.'s archive): find the used frames by coordinates
        splits = {}
        used = used_by_positions(R, [np.load(base / sp / "nequip_npz.npz")["R"] for sp in ("train", "val", "test")])
    idx = tune_pool(len(R), used, spec["n_frames"], seed=seed)
    write_tune(base, idx, z, R, E, F)
    np.save(base / "splits", {**splits, "tune": idx})
    print(f"wrote {base / 'tune'} ({len(idx)} frames)")


# ------------------------------------------------------------------ jobs
def jobs(cfg, stage, force=False):
    """`run-one` argument lists: every (system, lambda, seed) for tune; locked selection for final."""
    def job(sid, lam, seed):
        return ["run-one", "--config", cfg["_path"], "--system", sid, "--stage", stage,
                "--lam", lam, "--seed", str(seed)]

    if stage == "tune":
        return [job(sid, f"{float(l):g}", seed) for sid in cfg["systems"]
                for l in cfg["grid"]["lam"] for seed in cfg["grid"]["seeds"]]
    return [job(sid, v, seed) for sid, lam in lock_final(cfg, force).items()
            for v in ("stageA", "0", f"{lam:g}") for seed in cfg["final"]["seeds"]]


def run_one(cfg, s, stage, lam, seed):
    """Stage B for (system, lam, seed) if its checkpoint is missing, then evaluate. lam=None: Stage A."""
    ckpt, cfg_dir = None, out_dir(cfg) / s["id"] / "configs"
    if lam is not None:
        ckpt = ckpt_path(cfg, s, lam, seed)
        if not ckpt.exists():
            body = _merge(s["train"].get("overrides") or {}, {
                **system_overrides(s, seed), "stage_b": {"lam": float(lam)}, "seed": seed,
                "molecule": s["molecule"], "model": s["model"],
                "results_dir": str(ckpt.parent.parent.relative_to(ROOT))})
            tcfg = write_yaml(cfg_dir / f"train_{lam_tag(lam)}_seed{seed}.yml", s["train"]["config"], body)
            sh(sys.executable, "scripts/train_stability.py", "--config", tcfg)

    tag = eval_tag(cfg, s, stage, lam, seed)
    if (ROOT / EVAL_DIR / tag / "metrics.json").exists():
        print(f"{tag} already evaluated")
        return
    section = s["select_eval"] if stage == "tune" else s["final_eval"]
    body = _merge(section.get("overrides") or {}, system_overrides(s, seed))
    ecfg = write_yaml(cfg_dir / f"eval_{stage}_{lam_tag(lam)}_seed{seed}.yml", section["config"], body)
    cmd = [sys.executable, "scripts/evaluate.py", "--config", ecfg, "--molecule", s["molecule"],
           "--model", s["model"], "--tag", tag, "--seed", seed]
    sh(*cmd, *(["--checkpoint", ckpt] if ckpt else []))


# ------------------------------------------------------------------ statistics
def load_run(cfg, s, stage, lam, seed, criterion):
    """Failure counts and accuracy of one evaluation, or None if it has not finished."""
    d = ROOT / EVAL_DIR / eval_tag(cfg, s, stage, lam, seed)
    if not (d / "metrics.json").exists():
        return None
    st = np.load(d / "stability.npz")
    failed = st[CRITERIA[criterion]]
    m = json.loads((d / "metrics.json").read_text())
    run = {"n": int(failed.size), "failed": int(failed.sum()), "failed_bond_only": int(st["observed_bond_only"].sum()),
           "force_mae": m["force_mae_kcal_mol_A"], "energy_mae": m["energy_mae_kcal_mol"]}
    hist = ckpt_path(cfg, s, lam, seed).parent / "history.json" if lam is not None else None
    if hist is not None and hist.exists():
        run["guardrail_tripped"] = any(r.get("guardrail_tripped") for r in json.loads(hist.read_text())["history"])
    return run


def aggregate(runs):
    """Pool seeds. SE = max(binomial over all replicas, between-seed SE of per-seed fractions)."""
    n, k = sum(r["n"] for r in runs), sum(r["failed"] for r in runs)
    p = k / n
    se = math.sqrt(max(p * (1 - p), 1 / n) / n)  # floor: p = 0 still has nonzero error
    per_seed = [r["failed"] / r["n"] for r in runs]
    if len(per_seed) > 1:
        se = max(se, float(np.std(per_seed, ddof=1) / math.sqrt(len(per_seed))))
    return {"n": n, "failed": k, "frac": p, "se": se, "per_seed": per_seed,
            "force_mae": float(np.mean([r["force_mae"] for r in runs])),
            "energy_mae": float(np.mean([r["energy_mae"] for r in runs])),
            "failed_bond_only": sum(r["failed_bond_only"] for r in runs),
            "guardrail_tripped": any(r.get("guardrail_tripped") for r in runs)}


def in_budget(stats, acc_tol):
    if stats[0.0].get("guardrail_tripped"):
        raise ValueError("lambda=0 control hit the accuracy guardrail; revise the fine-tuning settings before selection")
    ref = stats[0.0]["force_mae"]
    return {l for l, s in stats.items() if not s.get("guardrail_tripped")
            and s["force_mae"] <= (1 + acc_tol) * ref}


def one_se(score, se, feasible):
    """(selected, best, bar): smallest feasible lambda with score <= best score + its SE."""
    best = min(feasible, key=lambda l: (score[l], l))
    bar = score[best] + se[best] + 1e-12  # absorbs float round-off at the boundary
    return min(l for l in feasible if score[l] <= bar), best, bar


def select_lambda(stats, acc_tol):
    """One system; stats maps lambda -> aggregate(). Returns (selected, best, feasible, bar)."""
    feasible = in_budget(stats, acc_tol)
    selected, best, bar = one_se({l: s["frac"] for l, s in stats.items()},
                                 {l: s["se"] for l, s in stats.items()}, feasible)
    return selected, best, sorted(feasible), bar


def select_shared(by_system, acc_tol):
    """One lambda for all systems: in budget on each, scored by the unweighted mean failure fraction."""
    lams = set.intersection(*(set(st) for st in by_system.values()))
    feasible = set.intersection(*(in_budget({l: st[l] for l in lams}, acc_tol) for st in by_system.values()))
    k = len(by_system)
    score = {l: sum(st[l]["frac"] for st in by_system.values()) / k for l in lams}
    se = {l: math.sqrt(sum(st[l]["se"] ** 2 for st in by_system.values())) / k for l in lams}
    selected, best, bar = one_se(score, se, feasible)
    return selected, best, sorted(feasible), bar, score, se


def z_test(k1, n1, k2, n2):
    """Two-sided two-proportion z-test p-value (replicas treated as independent)."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 1.0 if se == 0 else math.erfc(abs(k1 / n1 - k2 / n2) / se / math.sqrt(2))


def collect(cfg, s, stage, lams, seeds, criterion):
    runs, missing = {}, []
    for l in lams:
        for seed in seeds:
            r = load_run(cfg, s, stage, l, seed, criterion)
            if r is None:
                missing.append(eval_tag(cfg, s, stage, l, seed))
            else:
                runs.setdefault(l, []).append(r)
    return runs, missing


# ------------------------------------------------------------------ select
def table(stats, feasible, best, selected):
    lines = ["| λ | failed / replicas | failure frac ± SE | bond-only failed | force MAE (kcal/mol/Å) | p vs λ=0 | note |",
             "|---|---|---|---|---|---|---|"]
    c = stats[0.0]
    for l, s in stats.items():
        note = [t for t, on in (("ineligible", l not in feasible), ("lowest failures", l == best),
                                ("**selected**", l == selected), ("guardrail tripped", s["guardrail_tripped"])) if on]
        p = z_test(s["failed"], s["n"], c["failed"], c["n"])
        lines.append(f"| {l:g} | {s['failed']} / {s['n']} | {s['frac']:.3f} ± {s['se']:.3f} | {s['failed_bond_only']} | "
                     f"{s['force_mae']:.3f} | {p:.2g} | {', '.join(note)} |")
    return lines


def select(cfg, allow_missing=False):
    """Apply the selection rule; write selection.{json,md} and sweep.png under out_dir."""
    sel = cfg["select"]
    lams = [float(l) for l in cfg["grid"]["lam"]]
    by_system, missing = {}, []
    for sid, s in cfg["systems"].items():
        runs, miss = collect(cfg, s, "tune", lams, cfg["grid"]["seeds"], sel["criterion"])
        missing += miss
        if 0.0 not in runs:
            raise SystemExit(f"{sid}: the λ = 0 control has no finished runs")
        by_system[sid] = {l: aggregate(r) for l, r in sorted(runs.items())}
    if missing and not allow_missing:
        raise SystemExit("missing runs (--allow-missing to select anyway):\n  " + "\n  ".join(missing))

    lines = [f"# λ sweep: {cfg['name']} ({sel['mode']}, {sel['criterion']} failures, tune frames)", ""]
    out = {"mode": sel["mode"], "criterion": sel["criterion"], "acc_tol": sel["acc_tol"], "missing": missing,
           "systems": {}, "selected": {}}
    for sid, stats in by_system.items():
        selected, best, feasible, bar = select_lambda(stats, sel["acc_tol"])
        out["systems"][sid] = {"selected_alone": selected, "best": best, "feasible": feasible, "frac_bar": bar,
                               "stats": {f"{l:g}": v for l, v in stats.items()}}
        lines += [f"## {sid}", "", *table(stats, feasible, best, selected), "",
                  f"Selected alone: λ = {selected:g} (one-SE bar: failure frac ≤ {bar:.3f}).", ""]
        if sel["mode"] == "per_system":
            out["selected"][sid] = selected
    if sel["mode"] == "shared":
        selected, best, feasible, bar, score, se = select_shared(by_system, sel["acc_tol"])
        out["shared"] = {"selected": selected, "best": best, "feasible": feasible, "frac_bar": bar,
                         "score": {f"{l:g}": v for l, v in score.items()}, "se": {f"{l:g}": v for l, v in se.items()}}
        out["selected"] = {sid: selected for sid in by_system}
        lines += ["## Shared λ (mean failure fraction over systems)", "",
                  "| λ | mean frac ± SE | in budget on all systems |", "|---|---|---|"]
        lines += [f"| {l:g} | {score[l]:.3f} ± {se[l]:.3f} | {'yes' if l in feasible else 'no'} |" for l in sorted(score)]
        lines += ["", f"Selected shared λ = {selected:g} (one-SE bar: mean frac ≤ {bar:.3f})."]
    if all(v == 0 for v in out["selected"].values()):
        lines.append("\nNo λ > 0 beat the control by more than one SE; do not run `final`.")
    if missing:
        lines.append(f"\nMissing runs: {len(missing)}.")

    d = out_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    (d / "selection.json").write_text(json.dumps(out, indent=1))
    (d / "selection.md").write_text("\n".join(lines) + "\n")
    plot(by_system, out["selected"], d / "sweep.png")
    print("\n".join(lines))
    return out


def plot(by_system, selected, path):
    """Failure fraction and relative force MAE vs lambda, one line per system."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    for sid, stats in by_system.items():
        lams = list(stats)
        line = a.errorbar(lams, [stats[l]["frac"] for l in lams], [stats[l]["se"] for l in lams],
                          marker="o", capsize=3, label=sid)[0]
        b.plot(lams, [stats[l]["force_mae"] / stats[0.0]["force_mae"] for l in lams], marker="o", color=line.get_color())
        a.axvline(selected[sid], color=line.get_color(), ls="--", alpha=0.5)
    positive = [l for st in by_system.values() for l in st if l > 0]
    for ax, label in ((a, "failure fraction (tune)"), (b, "force MAE / λ=0 control (tune)")):
        if positive:
            ax.set_xscale("symlog", linthresh=min(positive))  # linear through 0, log above
        ax.set_xlabel("λ")
        ax.set_ylabel(label)
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ------------------------------------------------------------------ final
def lock_final(cfg, force=False):
    """Record {system: lambda} for the single test-set evaluation; refuse to change it later."""
    sel_file = out_dir(cfg) / "selection.json"
    if not sel_file.exists():
        raise SystemExit("run `select` first")
    sel = json.loads(sel_file.read_text())
    if sel["missing"] and not force:
        raise SystemExit("selection was made with missing runs; finish them or pass --force")
    chosen = {sid: float(l) for sid, l in sel["selected"].items() if float(l) > 0 or force}
    skipped = sorted(set(sel["selected"]) - set(chosen))
    if skipped:
        print(f"skipping {skipped}: λ = 0 selected; --force to evaluate anyway", file=sys.stderr)
    if not chosen:
        raise SystemExit("no system selected λ > 0; not spending the test set")
    lock = out_dir(cfg) / "final" / "lock.json"
    if lock.exists():
        old = json.loads(lock.read_text())
        if old != chosen and not force:
            raise SystemExit(f"test set already used with {old}; refusing {chosen} (--force to override)")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(chosen))
    return chosen


def summarize(cfg):
    """Test-set table per system: Stage A, control and selected lambda, with z-tests."""
    criterion = cfg["select"]["criterion"]
    lock = json.loads((out_dir(cfg) / "final" / "lock.json").read_text())
    lines, everything = [f"# Final: {cfg['name']} on the official test frames ({criterion} failures)", ""], {}
    for sid, lam in lock.items():
        s = cfg["systems"][sid]
        rows = {"Stage A (standard)": None, "Control (λ = 0)": 0.0, f"Margin (λ = {lam:g})": lam}
        stats = {}
        for name, l in rows.items():
            runs, missing = collect(cfg, s, "final", [l], cfg["final"]["seeds"], criterion)
            if missing:
                raise SystemExit("missing final runs:\n  " + "\n  ".join(missing))
            stats[name] = aggregate(runs[l])
        a, c = stats["Stage A (standard)"], stats["Control (λ = 0)"]
        lines += [f"## {sid}", "",
                  "| model | failed / replicas | bond-only failed | frac ± SE | force MAE | p vs Stage A | p vs control |",
                  "|---|---|---|---|---|---|---|"]
        for name, st in stats.items():
            lines.append(f"| {name} | {st['failed']} / {st['n']} | {st['failed_bond_only']} | {st['frac']:.3f} ± {st['se']:.3f} | "
                         f"{st['force_mae']:.3f} | {z_test(st['failed'], st['n'], a['failed'], a['n']):.2g} | "
                         f"{z_test(st['failed'], st['n'], c['failed'], c['n']):.2g} |")
        lines.append("")
        everything[sid] = stats
    (out_dir(cfg) / "final" / "summary.md").write_text("\n".join(lines))
    (out_dir(cfg) / "final" / "summary.json").write_text(json.dumps(everything, indent=1))
    print("\n".join(lines))


# ------------------------------------------------------------------ CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare-tune", "plan", "run-one", "select", "final", "summarize", "all"])
    p.add_argument("--config", required=True)
    p.add_argument("--system", help="system id (run-one)")
    p.add_argument("--stage", choices=["tune", "final"], default="tune")
    p.add_argument("--lam", help="lambda value, or `stageA` for the Stage A model (run-one)")
    p.add_argument("--seed", type=int)
    p.add_argument("--allow-missing", action="store_true", help="select with some runs unfinished")
    p.add_argument("--force", action="store_true", help="override the final-stage safeguards")
    a = p.parse_args()
    cfg = load_sweep(a.config)
    systems = cfg["systems"]

    if a.command == "prepare-tune":
        for s in systems.values():
            prepare_tune(s)
    elif a.command == "plan":
        print(json.dumps(jobs(cfg, a.stage, a.force)))
    elif a.command == "run-one":
        run_one(cfg, systems[a.system], a.stage, None if a.lam == "stageA" else float(a.lam), a.seed)
    elif a.command == "select":
        select(cfg, a.allow_missing)
    elif a.command == "summarize":
        summarize(cfg)
    elif a.command == "all":
        for s in systems.values():
            prepare_tune(s)
        for s in systems.values():
            for l in cfg["grid"]["lam"]:
                for seed in cfg["grid"]["seeds"]:
                    run_one(cfg, s, "tune", float(l), seed)
        select(cfg, a.allow_missing)
    elif a.command == "final":
        for sid, lam in lock_final(cfg, a.force).items():
            for l in (None, 0.0, lam):
                for seed in cfg["final"]["seeds"]:
                    run_one(cfg, systems[sid], "final", l, seed)
        summarize(cfg)


if __name__ == "__main__":
    main()
