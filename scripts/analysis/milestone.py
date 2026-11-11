"""Milestone numbers and figures (SchNet, ethanol, rMD17 950 frames, 100 ps Langevin at 500 K).

    python scripts/analysis/milestone.py fetch   # copy finished runs from the Modal Volume
    python scripts/analysis/milestone.py plot    # write docs/milestone/results.md and figures/

`plot` needs only numpy + matplotlib and the files under docs/milestone/data, so it runs
without Modal access. `fetch` needs the modal CLI; runs that are not finished are skipped.
"""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mlipstab.eval.stability import kaplan_meier  # noqa: E402

OUT = ROOT / "docs/milestone"
DATA = OUT / "data"
KCAL_TO_MEV = 43.364  # 1 kcal/mol = 43.364 meV

# Stage A learning-rate schedule: ReduceLROnPlateau cuts the LR after `patience` epochs without
# validation improvement; training stops once the LR is below 1e-6.
SHORT = "short schedule (StABlE default, patience 5)"   # stopped at epoch 422, underfit
LONG = "long schedule (converged, patience 50)"         # stopped at epoch 3502
# id -> (Stage A schedule, loss, margin weight, eval tag on the Volume, Stage B results dir)
# Two tags say "p5" but ran on the patience-50 checkpoint (launched after it replaced the link).
RUNS = {
    "p5_standard": (SHORT, "standard", None, "pretrained_schnet_ethanol_100ps", None),
    # control: the same Stage B fine-tuning with the margin term switched off
    "p5_control_lam0": (SHORT, "control", 0.0, "margin_p5v2_lam0_100ps", "margin_p5_lam0"),
    "p5_margin_lam0.0005": (SHORT, "margin", 0.0005, "margin_p5v2_lam0005_100ps", "margin_p5_lam0005"),
    "p5_margin_lam0.005": (SHORT, "margin", 0.005, "margin_p5v2_lam005_100ps", "margin_p5_lam005"),
    "p50_standard": (LONG, "standard", None, "pretrained_schnet_ethanol_p50_100ps", None),
    "p50_margin_lam0.0005": (LONG, "margin", 0.0005, "margin_p5_lam0005_100ps", "margin_r_lam0005"),
    "p50_margin_lam0.005": (LONG, "margin", 0.005, "margin_p50_lam005_g15_100ps", "margin_p50_lam005_g15"),
}
# Reference palette: categorical slots 1-2, fixed per loss (never per rank).
COLOR = {"standard": "#2a78d6", "margin": "#eb6834", "control": "#8a8984"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def fetch():
    for rid, (_, _, _, tag, stage_b) in RUNS.items():
        dest = DATA / rid
        with tempfile.TemporaryDirectory() as tmp:
            get = lambda remote, name: subprocess.run(
                ["modal", "volume", "get", "--force", "mlipstab", remote, f"{tmp}/{name}"],
                capture_output=True).returncode == 0
            if not get(f"results/eval/{tag}/metrics.json", "metrics.json"):
                print(f"skip {rid}: not finished")
                continue
            get(f"results/eval/{tag}/stability.npz", "stability.npz")
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "metrics.json").write_text(Path(tmp, "metrics.json").read_text())
            s = np.load(f"{tmp}/stability.npz")
            with open(dest / "stability.csv", "w", newline="") as f:  # *.npz is gitignored
                w = csv.writer(f)
                w.writerow(["time_ps", "failed", "time_bond_only_ps", "failed_bond_only"])
                w.writerows(zip(s["times_ps"], s["observed"].astype(int),
                                s["times_bond_only_ps"], s["observed_bond_only"].astype(int)))
            if stage_b and get(f"results/{stage_b}/ethanol_schnet_seed0/history.json", "history.json"):
                (dest / "history.json").write_text(Path(tmp, "history.json").read_text())
        print(f"fetched {rid}")


def load():
    runs = {}
    for rid, (sched, loss, lam, _, _) in RUNS.items():
        d = DATA / rid
        if not (d / "metrics.json").exists():
            continue
        st = np.genfromtxt(d / "stability.csv", delimiter=",", names=True)
        hist = json.loads((d / "history.json").read_text())["history"] if (d / "history.json").exists() else None
        runs[rid] = dict(sched=sched, loss=loss, lam=lam, m=json.loads((d / "metrics.json").read_text()),
                         times=st["time_ps"], failed=st["failed"].astype(bool), hist=hist)
    return runs


def label(r):
    if r["loss"] == "margin":
        return f"Margin loss (λ = {r['lam']:g})"
    return {"standard": "Standard loss", "control": "Fine-tuning control (λ = 0)"}[r["loss"]]


def table(runs):
    rows = ["| Stage A schedule | Loss | Force MAE (meV/Å) | Energy MAE (meV) | Stable at 100 ps | Failed / 256 "
            "| Mean τ (ps, capped at 100) | h(r) MAE | Dihedral F MAE (kcal/mol) | NVE drift (meV/atom/ps) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs.values():
        m, c = r["m"], r["m"]["combined"]
        e = m["energy_mae_kcal_mol"] * KCAL_TO_MEV
        e = f"{e:.1f}" if e < 1e4 else "n/a¹"
        rows.append(f"| {r['sched']} | {label(r)} | {m['force_mae_kcal_mol_A'] * KCAL_TO_MEV:.1f} | {e} "
                    f"| {100 * c['frac_stable_100ps']:.1f}% | {int(r['failed'].sum())} | {c['mean_tau_ps']:.1f} "
                    f"| {m['hr_mae']:.3f} | {m['dihedral_F_mae_kcal_mol']:.2f} | {m['nve_drift_meV_atom_ps']:.3f} |")
    return "\n".join(rows)


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    plt.rcParams.update({"font.size": 9, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "savefig.dpi": 200, "legend.frameon": False})
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    runs = load()
    scheds = [SHORT, LONG]
    dash = {None: "-", 0.0: ":", 0.0005: "--", 0.005: "-"}

    # 1. Stage A loss curves (parsed from the pretraining logs; one schedule per line)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, color in (("p50", COLOR["standard"]), ("p5", "#8a8984")):  # p5 drawn last: same seed, same early curve
        d = np.genfromtxt(DATA / f"pretraining_{name}.csv", delimiter=",", names=True)
        lab = SHORT if name == "p5" else LONG
        axes[0].plot(d["epoch"], d["val_force_mae_meV_A"], color=color, lw=2, label=lab)
        axes[0].plot(d["epoch"][-1], d["val_force_mae_meV_A"][-1], "o", color=color, ms=6)
        axes[0].annotate(f"{d['val_force_mae_meV_A'][-1]:.0f} meV/Å at epoch {int(d['epoch'][-1])}",
                         (d["epoch"][-1], d["val_force_mae_meV_A"][-1]), textcoords="offset points",
                         xytext=(-10, -16) if name == "p5" else (-8, 10), ha="right", color=INK, fontsize=8)
        lr = d[~np.isnan(d["lr"])]
        axes[1].plot(lr["epoch"], lr["lr"], color=color, lw=2, label=lab)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xscale("log")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_yticks([20, 50, 100, 200, 500])
    style(axes[0], "epoch", "validation force MAE (meV/Å)", "Stage A (standard loss): validation force error")
    style(axes[1], "epoch", "learning rate", "Learning rate (training stops below 1e-6)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figs / "fig1_pretraining_loss.png")

    # 2. Kaplan-Meier survival, standard vs margin loss, one panel per Stage A model
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, sched in zip(axes, scheds):
        for r in (r for r in runs.values() if r["sched"] == sched):
            t, s = kaplan_meier(r["times"], r["failed"])
            ax.step(np.r_[0, t, 100], np.r_[1, s, s[-1] if len(s) else 1], where="post", lw=2,
                    color=COLOR[r["loss"]], ls=dash[r["lam"]],
                    label=f"{label(r)}: {100 * r['m']['combined']['frac_stable_100ps']:.1f}% at 100 ps")
        style(ax, "time (ps)", "fraction of replicas without failure", f"Stage A: {sched}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0.6, 1.005)
        ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figs / "fig2_survival.png")

    # 3. Stage B per cycle: failures found in the training rollouts, and validation force error
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for r in (r for r in runs.values() if r["hist"]):
        h = r["hist"]
        kw = dict(color=COLOR[r["loss"]] if r["sched"] == SHORT else "#b9b8b2", ls=dash[r["lam"]], lw=2,
                  marker="o", ms=5, label=f"{r['sched'].split(' (')[0]}, λ = {r['lam']:g}")
        axes[0].plot([c["cycle"] for c in h], [100 * c["fail_frac"] for c in h], **kw)
        v = [c for c in h if "val_force_mae" in c]
        axes[1].plot([c["cycle"] for c in v], [1000 * c["val_force_mae"] for c in v], **kw)
    style(axes[0], "Stage B cycle", "replicas failing in a 20 ps rollout (%)",
          "Failures found before each cycle")
    style(axes[1], "Stage B cycle", "validation force MAE (meV/Å)", "Force error after each cycle")
    for ax in axes:
        ax.set_xticks([0, 1, 2])
        ax.set_ylim(bottom=0)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figs / "fig3_stage_b_cycles.png")

    # 4. Standard vs margin loss: failures within 100 ps, and force error
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    names = [f"{r['sched'].split()[0]}\n{'standard' if r['loss'] == 'standard' else 'λ = %g' % r['lam']}"
             for r in runs.values()]
    x = np.arange(len(runs))
    colors = [COLOR[r["loss"]] for r in runs.values()]
    for ax, vals, fmt in ((axes[0], [100 * (1 - r["m"]["combined"]["frac_stable_100ps"]) for r in runs.values()], "{:.1f}%"),
                          (axes[1], [r["m"]["force_mae_kcal_mol_A"] * KCAL_TO_MEV for r in runs.values()], "{:.1f}")):
        ax.bar(x, vals, width=0.6, color=colors)
        for xi, v in zip(x, vals):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=8)
        ax.set_xticks(x, names, fontsize=8)
    style(axes[0], "Stage A schedule / margin weight λ", "replicas failed within 100 ps (%)", "Instability (256 replicas, 500 K, 0.5 Å)")
    style(axes[1], "Stage A schedule / margin weight λ", "test force MAE (meV/Å)", "Accuracy (1,000 rMD17 test frames)")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(figs / "fig4_standard_vs_margin.png")

    (OUT / "results.md").write_text(
        "# Milestone results: SchNet on ethanol\n\n"
        "Generated by `python scripts/analysis/milestone.py plot` from `docs/milestone/data/`. "
        "Notes on setup, caveats and the Stage B settings sweep are in [README.md](README.md).\n\n"
        + table(runs) +
        "\n\n¹ Evaluated before the energy-offset fix in `mdsim_backend.py`; the energy MAE of these two runs "
        "is not meaningful. Force MAE and stability are unaffected.\n\n"
        "![Stage A loss](figures/fig1_pretraining_loss.png)\n\n![Survival](figures/fig2_survival.png)\n\n"
        "![Stage B per cycle](figures/fig3_stage_b_cycles.png)\n\n"
        "![Standard vs margin](figures/fig4_standard_vs_margin.png)\n")
    print(table(runs))


if __name__ == "__main__":
    {"fetch": fetch, "plot": plot}[sys.argv[1]]()
