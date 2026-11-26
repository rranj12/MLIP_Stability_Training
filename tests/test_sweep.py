"""Selection logic of scripts/sweep_lambda.py and the tune split (no torch, no GPU)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mlipstab.data.tune import tune_pool

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sweep/lambda.yml"
spec = importlib.util.spec_from_file_location("sweep_lambda", ROOT / "scripts/sweep_lambda.py")
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)


def stat(frac, mae, se=0.01):
    return {"frac": frac, "se": se, "force_mae": mae}


def write_config(tmp_path, **changes):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg.update(changes)
    p = tmp_path / "sweep.yml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_tune_pool_avoids_official_frames():
    used = np.arange(0, 2000, 2)
    idx = tune_pool(3000, used, 500)
    assert len(set(idx)) == 500 and not set(idx) & set(used.tolist())


def test_one_se_rule_prefers_smallest_indistinguishable_lambda():
    stats = {0.0: stat(0.20, 1.00), 1e-3: stat(0.055, 1.01), 3e-3: stat(0.050, 1.02), 1e-2: stat(0.01, 1.20)}
    selected, best, feasible, _ = sw.select_lambda(stats, acc_tol=0.05)
    assert 1e-2 not in feasible  # 20% worse forces: over budget even though it fails least
    assert best == 3e-3
    assert selected == 1e-3  # within one SE of the best, and weaker


def test_selects_control_when_nothing_beats_it():
    stats = {0.0: stat(0.10, 1.0), 1e-3: stat(0.09, 1.0, se=0.02), 1e-2: stat(0.30, 1.0)}
    assert sw.select_lambda(stats, 0.05)[0] == 0.0


def test_guardrail_aborts_cannot_win_selection():
    stats = {0.0: stat(0.2, 1.0), 1.0: {**stat(0.0, 1.0), "guardrail_tripped": True}}
    assert sw.select_lambda(stats, 0.05)[0] == 0.0
    stats[0.0]["guardrail_tripped"] = True
    with pytest.raises(ValueError, match="control hit the accuracy guardrail"):
        sw.select_lambda(stats, 0.05)


def test_gemnet_campaign_has_separate_selections_and_fixed_budgets():
    cfg = sw.load_sweep(ROOT / "configs/sweep/gemnet_t.yml")
    assert set(cfg["systems"]) == {"naphthalene_gemnet_t", "aspirin_gemnet_t"}
    assert cfg["select"]["mode"] == "per_system"
    assert cfg["select"]["criterion"] == "bond_only"
    assert len(sw.jobs(cfg, "tune")) == 28
    for s in cfg["systems"].values():
        train = sw._merge(sw.load_config(s["train"]["config"]), s["train"]["overrides"])
        assert train["stage_b"]["fixed_budget"]
        assert train["stage_b"]["n_cycles"] * train["stage_b"]["steps_per_cycle"] == 1500
        assert train["optim"]["name"] == "AdamW"


def test_each_seed_uses_one_stage_a_for_control_margin_and_baseline(tmp_path, monkeypatch):
    cfg = sw.load_sweep(ROOT / "configs/sweep/gemnet_t.yml")
    monkeypatch.setattr(sw, "ROOT", tmp_path)
    commands = []
    monkeypatch.setattr(sw, "sh", lambda *args: commands.append(args))
    s = cfg["systems"]["aspirin_gemnet_t"]
    for seed in (0, 1, 2):
        expected = {"dir": f"results/checkpoints/qm_pretrained/seed{seed}", "seed": seed}
        for stage in ("tune", "final"):
            for lam in (None, 0.0, 1.0):
                commands.clear()
                sw.run_one(cfg, s, stage, lam, seed)
                assert len(commands) == (1 if lam is None else 2)
                for cmd in commands:
                    body = yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text())
                    assert body["pretrained"] == expected
                    if cmd[1] == "scripts/train_stability.py":
                        assert body["seed"] == seed
                        assert body["stage_b"]["fixed_budget"]
                    else:
                        assert cmd[cmd.index("--seed") + 1] == seed


def test_shared_lambda_must_fit_every_system_budget():
    a = {0.0: stat(0.30, 1.0), 1e-3: stat(0.20, 1.0), 1e-2: stat(0.02, 1.02)}
    b = {0.0: stat(0.05, 2.0), 1e-3: stat(0.04, 2.0), 1e-2: stat(0.01, 2.5)}  # 1e-2 breaks b's accuracy
    selected, best, feasible, *_ = sw.select_shared({"a": a, "b": b}, acc_tol=0.05)
    assert feasible == [0.0, 1e-3] and selected == 1e-3
    assert sw.select_lambda(a, 0.05)[0] == 1e-2  # alone, a would pick the larger lambda


def test_aggregate_counts_between_seed_noise():
    runs = [{"n": 100, "failed": f, "force_mae": 1.0, "energy_mae": 1.0, "failed_bond_only": 0} for f in (0, 40)]
    agg = sw.aggregate(runs)
    assert agg["frac"] == 0.2
    assert agg["se"] == pytest.approx(0.2)  # std(0, 0.4)/sqrt(2) exceeds the binomial 0.028


def test_z_test():
    assert sw.z_test(10, 100, 10, 100) == pytest.approx(1.0)
    assert sw.z_test(66, 256, 6, 256) < 1e-10


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        sw.load_sweep(write_config(tmp_path, grid={"lam": [1e-3, 1e-2], "seeds": [0]}))
    with pytest.raises(ValueError):
        sw.load_sweep(write_config(tmp_path, systems=[{"molecule": "x", "model": "m"}] * 2))
    cfg = sw.load_sweep(write_config(tmp_path, systems=[
        {"molecule": "aspirin", "model": "schnet"},
        {"dataset": "md22", "molecule": "AT_AT_CG_CG", "model": "gemnet_t",
         "select_eval": {"overrides": {"simulate": {"n_replicas": 16}}}}]))
    md17, md22 = cfg["systems"]["aspirin_schnet"], cfg["systems"]["AT_AT_CG_CG_gemnet_t"]
    assert md17["train"]["config"].endswith("configs/margin/base.yml")
    assert md22["data"]["name"] == "md22" and "backfill" not in md22["tune_split"]
    assert md17["tune_split"]["backfill"]["unit"] == "kcal/mol"
    assert md22["train"]["config"].endswith("configs/margin/md22.yml")
    assert md22["select_eval"]["config"].endswith("configs/eval/md22_500K.yml")
    ov = md22["select_eval"]["overrides"]  # the system's override merges into the defaults
    assert ov["simulate"] == {"length_ps": 100, "n_replicas": 16} and ov["data"] == {"start_split": "tune"}
    assert md22["train"]["overrides"]["optim"] == {"lr": 1.0e-5}
    assert len(sw.jobs(cfg, "tune")) == 2 * 7 * 2
    with pytest.raises(ValueError):
        sw.load_sweep(write_config(tmp_path, systems=[{"dataset": "qm9", "molecule": "x", "model": "m"}]))


def test_prepare_tune_from_raw_md17(tmp_path, monkeypatch):
    monkeypatch.setattr(sw, "ROOT", tmp_path)
    cfg = sw.load_sweep(write_config(tmp_path, systems=[{"molecule": "ethanol", "model": "schnet"}]))
    s = cfg["systems"]["ethanol_schnet"]
    s["tune_split"]["n_frames"] = 50
    n, rng = 500, np.random.default_rng(0)
    raw = tmp_path / "data/raw/rmd17"
    raw.mkdir(parents=True)
    np.savez(raw / "rmd17_ethanol.npz", nuclear_charges=np.array([6, 1]), coords=rng.normal(size=(n, 2, 3)),
             energies=rng.normal(size=n), forces=rng.normal(size=(n, 2, 3)))
    base = sw.split_dir(s)
    base.mkdir(parents=True)
    splits = {"train": np.arange(0, 100), "val": np.arange(100, 110), "test": np.arange(110, 210)}
    np.save(base / "splits", splits)
    sw.prepare_tune(s)
    tune = np.load(base / "tune/nequip_npz.npz")
    idx = np.load(base / "splits.npy", allow_pickle=True).item()["tune"]
    assert tune["R"].shape == (50, 2, 3) and idx.min() >= 210


def fake_runs(tmp_path, cfg, sid, fails):
    for lam, k in fails.items():
        for seed in cfg["grid"]["seeds"]:
            d = tmp_path / "results/eval" / sw.eval_tag(cfg, cfg["systems"][sid], "tune", lam, seed)
            d.mkdir(parents=True)
            obs = np.zeros(256, bool)
            obs[:k] = True
            np.savez(d / "stability.npz", observed=obs, observed_bond_only=np.zeros(256, bool))
            (d / "metrics.json").write_text(json.dumps({"force_mae_kcal_mol_A": 1.0, "energy_mae_kcal_mol": 0.5}))


def test_select_end_to_end_and_final_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(sw, "ROOT", tmp_path)
    cfg = sw.load_sweep(write_config(tmp_path, grid={"lam": [0, 1e-3], "seeds": [0, 1]}))
    fake_runs(tmp_path, cfg, "ethanol_schnet", {0.0: 20, 1e-3: 18})
    fake_runs(tmp_path, cfg, "aspirin_schnet", {0.0: 120, 1e-3: 10})
    out = sw.select(cfg)
    assert out["selected"] == {"ethanol_schnet": 1e-3, "aspirin_schnet": 1e-3}
    assert out["systems"]["ethanol_schnet"]["selected_alone"] == 0.0
    assert sw.lock_final(cfg) == out["selected"]
    assert len(sw.jobs(cfg, "final")) == 2 * 3 * 3
    sel = sw.out_dir(cfg) / "selection.json"
    sel.write_text(json.dumps({**json.loads(sel.read_text()), "selected": {"aspirin_schnet": 1e-2}}))
    with pytest.raises(SystemExit):  # the test set was already used with another selection
        sw.lock_final(cfg)
