"""Score + adjudicate the Wave-1 batches against the pre-registered gates (research-log 0008).

  ladder  G-DECOMP: arms heads/headsenc/wm/full (evals/wave1_decomp_*) vs the archived
          same-protocol control (evals/loop9_s1_model_e[1-8]_*).
          Computes H = mean(full) - mean(control), per-scope shares of H, Welch p per arm
          vs control, and evaluates P-W1a + P-W1b including the +/-10pp extension rule.

  k3      3-regime screen: arms init/model/o2 (evals/wave1_k3_*).
          Evaluates P-W1c against the ladder's H (pass --h2 with the K=2 headroom).

Scoring goes through the validated score_eval_dir staging + the frozen scorer. Results are
printed and written to evals/wave1_scores.json (local; the adjudicated numbers get committed
into the LOOP-0011 note).

Usage:
    PYTHONPATH=src myenv\\Scripts\\python.exe scripts/score_wave1.py ladder
    PYTHONPATH=src myenv\\Scripts\\python.exe scripts/score_wave1.py k3 --h2 <H from ladder>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from score_eval_dir import _stage_run_layout, welch  # noqa: E402
from lifelong_learning.research.benchmarking import score_brain_run  # noqa: E402

LADDER_ARMS = {
    "control": "evals/loop9_s1_model_e[1-8]_*",  # archived no-swap arm, same Brain/protocol/seeds
    "heads": "evals/wave1_decomp_heads_e*",
    "heads+encoder": "evals/wave1_decomp_headsenc_e*",
    "world_model": "evals/wave1_decomp_wm_e*",
    "full": "evals/wave1_decomp_full_e*",
}
K3_ARMS = {
    "init": "evals/wave1_k3_init_e*",
    "model": "evals/wave1_k3_model_e*",
    "o2": "evals/wave1_k3_o2_e*",
}
G3_ARMS = {
    "control": "evals/loop9_s1_model_e[1-8]_*",       # archived, no swap
    "heads_swap": "evals/wave1_decomp_heads_e*",       # archived ceiling slice (oracle O2-heads)
    "g3_oracle": "evals/g3_oracle_e*",
    "g3_ar1": "evals/g3_ar1_e*",
    "g3_ar1ve": "evals/g3_ar1ve_e*",
}
G3RE_ARMS = {  # LOOP-0013 (log 0010): selection rung with spawn-until-full allocation
    "control": "evals/loop9_s1_model_e*",              # archived, n=16
    "heads_swap": "evals/wave1_decomp_heads_e*",
    "g3_ar1": "evals/g3_ar1_e*",                       # n=16 after the ar1x extension
    "g3_ar1re": "evals/g3_ar1re_e*",
    "g3_ar1ve2": "evals/g3_ar1ve2_e*",
}
T15_ARMS = {  # LOOP-0013 addendum (log 0011): trigger hardening at threshold 1.5
    "control": "evals/loop9_s1_model_e*",
    "g3_ar1": "evals/g3_ar1_e*",
    "g3_ar1t15": "evals/g3_ar1t15_e*",
}
STP_ARMS = {  # LOOP-0014 (log 0012): per-step success-collapse trigger
    "control": "evals/loop9_s1_model_e*",
    "heads_swap": "evals/wave1_decomp_heads_e*",
    "g3_ar1": "evals/g3_ar1_e*",
    "g3_stp": "evals/g3_stp_e*",
}
# Ground-truth switch schedule of the eval protocol, for trigger precision/recall.
G3_SWITCHES = list(range(100000, 800000, 100000))
G3_WINDOW_STEPS = 4 * 2048
STP_WINDOW_STEPS = 4096                     # per-step trigger: much tighter TP window
STP_RECALL_SWITCHES = G3_SWITCHES[1:]       # first switch structurally exempt (log 0012)
FP_ARMS = {  # LOOP-0015 (log 0013): drift-robust fingerprint selection
    "control": "evals/loop9_s1_model_e*",
    "heads_swap": "evals/wave1_decomp_heads_e*",
    "g3_ar1": "evals/g3_ar1_e*",            # blind K=2 flip, the method reference (n=16)
    "g3_ar1re": "evals/g3_ar1re_e*",         # drift-bound fingerprint (the null being fixed)
    "g3_fp": "evals/g3_fp_e*",
}
DORM_ARMS = {  # LOOP-0016 / W0c (log 0014): encoder-dormancy probe
    "control": "evals/loop9_s1_model_e*",    # archived, same Brain/protocol/seeds
    "dorm": "evals/dorm_e*",
}
DORM_SITES = ("conv1", "conv2", "conv3", "actor", "critic")
REDOC1_ARMS = {  # LOOP-0017 / branch F (log 0015): conv1-targeted ReDo
    "control": "evals/loop9_s1_model_e*",    # archived
    "redoc1": "evals/redoc1_e*",
}
IPCAL_ARMS = {  # LOOP-0018 / IP-1 calibration (log 0016): action-flip instrument
    "ipflip_ctrl": "evals/ipflip_ctrl_e*",
    "ipflip_o2": "evals/ipflip_o2_e*",
}
UPDATE_STEPS = 2048                          # 16 envs x 128 steps per inner update


def score_arm_full(pattern: str, staging: Path) -> list[dict]:
    dirs = sorted(Path(p) for p in glob.glob(str(REPO_ROOT / pattern)) if Path(p).is_dir())
    if not dirs:
        raise FileNotFoundError(f"No eval dirs match {pattern}")
    rows = []
    for d in dirs:
        s = score_brain_run(_stage_run_layout(d, staging))
        rows.append({"eval_dir": d.name, "composite": s.composite_score,
                     "hit80": s.hit_rate_80, "hit95": s.hit_rate_95})
    return rows


def summarize(arms: dict[str, str]) -> dict[str, dict]:
    staging = Path(tempfile.mkdtemp(prefix="wave1_score_"))
    out = {}
    for label, pat in arms.items():
        rows = score_arm_full(pat, staging)
        comp = np.array([r["composite"] for r in rows])
        out[label] = {
            "n": len(rows),
            "mean": float(np.mean(comp)),
            "sd": float(np.std(comp, ddof=1)),
            "hit80": float(np.mean([r["hit80"] for r in rows])),
            "hit95": float(np.mean([r["hit95"] for r in rows])),
            "composites": comp,
        }
        print(f"  {label:<15} n={len(rows):>2}  mean {np.mean(comp):.4f}  sd {np.std(comp, ddof=1):.4f}  "
              f"hit80 {out[label]['hit80']:.3f}  hit95 {out[label]['hit95']:.3f}")
    return out


def adjudicate_ladder(a: dict[str, dict]) -> dict:
    c = a["control"]["composites"]
    H = float(np.mean(a["full"]["composites"]) - np.mean(c))
    wf = welch(a["full"]["composites"], c)
    print(f"\nH (full - control) = {H:+.4f}   Welch p = {wf['p']:.3g}")

    res = {"H": H, "full_vs_control_p": wf["p"], "shares": {}, "arm_vs_control": {}}
    for scope in ("heads", "heads+encoder", "world_model"):
        w = welch(a[scope]["composites"], c)
        share = (np.mean(a[scope]["composites"]) - np.mean(c)) / H if H != 0 else float("nan")
        res["shares"][scope] = float(share)
        res["arm_vs_control"][scope] = w
        print(f"  share({scope:<14}) = {share:+7.1%}   delta {w['delta']:+.4f}  p={w['p']:.3g}")

    p_w1a = (H > 0) and (wf["p"] < 0.05) and (H >= 0.05)
    sh, she, swm = (res["shares"][k] for k in ("heads", "heads+encoder", "world_model"))
    kill_weightspace = (sh < 0.20) and (she < 0.40)
    routing = []
    if sh >= 0.40:
        routing.append("W2A weight-space (G3) opens: heads >= 40%")
    if swm >= 0.20:
        routing.append("W2B MoWM (D) opens: world_model >= 20%")
    if kill_weightspace:
        routing.append("weight-space KILLED (heads<20% AND heads+enc<40%) -> route by larger of WM/scaling")
    if not routing:
        routing.append("no subtree threshold met -> mixed/inconclusive, apply extension rule")

    # Extension rule: any deciding share within +/-10pp of its threshold -> extend to n=16.
    near = []
    if abs(sh - 0.40) <= 0.10 or abs(sh - 0.20) <= 0.10:
        near.append("heads")
    if abs(she - 0.40) <= 0.10:
        near.append("heads+encoder")
    if abs(swm - 0.20) <= 0.10:
        near.append("world_model")

    res.update({"P_W1a_pass": bool(p_w1a), "routing": routing, "extend_arms_to_16": near})
    print(f"\nP-W1a (H>0, p<0.05, H>=+0.05): {'PASS' if p_w1a else 'FAIL'}")
    for r in routing:
        print(f"P-W1b routing: {r}")
    print(f"Extension rule (+/-10pp): {'extend ' + ', '.join(near) if near else 'no extension needed'}")
    return res


def adjudicate_k3(a: dict[str, dict], h2: float) -> dict:
    gap = float(np.mean(a["o2"]["composites"]) - np.mean(a["model"]["composites"]))
    w = welch(a["o2"]["composites"], a["model"]["composites"])
    ti = welch(a["model"]["composites"], a["init"]["composites"])
    p_w1c = gap > h2
    print(f"\n(o2 - model)@K3 = {gap:+.4f} (p={w['p']:.3g})  vs  H@K2 = {h2:+.4f}"
          f"  ->  P-W1c {'PASS: headroom grows with K' if p_w1c else 'FAIL: scaling axis dies'}")
    print(f"secondary trained-vs-init @K3: {ti['delta']:+.4f} (p={ti['p']:.3g})")
    return {"o2_minus_model_k3": gap, "p": w["p"], "h2": h2, "P_W1c_pass": bool(p_w1c),
            "trained_vs_init_k3": ti}


def _g3_trigger_stats(pattern: str) -> dict:
    """Precision/recall of the in-run surprise trigger vs the ground-truth schedule,
    from the logged head_bank_trigger_fired steps in each eval's data json."""
    import json as _json

    tps = fps = dets = 0
    for d in sorted(glob.glob(str(REPO_ROOT / pattern))):
        js = glob.glob(str(Path(d) / "**" / "*_data.json"), recursive=True)
        if not js:
            continue
        data = _json.load(open(js[0]))
        fired = [int(step) for step, _ in data.get("brain_neuromod/head_bank_trigger_fired", [])]
        hit_switches = set()
        for f in fired:
            match = next((sw for sw in G3_SWITCHES if 0 <= f - sw <= G3_WINDOW_STEPS), None)
            if match is not None:
                tps += 1
                hit_switches.add(match)
            else:
                fps += 1
        dets += len(hit_switches)
    n_runs = len(glob.glob(str(REPO_ROOT / pattern)))
    return {
        "precision": tps / max(1, tps + fps),
        "recall": dets / max(1, len(G3_SWITCHES) * n_runs),
        "fires_per_run": (tps + fps) / max(1, n_runs),
    }


def adjudicate_g3(a: dict[str, dict]) -> dict:
    c = a["control"]["composites"]
    heads_gain = float(np.mean(a["heads_swap"]["composites"]) - np.mean(c))
    res = {"heads_gain_ref": heads_gain, "arms": {}, "triggers": {}}
    print(f"\nreference: oracle heads-swap gain = {heads_gain:+.4f}")
    for arm in ("g3_oracle", "g3_ar1", "g3_ar1ve"):
        w = welch(a[arm]["composites"], c)
        share = w["delta"] / heads_gain if heads_gain else float("nan")
        res["arms"][arm] = {**w, "share_of_heads_gain": float(share)}
        print(f"  {arm:<10} gain {w['delta']:+.4f} (p={w['p']:.3g})  = {share:+.1%} of heads gain")
    for arm in ("g3_ar1", "g3_ar1ve"):
        t = _g3_trigger_stats(G3_ARMS[arm])
        res["triggers"][arm] = t
        print(f"  {arm:<10} trigger precision {t['precision']:.2f}  recall {t['recall']:.2f}  "
              f"fires/run {t['fires_per_run']:.1f}")

    equiv = welch(a["g3_oracle"]["composites"], a["heads_swap"]["composites"])
    p_g3a = (equiv["p"] > 0.05) and (res["arms"]["g3_oracle"]["delta"] >= 0.8 * heads_gain)
    ar1 = res["arms"]["g3_ar1"]
    p_g3b = (ar1["delta"] >= 0.05) and (res["triggers"]["g3_ar1"]["precision"] >= 0.60)
    ve = res["arms"]["g3_ar1ve"]
    p_g3c = ve["delta"] >= ar1["delta"] - 0.02
    res.update({
        "P_G3a_equivalence": {"pass": bool(p_g3a), "vs_heads_p": equiv["p"], "vs_heads_delta": equiv["delta"]},
        "P_G3b_ar1": {"pass": bool(p_g3b)},
        "P_G3c_selection": {"pass": bool(p_g3c)},
    })
    print(f"\nP-G3a equivalence (bank == heads swap): {'PASS' if p_g3a else 'FAIL'} "
          f"(vs heads: d={equiv['delta']:+.4f}, p={equiv['p']:.2g})")
    print(f"P-G3b A-R1 (gain>=+0.05 AND precision>=0.60): {'PASS' if p_g3b else 'FAIL -> trigger premise dead'}")
    print(f"P-G3c selection (value_error >= other - 0.02): {'PASS' if p_g3c else 'FAIL'}")
    return res


def adjudicate_g3re(a: dict[str, dict]) -> dict:
    """LOOP-0013 gates (log 0010): P-G3d selection rung + P-G3e ar1 n=16 convergence."""
    c = a["control"]["composites"]
    heads_gain = float(np.mean(a["heads_swap"]["composites"]) - np.mean(c))
    res = {"heads_gain_ref": heads_gain, "arms": {}, "triggers": {}}
    print(f"\nreference: oracle heads-swap gain (n=8) = {heads_gain:+.4f}")
    for arm in ("g3_ar1", "g3_ar1re", "g3_ar1ve2"):
        w = welch(a[arm]["composites"], c)
        res["arms"][arm] = {**w, "share_of_heads_gain": float(w["delta"] / heads_gain) if heads_gain else None}
        t = _g3_trigger_stats(G3RE_ARMS[arm])
        res["triggers"][arm] = t
        print(f"  {arm:<10} n={a[arm]['n']:>2} gain {w['delta']:+.4f} (p={w['p']:.3g}) "
              f"= {res['arms'][arm]['share_of_heads_gain']:+.1%} of slice | "
              f"trig prec {t['precision']:.2f} rec {t['recall']:.2f}")

    ar1, re_, ve2 = (res["arms"][k]["delta"] for k in ("g3_ar1", "g3_ar1re", "g3_ar1ve2"))
    p_g3d = re_ >= ar1 - 0.02
    p_g3d_ve = ve2 >= ar1 - 0.02
    ar1_n16_converges = abs(ar1 - 0.1180) <= 0.03 and res["arms"]["g3_ar1"]["p"] < 0.01
    res.update({
        "P_G3d_reward_error": {"pass": bool(p_g3d)},
        "P_G3d_value_error_rerun": {"pass": bool(p_g3d_ve)},
        "P_G3e_ar1_convergence": {"pass": bool(ar1_n16_converges), "n8_ref": 0.1180},
    })
    print(f"\nP-G3d reward_error (>= ar1 - 0.02): {'PASS' if p_g3d else 'FAIL'}")
    print(f"P-G3d value_error re-run (>= ar1 - 0.02): {'PASS' if p_g3d_ve else 'FAIL'}")
    print(f"P-G3e ar1 n=16 convergence (within +/-0.03 of +0.118, p<0.01): "
          f"{'PASS' if ar1_n16_converges else 'FAIL'}")
    return res


def adjudicate_t15(a: dict[str, dict]) -> dict:
    """LOOP-0013 addendum gates (log 0011): P-T15a precision transfer, P-T15b gain lift."""
    c = a["control"]["composites"]
    ar1 = welch(a["g3_ar1"]["composites"], c)
    t15 = welch(a["g3_ar1t15"]["composites"], c)
    trig = _g3_trigger_stats(T15_ARMS["g3_ar1t15"])
    p_a = trig["precision"] >= 0.80
    p_b = t15["delta"] >= ar1["delta"] + 0.02
    res = {"ar1_ref": ar1, "t15": t15, "trigger": trig,
           "P_T15a": {"pass": bool(p_a)}, "P_T15b": {"pass": bool(p_b)}}
    print(f"\nar1@1.0 (n={a['g3_ar1']['n']}): gain {ar1['delta']:+.4f} | "
          f"t15@1.5 (n={a['g3_ar1t15']['n']}): gain {t15['delta']:+.4f} (p={t15['p']:.3g})")
    print(f"t15 trigger: precision {trig['precision']:.2f} recall {trig['recall']:.2f} "
          f"fires/run {trig['fires_per_run']:.1f}")
    print(f"P-T15a (precision >= 0.80): {'PASS' if p_a else 'FAIL'}")
    print(f"P-T15b (gain >= ar1 + 0.02 = {ar1['delta'] + 0.02:+.4f}): {'PASS' if p_b else 'FAIL'}")
    return res


def _stp_trigger_stats(pattern: str) -> dict:
    """Live precision/recall/lag of the per-step trigger vs the ground-truth schedule,
    from the logged head_bank_step_fire steps (log 0012: TP window 4,096 steps; recall
    over switches 2-7 — the first switch is structurally exempt, nothing banked yet)."""
    import json as _json
    import statistics as _st

    tps = fps = dets = 0
    lags: list[int] = []
    dirs = sorted(glob.glob(str(REPO_ROOT / pattern)))
    for d in dirs:
        js = glob.glob(str(Path(d) / "**" / "*_data.json"), recursive=True)
        if not js:
            continue
        data = _json.load(open(js[0]))
        fired = [int(step) for step, _ in data.get("brain_neuromod/head_bank_step_fire", [])]
        hit = set()
        for f in fired:
            match = next((sw for sw in G3_SWITCHES if 0 <= f - sw <= STP_WINDOW_STEPS), None)
            if match is not None:
                tps += 1
                if match not in hit and match in STP_RECALL_SWITCHES:
                    lags.append(f - match)
                hit.add(match)
            else:
                fps += 1
        dets += len(hit & set(STP_RECALL_SWITCHES))
    n_runs = max(1, len(dirs))
    return {
        "precision": tps / max(1, tps + fps),
        "recall": dets / max(1, len(STP_RECALL_SWITCHES) * n_runs),
        "fires_per_run": (tps + fps) / n_runs,
        "median_lag_steps": _st.median(lags) if lags else None,
    }


def adjudicate_stp(a: dict[str, dict]) -> dict:
    """LOOP-0014 gates (log 0012): P-S1a trigger quality live, P-S1b the lag payoff."""
    c = a["control"]["composites"]
    heads_gain = float(np.mean(a["heads_swap"]["composites"]) - np.mean(c))
    ar1 = welch(a["g3_ar1"]["composites"], c)
    stp = welch(a["g3_stp"]["composites"], c)
    trig = _stp_trigger_stats(STP_ARMS["g3_stp"])
    p_a = trig["precision"] >= 0.75 and trig["recall"] >= 5 / 6
    p_b = stp["delta"] >= 0.139
    res = {"heads_gain_ref": heads_gain, "ar1_ref": ar1, "stp": stp, "trigger": trig,
           "P_S1a": {"pass": bool(p_a)}, "P_S1b": {"pass": bool(p_b)}}
    print(f"\noracle heads slice = {heads_gain:+.4f} | ar1 (n={a['g3_ar1']['n']}) gain {ar1['delta']:+.4f}")
    print(f"g3_stp (n={a['g3_stp']['n']}): gain {stp['delta']:+.4f} (p={stp['p']:.3g}) "
          f"= {stp['delta'] / heads_gain:+.1%} of slice")
    print(f"stp trigger: precision {trig['precision']:.2f} recall {trig['recall']:.2f} "
          f"fires/run {trig['fires_per_run']:.1f} median lag {trig['median_lag_steps']} steps")
    print(f"P-S1a (precision >= 0.75 AND recall >= 5/6): {'PASS' if p_a else 'FAIL'}")
    print(f"P-S1b (gain >= +0.139): {'PASS' if p_b else 'FAIL'}")
    return res


def _fp_behavior_stats(pattern: str) -> dict:
    """Per-fire selection behavior (log 0013): a fire inside the TP window SHOULD flip
    (restore the other regime's slot; the first fire's spawn counts); a fire outside it
    SHOULD stay (false-fire suppression). Reconstructed from the logged fire steps and
    the head_bank_active trace (one entry per fire, value = active slot AFTER)."""
    import json as _json

    should_flip = did_flip = should_stay = did_stay = 0
    for d in sorted(glob.glob(str(REPO_ROOT / pattern))):
        js = glob.glob(str(Path(d) / "**" / "*_data.json"), recursive=True)
        if not js:
            continue
        data = _json.load(open(js[0]))
        fired = [int(step) for step, _ in data.get("brain_neuromod/head_bank_trigger_fired", [])]
        active = {int(step): int(v) for step, v in data.get("brain_neuromod/head_bank_active", [])}
        prev = 0
        for f in fired:
            after = active.get(f, prev)
            flipped = after != prev
            prev = after
            in_window = any(0 <= f - sw <= G3_WINDOW_STEPS for sw in G3_SWITCHES)
            if in_window:
                should_flip += 1
                did_flip += flipped
            else:
                should_stay += 1
                did_stay += not flipped
    return {
        "flip_when_should": did_flip / max(1, should_flip),
        "stay_when_should": did_stay / max(1, should_stay),
        "window_fires": should_flip,
        "off_window_fires": should_stay,
    }


def adjudicate_fp(a: dict[str, dict]) -> dict:
    """LOOP-0015 gates (log 0013): P-FP1 selection behavior, P-FP2 non-inferiority."""
    c = a["control"]["composites"]
    heads_gain = float(np.mean(a["heads_swap"]["composites"]) - np.mean(c))
    ar1 = welch(a["g3_ar1"]["composites"], c)
    re_ = welch(a["g3_ar1re"]["composites"], c)
    fp = welch(a["g3_fp"]["composites"], c)
    trig = _g3_trigger_stats(FP_ARMS["g3_fp"])
    beh = _fp_behavior_stats(FP_ARMS["g3_fp"])
    p_fp1 = beh["flip_when_should"] >= 0.75 and beh["stay_when_should"] >= 0.75
    p_fp2 = fp["delta"] >= ar1["delta"] - 0.02
    dividend = fp["delta"] >= ar1["delta"] + 0.02
    res = {"heads_gain_ref": heads_gain, "ar1_ref": ar1, "ar1re_ref": re_, "fp": fp,
           "trigger": trig, "behavior": beh,
           "P_FP1": {"pass": bool(p_fp1)}, "P_FP2": {"pass": bool(p_fp2)},
           "suppression_dividend": bool(dividend)}
    print(f"\noracle heads slice = {heads_gain:+.4f} | ar1 flip {ar1['delta']:+.4f} | "
          f"drift-bound fp {re_['delta']:+.4f}")
    print(f"g3_fp (n={a['g3_fp']['n']}): gain {fp['delta']:+.4f} (p={fp['p']:.3g}) "
          f"= {fp['delta'] / heads_gain:+.1%} of slice")
    print(f"trigger: precision {trig['precision']:.2f} recall {trig['recall']:.2f} "
          f"fires/run {trig['fires_per_run']:.1f}")
    print(f"behavior: flip-when-should {beh['flip_when_should']:.2f} "
          f"({beh['window_fires']} window fires), stay-when-should {beh['stay_when_should']:.2f} "
          f"({beh['off_window_fires']} off-window fires)")
    print(f"P-FP1 (flip>=0.75 AND stay>=0.75): {'PASS' if p_fp1 else 'FAIL'}")
    print(f"P-FP2 (gain >= ar1 - 0.02 = {ar1['delta'] - 0.02:+.4f}): {'PASS' if p_fp2 else 'FAIL'}")
    print(f"secondary — suppression dividend (gain >= ar1 + 0.02): {'YES' if dividend else 'no'}")
    return res


def _dorm_window_stats(pattern: str, prefix: str = "dormancy") -> dict:
    """LOOP-0016 (log 0014): per-run windowed trajectory stats for each probe site. The
    per-update dormant-fraction series splits into 8 equal windows; rise = f8 - min(f1..f4)
    (accumulation vs the early-training trough), fall = f8 - f1 (the C4 shape). Returns
    per-site medians over runs."""
    import json as _json

    per_site: dict[str, dict[str, list[float]]] = {
        s: {"f1": [], "f8": [], "rise": [], "fall": []} for s in DORM_SITES}
    runs = 0
    for d in sorted(glob.glob(str(REPO_ROOT / pattern))):
        js = glob.glob(str(Path(d) / "**" / "*_data.json"), recursive=True)
        if not js:
            continue
        data = _json.load(open(js[0]))
        got_any = False
        for site in DORM_SITES:
            series = data.get(f"brain_neuromod/{prefix}_{site}", [])
            if len(series) < 16:
                continue
            got_any = True
            vals = np.array([v for _, v in sorted(series, key=lambda pt: pt[0])], dtype=float)
            f = [float(w.mean()) for w in np.array_split(vals, 8)]
            per_site[site]["f1"].append(f[0])
            per_site[site]["f8"].append(f[-1])
            per_site[site]["rise"].append(f[-1] - min(f[:4]))
            per_site[site]["fall"].append(f[-1] - f[0])
        runs += int(got_any)
    return {"runs": runs, "sites": {
        s: {k: (float(np.median(v)) if v else None) for k, v in d.items()}
        for s, d in per_site.items()}}


def adjudicate_dorm(a: dict[str, dict]) -> dict:
    """LOOP-0016 gates (log 0014): P-W0c2 (instrument integrity) read FIRST; only if it
    passes is P-W0c1 (encoder no-accumulation) read."""
    stats = _dorm_window_stats(DORM_ARMS["dorm"])
    stats10 = _dorm_window_stats(DORM_ARMS["dorm"], prefix="dormancy10")
    w = welch(a["dorm"]["composites"], a["control"]["composites"])

    heads_fall_ok = all(
        stats["sites"][s]["fall"] is not None and stats["sites"][s]["fall"] < 0
        for s in ("actor", "critic"))
    inert_ok = (w["p"] > 0.05) and (abs(w["delta"]) < 0.02)
    p_w0c2 = heads_fall_ok and inert_ok

    conv_rises = {s: stats["sites"][s]["rise"] for s in ("conv1", "conv2", "conv3")}
    p_w0c1 = all(r is not None and r <= 0.05 for r in conv_rises.values())
    near = [s for s, r in conv_rises.items() if r is not None and abs(r - 0.05) <= 0.02]

    res = {"probe_runs": stats["runs"], "tau025": stats["sites"], "tau10": stats10["sites"],
           "composite_vs_control": w,
           "P_W0c2": {"pass": bool(p_w0c2), "heads_fall_ok": bool(heads_fall_ok),
                      "composite_inert_ok": bool(inert_ok)},
           "P_W0c1": {"pass": (bool(p_w0c1) if p_w0c2 else None), "conv_rises": conv_rises},
           "extend_to_16": near}
    print(f"\ncomposite: dorm vs control delta {w['delta']:+.4f} (p={w['p']:.3g})   "
          f"probe runs with trajectories: {stats['runs']}")
    for s in DORM_SITES:
        d = stats["sites"][s]
        if d["f1"] is None:
            print(f"  {s:<6} NO TRAJECTORY DATA")
            continue
        print(f"  {s:<6} f1 {d['f1']:.3f} -> f8 {d['f8']:.3f}   "
              f"rise(vs early trough) {d['rise']:+.3f}   fall {d['fall']:+.3f}")
    print(f"P-W0c2 (heads fall < 0 AND composite inert |d|<0.02, p>0.05): "
          f"{'PASS' if p_w0c2 else 'FAIL'}  "
          f"[heads_fall {'ok' if heads_fall_ok else 'FAIL'}, inert {'ok' if inert_ok else 'FAIL'}]")
    if p_w0c2:
        print(f"P-W0c1 (median rise <= +0.05 for every conv): {'PASS' if p_w0c1 else 'FAIL'}")
    else:
        print("P-W0c1: NOT READ (P-W0c2 failed -- instrument invalid per log 0014)")
    if near:
        print(f"Extension rule (within +/-0.02 of the +0.05 bar): extend {', '.join(near)} to n=16")
    return res


def adjudicate_redoc1(a: dict[str, dict]) -> dict:
    """LOOP-0017 gates (log 0015): P-F1a mechanism (conv1 accumulation eliminated) read
    FIRST; only then P-F1b (composite payoff, MDE-honest at +0.05)."""
    stats = _dorm_window_stats(REDOC1_ARMS["redoc1"])
    conv1 = stats["sites"]["conv1"]
    w = welch(a["redoc1"]["composites"], a["control"]["composites"])

    p_f1a = (conv1["f8"] is not None and conv1["f8"] <= 0.15
             and conv1["rise"] is not None and conv1["rise"] <= 0.05)
    p_f1b = (w["delta"] >= 0.05) and (w["p"] < 0.05)
    harm = w["delta"] <= -0.02
    near = abs(w["delta"] - 0.05) <= 0.02
    if conv1["f8"] is not None:
        near = near or abs(conv1["f8"] - 0.15) <= 0.02

    res = {"probe_runs": stats["runs"], "conv1": conv1, "sites": stats["sites"],
           "composite_vs_control": w,
           "P_F1a": {"pass": bool(p_f1a)},
           "P_F1b": {"pass": (bool(p_f1b) if p_f1a else None)},
           "harm_flag": bool(harm), "extend_to_16": bool(near)}
    print(f"\ncomposite: redoc1 vs control delta {w['delta']:+.4f} (p={w['p']:.3g})")
    for s in DORM_SITES:
        d = stats["sites"][s]
        if d["f1"] is None:
            print(f"  {s:<6} NO TRAJECTORY DATA")
            continue
        print(f"  {s:<6} f1 {d['f1']:.3f} -> f8 {d['f8']:.3f}   rise {d['rise']:+.3f}   fall {d['fall']:+.3f}")
    print(f"P-F1a (conv1 f8 <= 0.15 AND rise <= +0.05): {'PASS' if p_f1a else 'FAIL'}")
    if p_f1a:
        print(f"P-F1b (delta >= +0.05, p < 0.05): {'PASS' if p_f1b else 'FAIL'}"
              + ("" if p_f1b else "  -> registered null reading: first-layer plasticity"
                                 " loss is behaviorally cheap (< +0.05) at this horizon"))
    else:
        print("P-F1b: NOT READ (mechanism did not hold dormancy down)")
    if harm:
        print("HARM flag: delta <= -0.02 (the O1 precedent -- intervention hurts)")
    if near:
        print("Extension rule: deciding quantity within its +/-0.02 band -> extend to n=16")
    return res


def _wm_error_persistence(pattern: str, switches: list[int]) -> dict:
    """LOOP-0018 C-IP-b (log 0016): per true switch, how many consecutive post-switch
    UPDATES the WM next-state error stays above baseline_mean + 2*baseline_sd, where
    baseline = the 20 updates before the switch. The dense debug/wm_raw_error_mean series
    is bucketed to per-update means first. Returns the median persistence (updates)."""
    import json as _json
    import statistics as _st

    persistences = []
    for d in sorted(glob.glob(str(REPO_ROOT / pattern))):
        js = glob.glob(str(Path(d) / "**" / "*_data.json"), recursive=True)
        if not js:
            continue
        series = _json.load(open(js[0])).get("debug/wm_raw_error_mean", [])
        if not series:
            continue
        buckets: dict[int, list[float]] = {}
        for step, v in series:
            buckets.setdefault(int(step) // UPDATE_STEPS, []).append(float(v))
        upd = {u: float(np.mean(vs)) for u, vs in buckets.items()}
        for sw in switches:
            u0 = sw // UPDATE_STEPS
            base = [upd[u] for u in range(u0 - 20, u0) if u in upd]
            if len(base) < 10:
                continue
            thr = float(np.mean(base)) + 2.0 * float(np.std(base))
            k = 0
            while (u0 + k) in upd and upd[u0 + k] > thr:
                k += 1
            persistences.append(k)
    return {"median_persistence_updates": (_st.median(persistences) if persistences else None),
            "n_switches_measured": len(persistences)}


def adjudicate_ipcal(a: dict[str, dict]) -> dict:
    """LOOP-0018 gates (log 0016): C-IP-a headroom, C-IP-b regime-signal persistence."""
    ctrl = a["ipflip_ctrl"]["composites"]
    o2 = welch(a["ipflip_o2"]["composites"], ctrl)
    pers = _wm_error_persistence(IPCAL_ARMS["ipflip_ctrl"], G3_SWITCHES)

    c_ip_a = (o2["delta"] >= 0.10) and (o2["p"] < 0.05)
    med = pers["median_persistence_updates"]
    c_ip_b = med is not None and med >= 4
    near = abs(o2["delta"] - 0.10) <= 0.02

    res = {"o2_vs_ctrl": o2, "wm_persistence": pers,
           "C_IP_a": {"pass": bool(c_ip_a)}, "C_IP_b": {"pass": bool(c_ip_b)},
           "extend_to_16": bool(near)}
    print(f"\nipflip O2 ceiling vs ctrl: delta {o2['delta']:+.4f} (p={o2['p']:.3g})")
    print(f"WM next-state-error persistence at true switches: median "
          f"{med} updates over {pers['n_switches_measured']} switches")
    print(f"C-IP-a (ceiling >= +0.10, p<0.05): {'PASS' if c_ip_a else 'FAIL'}")
    print(f"C-IP-b (persistence >= 4 updates): {'PASS' if c_ip_b else 'FAIL'}")
    if near:
        print("Extension rule: ceiling within +/-0.02 of +0.10 -> extend to n=16")
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch", choices=["ladder", "k3", "g3", "g3re", "t15", "stp", "fp", "dorm", "redoc1", "ipcal"])
    p.add_argument("--h2", type=float, default=None, help="K=2 headroom H from the ladder (k3 only)")
    p.add_argument("--json-out", default="evals/wave1_scores.json")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    print(f"== Wave-1 {args.batch}: arm summaries ==")
    if args.batch == "ladder":
        arms = summarize(LADDER_ARMS)
        res = adjudicate_ladder(arms)
    elif args.batch == "g3":
        arms = summarize(G3_ARMS)
        res = adjudicate_g3(arms)
    elif args.batch == "g3re":
        arms = summarize(G3RE_ARMS)
        res = adjudicate_g3re(arms)
    elif args.batch == "t15":
        arms = summarize(T15_ARMS)
        res = adjudicate_t15(arms)
    elif args.batch == "stp":
        arms = summarize(STP_ARMS)
        res = adjudicate_stp(arms)
    elif args.batch == "fp":
        arms = summarize(FP_ARMS)
        res = adjudicate_fp(arms)
    elif args.batch == "dorm":
        arms = summarize(DORM_ARMS)
        res = adjudicate_dorm(arms)
    elif args.batch == "redoc1":
        arms = summarize(REDOC1_ARMS)
        res = adjudicate_redoc1(arms)
    elif args.batch == "ipcal":
        arms = summarize(IPCAL_ARMS)
        res = adjudicate_ipcal(arms)
    else:
        if args.h2 is None:
            p.error("k3 needs --h2 (the ladder's H)")
        arms = summarize(K3_ARMS)
        res = adjudicate_k3(arms, args.h2)

    out_path = REPO_ROOT / args.json_out
    prior = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    prior[args.batch] = {
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "composites"} | {"composites": v["composites"].tolist()}
                 for k, v in arms.items()},
        "adjudication": res,
    }
    out_path.write_text(json.dumps(prior, indent=1), encoding="utf-8")
    print(f"\n[wave1] scores + adjudication -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
