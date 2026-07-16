"""Generate the paper's figures from the committed result numbers.

Sources of truth: docs/research-notes/0006 (control ladder, March contrast) and
docs/research-notes/0007 §Results (LOOP-0009 replication). Re-run after any number changes:

    myenv\\Scripts\\python.exe paper/figures/make_figures.py

Outputs fig1_ladder.{pdf,png} and fig2_replication.{pdf,png} next to this script.
Palette: Okabe-Ito blue/vermillion (categorical, CVD-validated) + recessive grays for context.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

BLUE = "#0072B2"       # series: trained / this work
VERMILLION = "#D55E00" # series: the negative draw (seed 3)
INK = "#333333"        # primary text
MUTED = "#8a8a8a"      # context marks (March campaign), reference lines
BAR_GRAY = "#b8bfc6"   # de-emphasized ladder rungs

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "pdf.fonttype": 42,   # TrueType: camera-ready portals require embedded fonts
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
})


def fig1_ladder() -> None:
    """Control ladder (box protocol, n=8/arm): composite per rung, ceiling emphasized."""
    rungs = [
        ("O2: zero-forgetting restore\n(full-learner snapshot oracle)", 0.8022, BLUE),
        ("oracle code\n(ground-truth regime id → mask)", 0.5481, BAR_GRAY),
        ("O1: oracle-timed critic damp", 0.5451, BAR_GRAY),
        ("untrained Brain (control)", 0.5534, BAR_GRAY),
        ("D0: no controller,\ntuned static HPs", 0.5608, BAR_GRAY),
    ]
    labels = [r[0] for r in rungs]
    vals = [r[1] for r in rungs]
    colors = [r[2] for r in rungs]

    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    y = range(len(rungs))
    ax.barh(y, vals, height=0.62, color=colors, zorder=3)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("composite (post-switch window success)")
    ax.xaxis.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)

    # The +0.249 headroom bracket: control -> O2
    ax.annotate(
        "", xy=(0.8022, 4.62), xytext=(0.5534, 4.62),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9),
        annotation_clip=False,
    )
    ax.text(0.678, 4.98, "+0.249 headroom", ha="center", fontsize=8, color=INK,
            clip_on=False)
    ax.set_ylim(-0.55, 4.55)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_ladder.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_ladder.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig2_replication() -> None:
    """(a) per-Brain paired init->trained composites; (b) pooled delta vs eval n."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.6, 2.6), width_ratios=[1.0, 1.05])

    # --- (a) paired slopes, n=16/arm (March: n=32, prior campaign, shown for context)
    seeds = [
        ("seed 1", 0.5225, 0.6107, BLUE),
        ("seed 2", 0.5274, 0.5724, BLUE),
        ("seed 4", 0.5197, 0.5955, BLUE),
        ("seed 3", 0.5196, 0.4926, VERMILLION),
    ]
    march = ("March '26", 0.5285, 0.5577)

    for name, a, b, c in seeds:
        ax_a.plot([0, 1], [a, b], color=c, lw=1.6, marker="o", ms=4, zorder=3)
        ax_a.text(1.06, b, name, va="center", fontsize=7.5, color=c)
    ax_a.plot([0, 1], [march[1], march[2]], color=MUTED, lw=1.3, ls="--",
              marker="s", ms=3.5, zorder=2)
    ax_a.text(1.06, march[2] - 0.004, march[0], va="center", fontsize=7.5, color=MUTED)

    ax_a.set_xticks([0, 1], ["random init", "trained ep130"])
    ax_a.set_xlim(-0.25, 1.75)
    ax_a.set_ylabel("composite")
    ax_a.yaxis.grid(True, color="#e6e6e6", linewidth=0.6)
    ax_a.set_axisbelow(True)
    ax_a.set_title("(a) matched contrast per Brain", loc="left", fontsize=8.5)

    # --- (b) pooled delta vs n: converges, does not decay
    n9, d9 = [8, 16], [0.0537, 0.0455]
    nm, dm = [8, 16, 32], [0.046, 0.0275, 0.0292]
    ax_b.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax_b.plot(n9, d9, color=BLUE, lw=1.6, marker="o", ms=4.5, label="this work (4 Brains, pooled)")
    ax_b.plot(nm, dm, color=MUTED, lw=1.3, ls="--", marker="s", ms=3.5,
              label="March '26 Brain (1 seed)")
    ax_b.text(n9[-1], d9[-1] + 0.004, f"+{d9[-1]:.4f}\n(p=1.3e-6)", fontsize=7.5,
              color=BLUE, ha="center", va="bottom")
    ax_b.text(nm[-1], dm[-1] - 0.006, f"+{dm[-1]:.4f}", fontsize=7.5, color=MUTED,
              ha="center", va="top")

    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks([8, 16, 32], ["8", "16", "32"])
    ax_b.set_xlabel("evaluation seeds per arm (n)")
    ax_b.set_ylabel(r"$\Delta$ composite (trained $-$ init)")
    ax_b.set_ylim(-0.015, 0.075)
    ax_b.yaxis.grid(True, color="#e6e6e6", linewidth=0.6)
    ax_b.set_axisbelow(True)
    ax_b.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax_b.set_title("(b) estimate converges with n", loc="left", fontsize=8.5)

    fig.tight_layout(w_pad=2.6)
    fig.savefig(OUT / "fig2_replication.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_replication.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig3_memory() -> None:
    """(a) swap-scope decomposition of the ceiling; (b) the de-oracling ladder.
    Sources: docs/research-notes/0012 (decomposition) and 0013 (method + probes),
    eval protocol, control = archived loop9_s1_model arm."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.6, 2.5), width_ratios=[1.0, 1.0])

    # --- (a) decomposition (n=8/arm vs matched control 0.6202)
    scopes = [
        ("full snapshot\n(original O2)", 0.8636, BLUE),
        ("heads + encoder", 0.8546, BLUE),
        ("heads only", 0.8383, BLUE),
        ("world model only", 0.6003, BAR_GRAY),
        ("none (control)", 0.6202, BAR_GRAY),
    ]
    y = range(len(scopes))
    ax_a.barh(y, [s[1] for s in scopes], height=0.62,
              color=[s[2] for s in scopes], zorder=3)
    ax_a.set_yticks(list(y), [s[0] for s in scopes])
    ax_a.axvline(0.6202, color=MUTED, lw=0.8, ls=":")
    ax_a.set_xlim(0.5, 0.95)
    ax_a.set_xlabel("composite (eval protocol)")
    ax_a.xaxis.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    for yi, (_, v, _c) in zip(y, scopes):
        ax_a.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    ax_a.set_title("(a) the ceiling is policy-head memory", loc="left", fontsize=8.5)

    # --- (b) de-oracling: what a learned trigger collects (share of heads slice)
    arms = [
        ("oracle when + which\n(bank, equivalence rung)", 0.8409, BAR_GRAY),
        ("learned when, K=2 flip\n(value-loss trigger, n=16)", 0.7295, BLUE),
        ("faster learned when\n(per-step, churn-bound)", 0.7023, VERMILLION),
        ("learned which\n(content selection, drift-bound)", 0.6218, VERMILLION),
        ("none (control, n=16)", 0.6107, BAR_GRAY),
    ]
    y = range(len(arms))
    ax_b.barh(y, [a[1] for a in arms], height=0.62,
              color=[a[2] for a in arms], zorder=3)
    ax_b.set_yticks(list(y), [a[0] for a in arms])
    ax_b.axvline(0.6107, color=MUTED, lw=0.8, ls=":")
    ax_b.set_xlim(0.5, 0.95)
    ax_b.set_xlabel("composite (eval protocol)")
    ax_b.xaxis.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    for yi, (_, v, _c) in zip(y, arms):
        ax_b.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    ax_b.set_title("(b) a learned trigger collects 52% of it", loc="left", fontsize=8.5)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT / "fig3_memory.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_memory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_ladder()
    fig2_replication()
    fig3_memory()
    print(f"wrote figures to {OUT}")
