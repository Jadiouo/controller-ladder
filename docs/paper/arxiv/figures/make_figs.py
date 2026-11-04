#!/usr/bin/env python3
"""Redraw all paper figures and the generated Table 1 from the committed result files.

Reads only files under results/ and docs/paper/stats/equivalence.md. Every plotted number is parsed from
those files (each parse asserts a keyword on the source line, so a drifted file fails loudly). No simulation.

Run:  python3 make_figs.py        (needs numpy-free matplotlib only)
Out:  fig_testd.pdf, fig_ladder.pdf, fig_forest.pdf, fig_k4.pdf, tab_cross.tex (in this folder)
"""
from __future__ import annotations
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                      # repository root
RES = ROOT / "results"
EQ = ROOT / "docs" / "paper" / "stats" / "equivalence.md"

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "mathtext.fontset": "cm",
})
C_DARK, C_BLUE, C_ORANGE, C_GREY, C_RED = "#222222", "#0072B2", "#D55E00", "#888888", "#B2182B"

_cache: dict[Path, list[str]] = {}


def line(path: Path, n: int, must: str) -> str:
    if path not in _cache:
        _cache[path] = path.read_text(encoding="utf-8").splitlines()
    s = _cache[path][n - 1]
    assert must in s, f"{path.name}:{n} does not contain {must!r}: {s[:120]}"
    return s


NUM = r"[+-]?\d+(?:\.\d+)?"


def pct_ci(s: str):
    """'66.4 [62.2, 70.6]' -> (66.4, 62.2, 70.6)"""
    m = re.search(rf"({NUM})\s*\[({NUM}),\s*({NUM})\]", s)
    assert m, s
    return tuple(float(x) for x in m.groups())


def diff_ci(s: str):
    """'**-17.6 [-20.3, -14.9]**', '+5.6 pp [+0.6, +10.6]' or '| 0.9 | [-3.5, 5.4] |' -> (d, lo, hi)"""
    if "95% [" in s:                      # hypothesis-verdict rows: '... = +5.6 pp | 95% [+0.6, +10.6] | ...'
        d = re.search(rf"= ({NUM}) pp", s)
        ci = re.search(rf"95% \[({NUM}),\s*({NUM})\]", s)
        assert d and ci, s
        return float(d.group(1)), float(ci.group(1)), float(ci.group(2))
    m = re.search(rf"({NUM})\s*(?:pp)?\s*\|?\s*\[({NUM}),\s*({NUM})\]", s)
    assert m, s
    return tuple(float(x) for x in m.groups())


def cells(s: str):
    return [c.strip() for c in s.strip().strip("|").split("|")]


# ------------------------------------------------------------------ data
def testd():
    p = RES / "spike15e" / "table.md"
    rows = []
    for n, key, lab in [(20, "R0", "R0"), (22, "L-ens", "L-ens"), (24, "C-ens", "C-ens"), (23, "B ", "B"), (21, "R1", "R1")]:
        c = cells(line(p, n, key))
        m, lo, hi = pct_ci(c[1])
        rec = pct_ci(c[3]) if len(c) > 3 and c[3] else None
        rows.append((lab, m, lo, hi, rec))
    return rows


def ladder():
    p = RES / "spike15a" / "table.md"
    spec = [  # (line, keyword, label, group)
        (21, "R0", "R0 fitted $\\theta$ (T-adapt3-full)", "ladder"),
        (22, "R1", "R1 + true $\\theta$", "ladder"),
        (23, "R2", "R2 + true chain state", "ladder"),
        (24, "R4", "R4 oracle (no probe)", "ladder"),
        (25, "R5", "R5 canonical oracle", "ladder"),
        (26, "Rrev1", "Rrev1 true chain state only", "desc"),
        (27, "L-est-full (mean", "L-est-full (3 nets)", "est"),
        (31, "L-dec4 (3 seeds merged)", "L-dec4 (3 nets)", "est"),
        (35, "L-ens", "L-ens (mean $\\hat\\theta$)", "desc"),
        (36, "C-MAP", "C-MAP (classical)", "desc"),
    ]
    out = []
    for n, key, lab, grp in spec:
        c = cells(line(p, n, key))
        out.append((lab, *pct_ci(c[1]), grp))
    steps = {
        "R1-R0": diff_ci(line(p, 42, "R1 - R0")),
        "R2-R1": diff_ci(line(p, 43, "R2 - R1")),
        "R4-R2": diff_ci(line(p, 44, "R4 - R2")),
    }
    return out, steps


def eq_row(rid: str):
    for s in EQ.read_text(encoding="utf-8").splitlines():
        if s.startswith(f"| {rid} |"):
            c = cells(s)
            return c
    raise AssertionError(rid)


def eq_diff(rid: str):
    c = eq_row(rid)
    d = float(re.search(NUM, c[7]).group())
    lo, hi = (float(x) for x in re.findall(NUM, c[8])[:2])
    return d, lo, hi, int(c[4])


# forest: (id, label, (path, line, keyword), n, seeds, learned_flag, strongest_classical?)
SUCCESS = [
    ("S5", "S5 juggling: PPO vs T-robust", ("spike15b", 52, "holdout pooled"), 500, "HOLD", "degrades", True),
    ("S6", "S6 ideal whip: PPO vs T-adapt", ("spike15b", 18, "holdout pooled"), 500, "HOLD", "degrades", True),
    ("S6", "S6 ideal whip: L-est vs T-adapt", ("spike09b", 31, "L-est"), 50, "TEST", "estimator", True),
    ("S7", "S7 PPO vs R0", ("spike13a", 52, "conf_ppo on holdout"), 500, "HOLD", "not learned", True),
    ("S7", "S7 L-est-full vs R0", ("spike10b", 60, "L-est-full"), 500, "TEST-B", "estimator", True),
    ("S7", "S7 L-dec4 vs R0 (H14)", ("spike15a", 49, "L-dec4 (3 seeds merged)"), 500, "TEST-C", "estimator", True),
    ("S7", "S7 B vs C-ens (H16a)", ("spike15e", 9, "H16a"), 500, "TEST-D", "estimator + selection", True),
    ("S7", "S7 B vs R0 (weaker baseline)", ("spike15e", 8, "H15b"), 500, "TEST-D", "estimator + selection", False),
    ("S8", "S8 MIP D-canon vs T-adapt3-full", ("spike12a", 53, "MIP"), 200, "NEW", "BC student", True),
]
TRACK = [  # equivalence.md row ids; sign: positive = learned better (relative RMSE, %)
    ("S1", "S1 PPO vs C1 (W0 clean)", "S1-W0-clean", "TEST", "RL, 5 seeds"),
    ("S2", "S2 PPO vs C1-new (noise+delay, clean)", "S2-N1-clean", "TEST", "RL, 5 seeds"),
    ("S2", "S2 PPO vs C1-new (noise+delay, disturbed)", "S2-N1-dist", "TEST", "RL, 5 seeds"),
    ("S3", "S3 BC student vs teacher", "S3-bc", "HOLD", "imitation"),
    ("S3", "S3 flow-1 student vs teacher", "S3-flow1", "HOLD", "imitation"),
    ("S3", "S3 DDIM-10 student vs teacher", "S3-diffusion", "HOLD", "imitation"),
]


def forest_data():
    s = []
    for sid, lab, (d, n, key), nn, seeds, flag, strong in SUCCESS:
        s.append((sid, lab, *diff_ci(line(RES / d / "table.md", n, key)), nn, seeds, flag, strong))
    t = []
    for sid, lab, rid, seeds, flag in TRACK:
        d, lo, hi, nn = eq_diff(rid)
        t.append((sid, lab, d, lo, hi, nn, seeds, flag, True))
    return s, t


# ------------------------------------------------------------------ figures
def fig_testd():
    rows = testd()
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    cols = {"R0": C_GREY, "L-ens": C_GREY, "C-ens": C_ORANGE, "B": C_BLUE, "R1": C_DARK}
    names = {"R0": "R0\nfitted", "L-ens": "L-ens\nno robust\nselection", "C-ens": "C-ens\nclassical\nensemble", "B": "B\nlearned\nensemble", "R1": "R1\ntrue $\\theta$"}
    for i, (lab, m, lo, hi, rec) in enumerate(rows):
        ax.bar(i, m, color=cols[lab], width=0.62, zorder=2)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="black", capsize=2.5, lw=0.9, zorder=3)
        ax.text(i, 51.0, f"{m:.1f}", ha="center", va="bottom", color="white", fontsize=7, zorder=4)
        if rec:
            ax.text(i, hi + 1.0, f"{rec[0]:.2f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([names[r[0]] for r in rows])
    ax.set_ylim(50, 94)
    ax.set_ylabel("success on TEST-D (%), n=500")
    ax.grid(axis="y", lw=0.3, zorder=0)
    ax.text(0.01, 0.99, "numbers above bars: recovered fraction of R1$-$R0", transform=ax.transAxes, va="top", fontsize=6.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_testd.pdf")
    plt.close(fig)


def fig_ladder():
    rows, steps = ladder()
    fig, ax = plt.subplots(figsize=(3.45, 3.1))
    y = list(range(len(rows)))[::-1]
    style = {"ladder": (C_DARK, "o"), "est": (C_BLUE, "s"), "desc": (C_ORANGE, "D")}
    for yy, (lab, m, lo, hi, grp) in zip(y, rows):
        c, mk = style[grp]
        ax.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt=mk, color=c, mfc=("white" if grp == "desc" else c), ms=4, capsize=2, lw=0.9)
        ax.text(max(hi, m) + 0.8, yy, f"{m:.1f}", va="center", fontsize=6.5, zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.6))
    r0 = rows[0][1]
    r1 = rows[1][1]
    ax.axvline(r0, color=C_GREY, ls=":", lw=0.8, zorder=1)
    ax.axvline(r1, color=C_GREY, ls=":", lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("TEST-C success (%), n=500")
    ax.set_xlim(58, 102)
    ax.grid(axis="x", lw=0.3)
    d = steps["R1-R0"]
    ax.text(0.375, 0.74, f"R1$-$R0 = {d[0]:+.1f} pp\n[{d[1]:+.1f}, {d[2]:+.1f}]\n(dotted: R0, R1)", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.3, zorder=7, bbox=dict(fc="white", ec="none", pad=1.0))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=C_DARK, ls="", ms=4, label="ladder"),
         Line2D([], [], marker="s", color=C_BLUE, ls="", ms=4, label="learned est."),
         Line2D([], [], marker="D", color=C_ORANGE, mfc="white", ls="", ms=4, label="post hoc")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.4, -0.17), ncol=3, frameon=False, columnspacing=0.8, handletextpad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_ladder.pdf")
    plt.close(fig)


def fig_forest():
    s, t = forest_data()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.85), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, data, unit in [(a, s, "learned $-$ classical success (pp)"), (b, t, "relative RMSE change (%)")]:
        yy = list(range(len(data)))[::-1]
        for y, (sid, lab, d, lo, hi, n, seeds, flag, strong) in zip(yy, data):
            c = C_BLUE if strong else C_ORANGE
            ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=c, mfc=(c if strong else "white"), ms=4, capsize=2, lw=0.9)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_yticks(yy)
        ax.set_yticklabels([r[1] for r in data])
        ax.set_xlabel(unit)
        ax.grid(axis="x", lw=0.3)
    a.axvspan(-10, 10, color=C_GREY, alpha=0.12, lw=0)
    a.axvline(10, color=C_RED, ls="--", lw=0.8)
    a.text(10.3, len(s) - 0.55, "+10 pp bar", color=C_RED, fontsize=6.3, va="top")
    b.axvspan(-10, 10, color=C_GREY, alpha=0.12, lw=0)
    a.set_title("(a) success-rate settings", loc="left")
    b.set_title("(b) tracking RMSE", loc="left")
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "fig_forest.pdf")
    plt.close(fig)
    return s, t


def fig_k4():
    p = RES / "spike12a" / "table.md"
    names, multi, canon = [], [], []
    for n, key in [(9, "MLP"), (10, "MIP"), (11, "Diffusion"), (12, "Flow 10"), (13, "Flow 1")]:
        c = cells(line(p, n, key))
        names.append(c[0].replace(" (BC)", "").replace(" (2-step regression)", ""))
        multi.append(pct_ci(c[2]))
        canon.append(pct_ci(c[3]))
    base = {}
    for n, key in [(14, "oracle_canon"), (15, "adapt3full"), (16, "nom")]:
        c = cells(line(p, n, key))
        base[key] = pct_ci(c[2])
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    w = 0.36
    for i in range(5):
        for k, (data, col, lab) in enumerate([(multi, C_GREY, "D-multi (standard teacher)"), (canon, C_BLUE, "D-canon (canonical teacher)")]):
            m, lo, hi = data[i]
            x = i + (k - 0.5) * w
            ax.bar(x, m, width=w, color=col, label=lab if i == 0 else None, zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="black", capsize=1.8, lw=0.8, zorder=3)
    ax.axhline(base["adapt3full"][0], color=C_ORANGE, ls="--", lw=0.9)
    ax.text(4.55, base["adapt3full"][0] + 2.2, "T-adapt3-full", color=C_ORANGE, ha="right", va="bottom", fontsize=6.5,
            bbox=dict(fc="white", ec="none", pad=0.8), zorder=5)
    ax.set_xticks(range(5))
    ax.set_xticklabels([n.replace("Diffusion DDIM-10", "DDIM-10").replace("MLP regression", "MLP").replace("Flow 10 steps", "Flow-10").replace("Flow 1 step", "Flow-1") for n in names], rotation=0)
    ax.set_ylabel("success on NEW seeds (%), n=200")
    ax.set_ylim(0, 92)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0).set_zorder(10)
    ax.grid(axis="y", lw=0.3, zorder=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_k4.pdf")
    plt.close(fig)


def fig_protocol():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    W, H = 7.1, 2.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, txt, fc, fs=6.6, ls=1.25):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec="#444444", lw=0.7))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, linespacing=ls)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                     color="#444444", shrinkA=0, shrinkB=0))

    # top row: seed-set flow
    ty, th = 1.55, 1.0
    ws = [1.05, 1.35, 1.2, 1.7, 1.1]
    gap = (W - 0.1 - sum(ws)) / 4
    xs = [0.05]
    for w in ws[:-1]:
        xs.append(xs[-1] + w + gap)
    cy = ty + th / 2
    ax.text(0.05, ty + th + 0.12, "Seed sets (tune \u2192 select \u2192 confirm \u2192 TEST sets)", fontsize=6.8,
            ha="left", va="center", fontweight="bold")
    box(xs[0], ty, ws[0], th, "CAL\n1000-1049\n(+2000-2149)\ntune, select, gate", "#DCEBF7")
    box(xs[1], ty, ws[1], th, "HOLD\n1100-1599\nclassical selection\n(reused, optimistic)", "#DCEBF7")
    box(xs[2], ty, ws[2], th, "NEW\n700000+, n=200\nfresh\nconfirmation", "#DCEBF7")
    box(xs[3], ty, ws[3], th, "TEST sets\n0-49 (older spikes, reused)\nlocked, each used once:\nB  20000-20499\nC  30000-30499\nD  40000-40499", "#FBE5D6", fs=6.0, ls=1.12)
    box(xs[4], ty, ws[4], th, "access log: one line\nper arena evaluation\non TEST seeds", "#E8F3E1", fs=6.0)
    for i in range(3):
        arrow(xs[i] + ws[i], cy, xs[i + 1], cy)
    ax.add_patch(FancyArrowPatch((xs[3] + ws[3], cy), (xs[4], cy), arrowstyle="-", lw=0.8, ls=(0, (2, 2)),
                                 color="#444444", shrinkA=0, shrinkB=0))

    # bottom row: three rules
    by, bh = 0.08, 1.0
    bw = (W - 0.1 - 2 * 0.18) / 3
    ax.text(0.05, by + bh + 0.14, "Rules governing every comparison", fontsize=6.8, ha="left", va="center", fontweight="bold")
    box(0.05, by, bw, bh, "three budget axes per method\nepisodes | wall-clock | label source\n(declared cap 2x; over-runs disclosed)", "#FFF4CC")
    box(0.05 + bw + 0.18, by, bw, bh, "sanity gate before a null is read\nconstant-bias correction: +10 pp over\ninit, CI > 0; learned = +5 pp on CAL", "#FFF4CC")
    box(0.05 + 2 * (bw + 0.18), by, bw, bh, "pre-specified cards (time-stamped)\nwin: CI excludes 0 and >= +10 pp\nPRE-git / PRE-README / POST tags", "#FFF4CC")
    fig.savefig(HERE / "fig_protocol.pdf")
    plt.close(fig)


def sgn(x, d=1):
    return f"{x:+.{d}f}".replace("-", "$-$")


def tab_cross(s, t):
    rows = []
    rows.append(r"\begin{tabular}{@{}llrlrl@{}}\toprule")
    rows.append(r"Set. & Learned vs.\ classical & $n$ & Seeds & $\Delta$ [95\% CI] & Learner \\ \midrule")
    for sid, lab, d, lo, hi, n, seeds, flag, strong in s:
        l = lab.split(": ", 1)[-1] if ": " in lab else lab
        l = l.replace("S7 ", "").replace("S8 ", "").replace("&", r"\&")
        note = "" if strong else r"$^\dagger$"
        rows.append(rf"{sid} & {l}{note} & {n} & {seeds} & {sgn(d)} [{sgn(lo)}, {sgn(hi)}] & {flag} \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{6}{@{}l}{Tracking, relative RMSE change in \% (positive = learned better)} \\")
    for sid, lab, d, lo, hi, n, seeds, flag, strong in t:
        l = lab.split(" ", 1)[1]
        rows.append(rf"{sid} & {l} & {n} & {seeds} & {sgn(d)} [{sgn(lo)}, {sgn(hi)}] & {flag} \\")
    rows.append(r"\bottomrule\end{tabular}")
    (HERE / "tab_cross.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    fig_protocol()
    fig_testd()
    fig_ladder()
    s, t = fig_forest()
    fig_k4()
    tab_cross(s, t)
    print("ok:", *(p.name for p in sorted(HERE.glob("fig_*.pdf"))), "tab_cross.tex")
