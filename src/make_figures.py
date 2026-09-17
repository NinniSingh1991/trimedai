"""Every figure in the article, drawn from the result files.

    python make_figures.py              # all twelve
    python make_figures.py frontier     # one of them, by function name

Figure 1   the framework, and the order of events within one interaction
Figure 2   the same ordering in detail, and what a fixed threshold costs
Figure 3   harm against caregiver burden, as the planning discount varies
Figure 4   value delivered against the same burden
Figure 5   the contribution of each stage, and of the backup rule
Figure 6   regret, and the window in which the initialisation acts
Figure 7   normalised score on each latent setting
Figure 8   behaviour when the person is not one of the settings given
Figure 9   learning curves, with the dispersion across seeds
Figure 10  the severity at which a caregiver is summoned, by context
Figure 11  when the simulator leaves the agent's model class
Figure 12  error in the deterioration signal, and in the assumed urgency

Every panel uses the floor-anchored score of equation (39), and every reference
quantity is the average-reward optimal policy of triage_env.solve_average.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import RESULTS, FIGURES
from triage_env import (CORE_NAMES, PROFILES, REQUESTS, N_SEV, CRISIS, ALERT,
                        sid, solve_average, myopic_policy, news2_policy)

plt.rcParams.update({"font.family": "serif", "font.size": 8.5,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 220, "savefig.bbox": "tight"})

NAVY, GREY, RED, GREEN, PURPLE, BLUE, GOLD = (
    "#1f4e79", "#7f7f7f", "#c00000", "#548235", "#7030a0", "#4472c4", "#bf9000")


def L(name):
    return json.load(open(os.path.join(RESULTS, name), encoding="utf-8"))


R = L("har_results.json")
F = L("har_frontier.json")
E = L("har_extended.json")
O = L("har_onboarding.json")


def M(k, f):
    return float(np.mean(R["main"][k][f]))


def NORM(k):
    return R["normalised"][k]["mean"]


LAB = {"CNN": "CNN + population rule", "LSTM": "LSTM + population rule",
       "RNN": "RNN + population rule", "MCDM": "Multi-criteria rule",
       "NEWS2": "Fixed-threshold rule", "RLHF": "RLHF (pooled)",
       "QMDP": "QMDP", "TriMedAI": "TriMedAI (proposed)",
       "PosteriorSampling": "Posterior sampling", "Bandit": "Per-user bandit",
       "PerUserQ": "Per-user Q-learning", "GAI": "Generative baseline",
       "Random": "Floor (random)", "ProfileKnown": "TriMedAI, latent given"}
COL = {"TriMedAI": NAVY, "QMDP": BLUE, "NEWS2": PURPLE, "RLHF": RED,
       "MCDM": GOLD, "CNN": GREEN, "LSTM": "#78a353", "RNN": "#9dc37e",
       "PosteriorSampling": "#2e75b6", "Bandit": "#a6a6a6",
       "PerUserQ": "#808080", "GAI": "#d9d9d9", "Random": "#404040",
       "ProfileKnown": "#7ea6cc"}


TITLE_FS, BODY_FS = 8.6, 6.6
LINE_H = 3.05            # vertical units taken by one line of body text
PAD_TOP, PAD_BOT = 4.6, 1.8


def architecture():
    """One figure in place of the four the first submission used.

    Box heights are computed from the number of lines they hold, so nothing
    overflows; every line is kept short enough to sit inside its box at the body
    font size.
    """
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 66)
    ax.axis("off")
    ax.grid(False)

    def box(x, y_top, w, title, lines, fc="#eef3f8", ec=NAVY):
        h = PAD_TOP + LINE_H * len(lines) + PAD_BOT
        y = y_top - h
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.35,rounding_size=1.1",
                                    fc=fc, ec=ec, lw=1.05, zorder=2))
        ax.text(x + w / 2, y_top - 1.5, title, ha="center", va="top",
                fontsize=TITLE_FS, fontweight="bold", color=ec, zorder=3)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y_top - PAD_TOP - LINE_H * (i + 0.72), ln,
                    ha="center", va="center", fontsize=BODY_FS, color="#222222",
                    zorder=3)
        return y

    def arrow(p1, p2, label=None, color="#333333", rad=0.0, off=(0, 1.2)):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>",
                                     connectionstyle="arc3,rad=%s" % rad,
                                     mutation_scale=9, lw=0.95, color=color,
                                     zorder=4))
        if label:
            ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1],
                    label, ha="center", va="bottom", fontsize=6.6, color=color,
                    zorder=5)

    # ---- row 1: what is sensed, and who is being served --------------------
    b1 = box(0.5, 65, 21, "Wearable sensing",
             ["waist-worn inertial signal",
              "9 channels at 50 Hz",
              "2.56 s windows, 128 samples",
              "UCI HAR, subject-wise split"],
             fc="#f2f2f2", ec=GREY)
    b2 = box(24, 65, 21, "Context recognition",
             ["1-D convolution over time",
              "six activities of daily living",
              "returns the recognised context,",
              "which may be wrong"],
             fc="#f2f2f2", ec=GREY)
    b3 = box(48, 65, 51.5, "The person: never observed directly",
             ["latent = (frailty, one weight per delivery channel)",
              "observed each interaction: the channel the person",
              "indicates they would have preferred, and the",
              "deterioration flag reported by the monitoring layer",
              "carried state: severity 0 to 4, which persists"],
             fc="#fdf5e6", ec="#bf9000")

    arrow((21.5, 57), (24, 57))

    # ---- row 2: the three stages -------------------------------------------
    top2 = 40
    box(0.5, top2, 30, "Stage 1   Reward inference",
        ["particle filter over the continuous",
         "latent, 256 particles, eq. (20)",
         "preference likelihood, eq. (21)",
         "deterioration likelihood, eq. (22)",
         "returns the posterior over the latent"])
    box(34, top2, 31, "Stage 2   Recommendation",
        ["builds the belief-implied model, eq. (23)",
         "solves it by value iteration, eq. (24)-(25)",
         "warm-started, re-solved every 10 steps",
         "plans at the 0.75 frailty quantile,",
         "which errs on the cautious side"])
    box(68.5, top2, 31, "Stage 3   Adaptation",
        ["latents drawn from the agent's prior,",
         "each solved exactly, then averaged by",
         "the first-order outer step, eq. (8)-(9)",
         "computed offline, before deployment;",
         "serves the user until the posterior forms"])

    arrow((30.5, 30), (34, 30), "posterior", off=(0, 0.9))
    arrow((68.5, 26), (65, 26), None, color=PURPLE)
    ax.text(66.7, 24.0, "warm start", ha="center", va="top",
            fontsize=6.4, color=PURPLE, zorder=5)
    arrow((36, b2), (18, 40.2))
    arrow((58, b3), (24, 40.2), rad=0.10)
    ax.text(31.5, 44.6, "recognised context", fontsize=6.4, color="#333333")
    ax.text(31.5, 42.2, "stated channel, deterioration flag", fontsize=6.4,
            color="#333333")

    # ---- row 3: the order of events within one interaction -----------------
    box(0.5, 14.5, 99, "Order of events within one interaction, eq. (15) to (17)",
        ["1.  the condition develops first: severity rises by one with probability "
         "frailty x context urgency",
         "2.  the chosen action takes effect afterwards, so a caregiver summoned "
         "late cannot avert a deterioration already under way",
         "3.  the reward of eq. (19) is received, the two signals are observed, "
         "the belief is revised and the policy re-solved when due",
         "the value of summoning a caregiver is therefore carried entirely by the "
         "severity the person is left in"],
        fc="#eef6ee", ec=GREEN)
    for x in (15, 49, 84):
        arrow((x, 18.3), (x, 14.8))

    fig.savefig(os.path.join(FIGURES, "r01_architecture.png"))
    plt.close(fig)
    print("wrote r01_architecture.png")



# ------------------------------------------------- Figure 2, the ordering ---
def timing():
    fig = plt.figure(figsize=(9.4, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.24)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, lines, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.5,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.0, zorder=2))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top",
                fontsize=7.8, fontweight="bold", color=ec)
        ax.text(x + w / 2, y + h - 6.4, "\n".join(lines), ha="center", va="top",
                fontsize=6.2, color="#222222", linespacing=1.45)

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.9, color="#333333"))

    box(0.5, 30, 47, 14, "1.  the condition develops",
        ["severity rises by one with", "probability frailty × urgency,",
         "eq. (14)–(15); this happens first"], "#fdecea", RED)
    box(52.5, 30, 47, 14, "2.  the action takes effect",
        ["a caregiver summoned now has", "still to travel, so a deterioration",
         "already under way is not averted"], "#eaf1fa", NAVY)
    box(0.5, 13, 47, 14, "3.  reward and observations",
        ["reward eq. (19); the stated channel", "and the deterioration flag are",
         "observed, eq. (21)–(22)"], "#eef6ee", GREEN)
    box(52.5, 13, 47, 14, "4.  belief and policy",
        ["every action at the visited state is", "backed up from the inferred model,",
         "not only the action taken"], "#f7f0fa", PURPLE)
    arr(47.5, 37, 52.5, 37); arr(76, 30, 76, 27); arr(47.5, 20, 52.5, 20)
    ax.text(50, 6.0, "the value of escalating is carried entirely by the severity "
                     "the person is left in",
            ha="center", fontsize=6.9, style="italic", color="#444444")

    ax2 = fig.add_subplot(gs[1])
    names = [p.replace(" impaired", "") for p in CORE_NAMES]
    xp = np.arange(len(names))
    # computed once and written out, so that the figure and the sentence in
    # section 3.4.1 that quotes these rates read from the same source
    store = {"profiles": list(CORE_NAMES), "seeds": 5, "n_users": 300,
             "steps_per_user": 220, "by_threshold": {}}
    for k, (lab, col, key) in enumerate((("escalate at 3", RED, 3),
                                         ("escalate at 2", GOLD, 2),
                                         ("never escalate", GREY, 9))):
        vals = _fixed_threshold_crises(key)
        store["by_threshold"][lab] = dict(zip(CORE_NAMES, vals))
        ax2.bar(xp + (k - 1) * 0.26, vals, 0.26, color=col, label=lab)
        for xi, v in zip(xp, vals):
            if v < 0.05:
                ax2.annotate("0.00", (xi + (k - 1) * 0.26, 0.04),
                             ha="center", va="bottom", fontsize=6.0,
                             color=col, rotation=90)
    json.dump(store, open(os.path.join(RESULTS, "har_thresholds.json"), "w"),
              indent=1)
    ax2.set_xticks(xp)
    ax2.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=6.6)
    ax2.set_ylabel("Deteriorations per 100 interactions", fontsize=7.6)
    ax2.legend(fontsize=6.4, loc="upper left")
    ax2.set_title("What a fixed escalation threshold costs", fontsize=8.2)
    fig.savefig(os.path.join(FIGURES, "r02_timing.png"))
    plt.close(fig)
    print("wrote r02_timing.png")


def _fixed_threshold_crises(threshold, seeds=5, users=300, steps=220):
    """crisis rate under a policy that escalates at a fixed severity"""
    from triage_env import step, ACTIONS, AFFINITY, ASK
    out = []
    for pf in CORE_NAMES:
        tot, n = 0, 0
        for s in range(seeds):
            rng = np.random.default_rng(4000 + s)
            for _ in range(users):
                sev = 0
                for t in range(steps):
                    rq = int(rng.integers(len(REQUESTS)))
                    if sev >= threshold:
                        a = ALERT
                    else:
                        vals = [AFFINITY[rq].get(ACTIONS[i], 0.0)
                                if i not in (ASK, ALERT) else -9.0
                                for i in range(len(ACTIONS))]
                        a = int(np.argmax(vals))
                    _, sev, cri, _ = step(pf, rq, sev, a, rng)
                    tot += int(cri); n += 1
        out.append(100.0 * tot / n)
    return out


# ------------------------------------- Figure 3, harm against burden --------
def frontier():
    """Two panels: the whole plane, and the corner every good method occupies."""
    fig, (ax, az) = plt.subplots(1, 2, figsize=(8.4, 3.7))
    gammas = F["gammas"]
    xs = [F["points"][str(g)]["TriMedAI"]["alerts"] for g in gammas]
    ys = [F["points"][str(g)]["TriMedAI"]["crisis"] for g in gammas]
    others = ("RLHF", "MCDM", "CNN", "PosteriorSampling", "QMDP")

    for a, zoom in ((ax, False), (az, True)):
        a.plot(xs, ys, "-o", color=NAVY, lw=1.8, ms=5, zorder=6,
               label="TriMedAI, as the planning discount varies")
        for g, x, y in zip(gammas, xs, ys):
            if (zoom and g < 0.9) or (not zoom and g > 0.9):
                continue
            a.annotate("γ = %g" % g, (x, y), fontsize=6.3, color=NAVY,
                       xytext=(4, 4), textcoords="offset points")
        a.scatter([F["Oracle"]["alerts"]], [F["Oracle"]["crisis"]], marker="*",
                  s=190, color="black", zorder=7, label="Reference policy")
        a.scatter([F["Fixed threshold"]["alerts"]],
                  [F["Fixed threshold"]["crisis"]], marker="s", s=70,
                  color=PURPLE, zorder=7, label="Fixed-threshold rule")
        for k in others:
            a.scatter([M(k, "alerts_per_100")], [M(k, "crisis_per_100")], s=42,
                      color=COL[k], label=LAB[k], zorder=5)
        a.axhline(F["Oracle"]["crisis"], color="black", ls=":", lw=0.9)
        a.set_xlabel("Caregiver call-outs per 100 interactions", fontsize=8)
    ax.set_ylabel("Deteriorations per 100 interactions", fontsize=8)
    ax.set_title("Every method evaluated", fontsize=8.6)
    az.set_title("The corner the useful methods occupy", fontsize=8.6)
    az.set_xlim(7.2, 10.9)
    az.set_ylim(-0.07, 0.78)
    ax.legend(fontsize=6.1, loc="upper right", framealpha=0.94)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "r03_frontier.png"))
    plt.close(fig)
    print("wrote r03_frontier.png")


# ------------------------------- Figure 4, value against the same burden ----
def value_burden():
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    gammas = F["gammas"]
    xs = [F["points"][str(g)]["TriMedAI"]["alerts"] for g in gammas]
    ys = [F["points"][str(g)]["TriMedAI"]["norm"] for g in gammas]
    ax.plot(xs, ys, "-o", color=NAVY, lw=1.8, ms=5, zorder=6,
            label="TriMedAI, as the planning discount varies")
    ax.scatter([F["Fixed threshold"]["alerts"]], [F["Fixed threshold"]["norm"]],
               marker="s", s=70, color=PURPLE, zorder=7,
               label="Fixed-threshold rule")
    ax.scatter([F["Oracle"]["alerts"]], [100.0], marker="*", s=190, color="black",
               zorder=7, label="Reference policy")
    for k in ("RLHF", "MCDM", "CNN", "PosteriorSampling", "QMDP"):
        ax.scatter([M(k, "alerts_per_100")], [NORM(k)], s=42, color=COL[k],
                   label=LAB[k], zorder=5)
    gap = NORM("TriMedAI") - NORM("NEWS2")
    ax.annotate("", xy=(F["Fixed threshold"]["alerts"], NORM("TriMedAI")),
                xytext=(F["Fixed threshold"]["alerts"], NORM("NEWS2")),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.1))
    ax.text(F["Fixed threshold"]["alerts"] - 0.15,
            0.5 * (NORM("TriMedAI") + NORM("NEWS2")),
            "%.1f points" % gap, color=RED, fontsize=7, ha="right", va="center")
    ax.axhline(100, color="black", ls=":", lw=0.9)
    ax.set_xlabel("Caregiver call-outs per 100 interactions")
    ax.set_ylabel("Normalised score (floor 0, reference 100)")
    ax.set_title("Value delivered against the same burden", fontsize=8.6)
    ax.legend(fontsize=6.3, loc="lower left", framealpha=0.94)
    fig.savefig(os.path.join(FIGURES, "r04_value.png"))
    plt.close(fig)
    print("wrote r04_value.png")


# ---------------------------- Figure 6, regret and the onboarding window ----
def regret_onboarding():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    order = ["CNN", "MCDM", "RLHF", "NEWS2", "QMDP", "TriMedAI"]
    reg = [M(k, "regret_per_100") for k in order]
    a1.barh(np.arange(len(order)), reg, 0.6, color=[COL[k] for k in order])
    a1.set_yticks(np.arange(len(order)))
    a1.set_yticklabels([LAB[k] for k in order], fontsize=7)
    a1.set_xlabel("Regret per 100 interactions (primary endpoint)")
    for i, v in enumerate(reg):
        a1.annotate("%.2f" % v, (v, i), va="center", ha="left", fontsize=7,
                    xytext=(3, 0), textcoords="offset points")
    a1.set_xlim(0, max(reg) * 1.22)

    rows = O["rows"]
    xw = np.arange(len(rows))
    a2.bar(xw - 0.2, [r["crisis_with"] for r in rows], 0.4, color=NAVY,
           label="with meta-initialisation")
    a2.bar(xw + 0.2, [r["crisis_without"] for r in rows], 0.4, color=RED,
           label="without")
    for i, r in enumerate(rows):
        if r["p_crisis"] < 0.01:
            a2.annotate("p < 0.01",
                        (i, max(r["crisis_with"], r["crisis_without"])),
                        ha="center", va="bottom", fontsize=6.2)
    a2.set_xticks(xw)
    a2.set_xticklabels([r["window"] for r in rows], fontsize=7)
    a2.set_xlabel("Interaction window")
    a2.set_ylabel("Deteriorations per 100 interactions", fontsize=7.6)
    a2.legend(fontsize=6.5, loc="upper right")
    a2.set_title("The stage acts during onboarding and nowhere else", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "r06_regret_onboarding.png"))
    plt.close(fig)
    print("wrote r06_regret_onboarding.png")


# ------------------------------- Figure 7, score on each latent setting -----
def dispersion():
    core, held = E["core_profiles"], E["held_out_profiles"]

    def nz(tag, m, key):
        fl = E[tag]["Random"][key]["reward"]
        oc = E[tag]["Oracle"][key]["reward"]
        return 100.0 * (E[tag][m][key]["reward"] - fl) / (oc - fl)

    meths = ["CNN", "NEWS2", "RLHF", "QMDP", "TriMedAI"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True,
                             gridspec_kw={"width_ratios": [4, 3]})
    for ax, tag, pool, title in (
            (axes[0], "in_family", core, "Settings in the population given"),
            (axes[1], "out_of_family", held, "Settings never given")):
        xp = np.arange(len(pool)); w = 0.16
        for k, m in enumerate(meths):
            ax.bar(xp + (k - 2) * w, [nz(tag, m, p) for p in pool], w,
                   color=COL[m], label=LAB[m] if tag == "in_family" else None)
        ax.set_xticks(xp)
        ax.set_xticklabels(["%s\n(frailty %.2f)"
                            % (p.replace(" (held out)", "").replace(" impaired", ""),
                               E["frailty"][p]) for p in pool], fontsize=6.4)
        ax.set_title(title, fontsize=8.4)
        ax.set_ylim(0, 112)
        ax.axhline(100, color="black", ls="--", lw=0.9)
    axes[0].set_ylabel("Normalised score against that setting's own reference")
    axes[0].legend(fontsize=6.3, loc="lower left", ncol=3)
    fig.savefig(os.path.join(FIGURES, "r07_dispersion.png"))
    plt.close(fig)
    print("wrote r07_dispersion.png")


# -------------------------- Figure 8, a person outside the hypothesis set ---
def robustness():
    def nz(tag, m):
        fl = E[tag]["Random"]["all"]["reward"]
        oc = E[tag]["Oracle"]["all"]["reward"]
        return 100.0 * (E[tag][m]["all"]["reward"] - fl) / (oc - fl)

    meths = ["CNN", "NEWS2", "RLHF", "QMDP", "TriMedAI"]
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    xp = np.arange(len(meths))
    b1.bar(xp - 0.2, [nz("in_family", m) for m in meths], 0.4, color=NAVY,
           label="population given")
    b1.bar(xp + 0.2, [nz("out_of_family", m) for m in meths], 0.4, color="#8ea9c1",
           label="population never given")
    b1.set_xticks(xp)
    b1.set_xticklabels([LAB[m].replace(" + ", "\n+ ").replace(" (", "\n(")
                        for m in meths], fontsize=6.4)
    b1.set_ylabel("Normalised score")
    b1.set_ylim(0, 112)
    b1.legend(fontsize=6.6, loc="lower right")

    rin = [E["in_family"][m]["all"]["regret_per_100"] for m in ("QMDP", "TriMedAI")]
    rout = [E["out_of_family"][m]["all"]["regret_per_100"]
            for m in ("QMDP", "TriMedAI")]
    xp2 = np.arange(2)
    b2.bar(xp2 - 0.2, rin, 0.4, color=BLUE, label="population given")
    b2.bar(xp2 + 0.2, rout, 0.4, color=RED, label="population never given")
    for i in range(2):
        d = 100 * (rout[i] / rin[i] - 1)
        b2.annotate("%+.0f%%" % d, (i + 0.2, rout[i]), ha="center", va="bottom",
                    fontsize=7)
    b2.set_xticks(xp2)
    b2.set_xticklabels(["QMDP", "TriMedAI"], fontsize=7.5)
    b2.set_ylabel("Regret per 100 interactions")
    b2.legend(fontsize=6.6, loc="upper left")
    b2.set_title("The cost of being wrong about the person", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "r08_robustness.png"))
    plt.close(fig)
    print("wrote r08_robustness.png")


# ------------------------------------- Figure 10, escalation thresholds -----
def thresholds():
    def thr(pi, rq):
        got = [s for s in range(CRISIS) if pi[sid(rq, s)] == ALERT]
        return min(got) if got else CRISIS

    n2 = news2_policy()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    w = 0.8 / (len(CORE_NAMES) + 2)
    xp = np.arange(len(REQUESTS))
    for k, p in enumerate(CORE_NAMES):
        pi = solve_average(p)[0]
        ax.bar(xp + (k - 2) * w, [thr(pi, r) for r in range(len(REQUESTS))], w,
               label="%s (frailty %.2f)" % (p.replace(" impaired", ""),
                                            PROFILES[p]["frailty"]))
    ax.bar(xp + 2 * w, [thr(n2, r) for r in range(len(REQUESTS))], w,
           color=PURPLE, label="fixed-threshold rule")
    ax.bar(xp + 3 * w, [thr(myopic_policy(CORE_NAMES[3]), r)
                        for r in range(len(REQUESTS))], w,
           color=RED, label="immediate-value rule")
    ax.set_xticks(xp + 0.5 * w)
    ax.set_xticklabels([r.replace(" ", "\n") for r in REQUESTS], fontsize=7)
    ax.set_ylabel("Severity at which a caregiver is summoned")
    ax.set_yticks(range(N_SEV))
    ax.set_yticklabels(["0", "1", "2", "3", "never"])
    ax.legend(fontsize=6.2, ncol=2, loc="upper center")
    fig.savefig(os.path.join(FIGURES, "r10_thresholds.png"))
    plt.close(fig)
    print("wrote r10_thresholds.png")



# ================================================================= curves ====
def curves():
    C = L("har_curves.json")
    x = np.asarray(C["interaction_index"], float)
    style = {"TriMedAI": dict(color=NAVY, lw=2.2, zorder=6),
             "NEWS2": dict(color=PURPLE, lw=1.4),
             "RLHF": dict(color=RED, lw=1.3),
             "MCDM": dict(color="#bf9000", lw=1.2),
             "CNN": dict(color=GREEN, lw=1.2),
             "LSTM": dict(color="#78a353", lw=1.0),
             "RNN": dict(color="#9dc37e", lw=1.0)}
    lab = {"TriMedAI": "TriMedAI (proposed)",
           "NEWS2": "Fixed-threshold escalation rule",
           "RLHF": "Pooled-feedback policy (RLHF)",
           "MCDM": "Multi-criteria rule",
           "CNN": "CNN + population rule",
           "LSTM": "LSTM + population rule",
           "RNN": "RNN + population rule"}
    shown = [m for m in C["methods"] if m in style]

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
    panels = [("agreement", "Agreement with the reference policy (%)", "curves"),
              ("normalised", "Normalised score (floor 0, reference 100)", "norm"),
              ("crisis", "Deteriorations per 100 interactions", "curves")]
    for ax, (key, ylab, where) in zip(axes, panels):
        for m in shown:
            if where == "norm":
                y = np.asarray(C["normalised"][m], float)
                sd = np.asarray(C["normalised_sd"][m], float)
            else:
                y = np.asarray(C["curves"][m][key], float)
                sd = np.asarray(C["curves_sd"][m][key], float)
            ax.plot(x, y, label=lab[m], **style[m])
            ax.fill_between(x, y - sd, y + sd, color=style[m]["color"], alpha=0.13,
                            lw=0)
        ax.set_xlabel("Interaction with the user")
        ax.set_ylabel(ylab, fontsize=7.6)
        ax.set_xlim(0, x[-1])
    axes[1].axhline(100, color="black", ls="--", lw=0.9)
    axes[1].set_ylim(top=112)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.6, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.09), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "r09_curves.png"))
    plt.close(fig)
    print("wrote r09_curves.png")



# =============================================================== ablation ====
def ablation():
    R = L("har_results.json")
    nz = R["normalised"]
    variants = [v for v in R["ablation"]]
    vals = [nz[v]["mean"] for v in variants]
    errs = [nz[v]["sd"] for v in variants]
    cris = [float(np.mean(R["ablation"][v]["crisis_per_100"])) for v in variants]
    order = np.argsort(vals)
    names = [variants[i] for i in order]
    cols = [NAVY if n == "TriMedAI (full)" else "#a6a6a6" for n in names]

    backs = list(R["backups"])
    bvals = [nz[b]["mean"] for b in backs]
    berrs = [nz[b]["sd"] for b in backs]
    badapt = [float(np.mean(R["backups"][b]["steps_to_adapt"])) for b in backs]

    fig = plt.figure(figsize=(9.6, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.0, 1.3, 3.0], wspace=1.15)
    a1, a2, a3 = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                  fig.add_subplot(gs[2]))
    yp = np.arange(len(order))
    a1.barh(yp, [vals[i] for i in order], xerr=[errs[i] for i in order],
            color=cols, height=0.64, error_kw=dict(lw=0.8, capsize=2))
    a1.set_yticks(yp)
    a1.set_yticklabels([n.replace(" ", "\n", 0) for n in names], fontsize=7.0)
    a1.set_xlabel("Normalised score")
    a1.set_xlim(0, 108)
    a1.axvline(100, color="black", ls="--", lw=0.9)
    a1.set_title("Which stage carries the result", fontsize=8.4)
    a2.barh(yp, [cris[i] for i in order], color=cols, height=0.64)
    a2.set_yticks(yp)
    a2.set_yticklabels([])
    a2.set_xlabel("Deteriorations per 100")

    yb = np.arange(len(backs))
    bcols = [NAVY if i == 0 else "#a6a6a6" for i in range(len(backs))]
    a3.barh(yb, bvals, xerr=berrs, color=bcols, height=0.6,
            error_kw=dict(lw=0.8, capsize=2))
    for i, (v, ad) in enumerate(zip(bvals, badapt)):
        a3.annotate("%.0f interactions to settle" % ad, (v, i), va="center",
                    ha="right", fontsize=6.4, color="white",
                    xytext=(-4, 0), textcoords="offset points")
    a3.set_yticks(yb)
    a3.set_yticklabels([b.replace(" (", "\n(") for b in backs], fontsize=7.0)
    a3.set_xlabel("Normalised score")
    a3.set_xlim(0, 108)
    a3.axvline(100, color="black", ls="--", lw=0.9)
    a3.set_title("What one interaction may revise\n"
                 "(belief and initialisation held fixed)", fontsize=8.4)
    fig.savefig(os.path.join(FIGURES, "r05_ablation.png"))
    plt.close(fig)
    print("wrote r05_ablation.png")



# ========================================================== misspecification ==
SHORT = {"product / Boltzmann / constant relief (the agent's model class)":
         "the agent's own\nmodel class",
         "proportional-hazard deterioration": "deterioration not\nof product form",
         "epsilon-greedy stated preference": "preference not\nBoltzmann",
         "graded caregiver relief": "relief after an alert\nnot constant",
         "all three together": "all three\ntogether"}


def misspec():
    M = L("har_misspec.json")
    worlds = list(M["form"])
    meths = ["Fixed threshold", "QMDP", "TriMedAI"]
    cols = {"Fixed threshold": PURPLE, "QMDP": BLUE, "TriMedAI": NAVY}
    lab = {"Fixed threshold": "Fixed-threshold rule", "QMDP": "QMDP",
           "TriMedAI": "TriMedAI (proposed)"}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    xp = np.arange(len(worlds))
    w = 0.26
    for k, m in enumerate(meths):
        a1.bar(xp + (k - 1) * w, [M["form"][d][m]["norm"] for d in worlds], w,
               yerr=[M["form"][d][m]["norm_sd"] for d in worlds],
               color=cols[m], label=lab[m], error_kw=dict(lw=0.7, capsize=2))
        a2.bar(xp + (k - 1) * w, [M["form"][d][m]["regret"] for d in worlds], w,
               color=cols[m], label=lab[m])
    for ax, ylab in ((a1, "Normalised score"), (a2, "Regret per 100 interactions")):
        ax.set_xticks(xp)
        ax.set_xticklabels([SHORT.get(d, d) for d in worlds], fontsize=6.4)
        ax.set_ylabel(ylab, fontsize=8)
    a1.axhline(100, color="black", ls="--", lw=0.9)
    a1.set_ylim(0, 112)
    a1.legend(fontsize=6.6, loc="lower left")
    a1.set_title("The true environment leaves the agent's model class",
                 fontsize=8.4)
    a2.set_title("What being wrong about the form costs", fontsize=8.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "r11_misspec.png"))
    plt.close(fig)
    print("wrote r11_misspec.png")


def noise():
    M = L("har_misspec.json")
    keys = list(M["noise"])

    def pick(prefix):
        got = [(k, float(k.split("=")[-1].rstrip(")").split(",")[0]))
               for k in keys if k.startswith(prefix)]
        return sorted(got, key=lambda kv: kv[1])

    groups = [("missed deterioration reports (fnr)", "missed reports (false negatives)"),
              ("spurious deterioration reports (fpr)", "spurious reports (false positives)"),
              ("both, symmetric", "both, symmetric"),
              ("error in the assumed urgency u(x)", "error in the assumed u(x)")]
    base = M["form"]["product / Boltzmann / constant relief (the agent's model class)"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.3))
    mk = ["o", "s", "^", "d"]
    for i, (pref, label) in enumerate(groups):
        got = pick(pref)
        xs = [0.0] + [v for _, v in got]
        ys = [base["TriMedAI"]["norm"]] + [M["noise"][k]["TriMedAI"]["norm"] for k, _ in got]
        fe = [base["TriMedAI"]["frailty_error"]] + \
             [M["noise"][k]["TriMedAI"]["frailty_error"] for k, _ in got]
        a1.plot(xs, ys, marker=mk[i], ms=3.4, lw=1.3, label=label)
        a2.plot(xs, fe, marker=mk[i], ms=3.4, lw=1.3, label=label)
    a1.set_xlabel("Error rate on the deterioration signal, or spread of the u(x) error")
    a1.set_ylabel("Normalised score", fontsize=8)
    a1.axhline(100, color="black", ls="--", lw=0.9)
    a1.legend(fontsize=6.4, loc="lower left")
    a2.set_xlabel("Error rate on the deterioration signal, or spread of the u(x) error")
    a2.set_ylabel("Error in the recovered frailty", fontsize=8)
    a2.legend(fontsize=6.4, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "r12_noise.png"))
    plt.close(fig)
    print("wrote r12_noise.png")



ORDER = ["architecture", "timing", "frontier", "value_burden", "ablation",
         "regret_onboarding", "dispersion", "robustness", "curves",
         "thresholds", "misspec", "noise"]


if __name__ == "__main__":
    for name in (sys.argv[1:] or ORDER):
        globals()[name]()
