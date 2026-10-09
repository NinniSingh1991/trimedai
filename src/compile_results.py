"""Compile every result into one Markdown summary (numbers read from the result files only)."""
import json, os
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
OLD = os.path.join(ROOT, "results")
SH = ["CNN", "NEWS2", "RLHF", "QMDP", "TriMedAI"]
NAME = {"CNN": "fixed population rule", "NEWS2": "fixed-threshold rule", "RLHF": "pooled-feedback policy",
        "QMDP": "QMDP (solved per setting)", "TriMedAI": "framework"}
L = []
w = L.append


def pm(d, k, scale=100.0, dp=2):
    return "%.*f ± %.*f" % (dp, scale * d[k]["mean"], dp, scale * d[k]["sd"])


def test(a, b):
    a, b = np.asarray(a), np.asarray(b)
    p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
    return a.mean() - b.mean(), p


# ---------------------------------------------------------------- section
w("# Results summary\n\nAll numbers below are read directly from the result files in `results/`.\n")
w("## Sensitivity sweep at 30 seeds\n")
f_old = os.path.join(OLD, "har_sensitivity.json")
old = json.load(open(f_old)) if os.path.exists(f_old) else None
rows = [json.load(open(os.path.join(R, "sens30_row%d.json" % i))) for i in range(6)]
maxdev, n, wins, over = 0.0, 0, 0, 0
beats = {m: [0, 0, 0, 0] for m in SH if m != "TriMedAI"}      # ahead, sig better, tie, sig worse
cells = []
for r in rows:
    for c, oc in zip(r["results"], old[r["label"]]["results"] if old else [None] * len(r["results"])):
        n += 1
        for m in SH:
            z = np.array(c[m]["normalised_seeds"])
            if oc is not None:
                maxdev = max(maxdev, abs(z[:10].mean() - oc[m]["normalised"]))
            over += c[m]["normalised"] > 100
        wins += max(SH, key=lambda m: c[m]["normalised"]) == "TriMedAI"
        zt = c["TriMedAI"]["normalised_seeds"]
        for m in beats:
            d, p = test(zt, c[m]["normalised_seeds"])
            beats[m][0] += d > 0
            beats[m][1 if (p < 0.05 and d > 0) else (3 if p < 0.05 else 2)] += 1
        dq, pq = test(zt, c["QMDP"]["normalised_seeds"])
        cells.append((r["label"], c["value"], c["TriMedAI"]["normalised"], c["TriMedAI"]["normalised_sd"],
                      c["QMDP"]["normalised"], c["NEWS2"]["normalised"], c["RLHF"]["normalised"],
                      c["CNN"]["normalised"], c["TriMedAI"]["crisis"], dq, pq))
w("- Budget: %s." % rows[0]["_budget"])
if old:
    w("- Reproducibility: the mean of seeds 0-9 reproduces the published 10-seed values; largest difference %.2e." % maxdev)
w("- Framework highest score in %d of %d configurations; cells above 100 percent: %d." % (wins, n, over))
for m, (a, b, t, s) in beats.items():
    w("- vs %s: ahead in %d/%d; paired Wilcoxon (30 seeds, p < 0.05): framework better %d, no significant difference %d, framework worse %d." % (NAME[m], a, n, b, t, s))
w("\n| Constant | Value | Framework (mean ± SD) | QMDP | Threshold rule | Pooled feedback | Population rule | Framework deteriorations /100 | Framework − QMDP | p |\n|---|---|---|---|---|---|---|---|---|---|")
for lab, v, ft, fs, q, nw, rl, cn, cr, dq, pq in cells:
    w("| %s | %s | %.2f ± %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %+.2f | %.4f |" % (lab, v, ft, fs, q, nw, rl, cn, cr, dq, pq))

# ---------------------------------------------------------------- section
e1 = json.load(open(os.path.join(R, "e1_recognition.json")))
w("\n## Recognition on older adults (HAR70+)\n")
w("- Data: %s" % e1["info"])
w("\n| Recogniser (accelerometer only, %d training seeds) | Accuracy (%%) | Balanced accuracy (%%) | Macro-F1 (%%) |\n|---|---|---|---|"
  % len(e1["A_uci21_on_har70"]["per_seed"]))
for k, lab in [("A_uci21_on_uci_test", "UCI HAR 21 young → UCI HAR 9 young"),
               ("A_uci21_on_har70", "UCI HAR 21 young → HAR70+ 18 older"),
               ("A_uci30_on_har70", "UCI HAR 30 young → HAR70+ 18 older"),
               ("B_har70_leave3out", "HAR70+ leave-3-subjects-out (older → older)"),
               ("C_pooled_on_har70", "UCI HAR 30 + other older adults → held-out older")]:
    d = e1[k]
    w("| %s | %s | %s | %s |" % (lab, pm(d, "accuracy"), pm(d, "balanced_accuracy"), pm(d, "macro_f1")))
w("\nPer-class recall, mean over training seeds (%):\n\n| Recogniser | " + " | ".join(
    ["walking", "stairs up", "stairs down", "sitting", "standing", "lying"]) + " |\n|---|---|---|---|---|---|---|")
for k in ["A_uci21_on_har70", "B_har70_leave3out", "C_pooled_on_har70"]:
    rs = e1[k]["per_seed"]
    vals = [np.mean([r["recall"][c] for r in rs]) * 100 for c in ["walking", "stairs up", "stairs down", "sitting", "standing", "lying"]]
    w("| %s | %s |" % (k, " | ".join("%.1f" % v for v in vals)))
four = []
for k in ["A_uci21_on_har70", "B_har70_leave3out", "C_pooled_on_har70"]:
    acc4 = []
    for r in e1[k]["per_seed"]:
        cm = np.array(r["confusion"]); idx = [0, 3, 4, 5]
        acc4.append(np.trace(cm[np.ix_(idx, idx)]) / cm[idx].sum())
    four.append((k, np.mean(acc4) * 100, np.std(acc4, ddof=1) * 100))
w("\nAccuracy over the four well-populated classes only (walking, sitting, standing, lying; windows of other true classes excluded, predictions into any class counted):")
for k, m_, s_ in four:
    w("- %s: %.2f ± %.2f %%" % (k, m_, s_))
if os.path.exists(os.path.join(R, "e1h_harth_control.json")):
    h = json.load(open(os.path.join(R, "e1h_harth_control.json")))
    w("\n### Age versus device and setting (HARTH)\n")
    w("- Data: %s" % h["info"])
    w("\n| Train → test | What changes | Accuracy (%) | Balanced accuracy (%) |\n|---|---|---|---|")
    for k, lab, what in [("young_to_old", "HARTH 22 (25-68 y) → HAR70+ 18 (70-95 y), same sensor", "age only"),
                         ("harth_cv", "HARTH leave-subjects-out", "nothing (reference)"),
                         ("uci_to_harth", "UCI HAR 30 (phone, lab) → HARTH 22 (back sensor, free living)", "device and setting")]:
        w("| %s | %s | %s | %s |" % (lab, what, pm(h[k], "accuracy"), pm(h[k], "balanced_accuracy")))
e4 = json.load(open(os.path.join(R, "e4_more_subjects.json")))
w("\n## Recognition on 110 people (UCI HAR 30 + KU-HAR 80)\n")
w("- Data: %s" % e4["info"])
w("\n| Protocol (6 channels: body acceleration + gyroscope; %d training seeds) | Accuracy (%%) | Balanced accuracy (%%) | Macro-F1 (%%) |\n|---|---|---|---|"
  % len(e4["pooled_cv_120"]["per_seed"]))
for k, lab in [("reference_uci21_on_uci9", "UCI HAR 21 → 9 (30 people)"), ("pooled_cv_120", "5-fold subject-grouped CV, all 110 people"),
               ("pooled_cv_uci_part", "  of which UCI HAR people"), ("pooled_cv_kuhar_part", "  of which KU-HAR people")]:
    d = e4[k]
    w("| %s | %s | %s | %s |" % (lab, pm(d, "accuracy"), pm(d, "balanced_accuracy"), pm(d, "macro_f1")))

# ---------------------------------------------------------------- decision layer
w("\n## Decision layer under each recognition source (30 seeds, 50 users, main constants)\n")
w("| Recognition source | Context mix | Recogniser accuracy (%) | Framework | QMDP | Threshold rule | Pooled feedback | Population rule | Framework deteriorations /100 | Framework − QMDP (p) | Framework − threshold (p) |\n|---|---|---|---|---|---|---|---|---|---|---|")
order = ["perfect", "uci9_main", "uci9_acc", "pooled110", "har70_uci", "har70_harth", "har70_older", "har70_pooled"]
for c in order:
    for mix in ["", "_observed"]:
        f = os.path.join(R, "e1d_%s%s.json" % (c, mix))
        if not os.path.exists(f):
            continue
        d = json.load(open(f)); r = d["results"]
        acc = list(d["recognition_accuracy"].values())[0] * 100
        dq, pq = test(r["TriMedAI"]["normalised_seeds"], r["QMDP"]["normalised_seeds"])
        dn, pn = test(r["TriMedAI"]["normalised_seeds"], r["NEWS2"]["normalised_seeds"])
        w("| %s | %s | %.2f | %.2f ± %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %+.2f (%.4f) | %+.2f (%.2g) |" % (
            c, "observed" if mix else "uniform", acc, r["TriMedAI"]["normalised"], r["TriMedAI"]["normalised_sd"],
            r["QMDP"]["normalised"], r["NEWS2"]["normalised"], r["RLHF"]["normalised"], r["CNN"]["normalised"],
            r["TriMedAI"]["crisis"], dq, pq, dn, pn))
        if mix and "context_probs" in d and d["context_probs"]:
            probs = d["context_probs"]
w("\nObserved context mix (share of HAR70+ windows per class: walking, up, down, sitting, standing, lying): %s" % (
    ", ".join("%.4f" % p for p in probs) if "probs" in dir() else "n/a"))

# ---------------------------------------------------------------- section
f = os.path.join(R, "e3_latency_load.json")
if os.path.exists(f):
    e3 = json.load(open(f))
    w("\n## Decision latency under multi-user load\n")
    w("- Machine: %s" % e3["machine"])
    w("- Recognition, main CNN, ms per window by batch size: %s" % {k: round(v, 4) for k, v in e3["recognition_ms_per_window"].items()})
    w("\n| Users held | Decisions timed | Mean (ms) | Median (ms) | 95th pct (ms) | 99th pct (ms) | Max (ms) | Recognition (ms/window) | Round, end to end (s) | Memory per user (KB) | Users served in real time per process |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in e3["single_process"].items():
        w("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.3f | %.4f | %.3f | %.1f | %d |" % (
            k, v["decisions"], v["decision_ms_mean"], v["decision_ms_median"], v["decision_ms_p95"], v["decision_ms_p99"],
            v["decision_ms_max"], v["recognition_ms_per_window"], v["round_seconds_end_to_end_estimate"],
            v["memory_kb_per_user"], v["users_in_real_time_per_process"]))
    w("\n| Worker processes | Users | Decisions | Decisions per second |\n|---|---|---|---|")
    for k, v in e3["multi_process"].items():
        w("| %s | %d | %d | %.0f |" % (k, v["users"], v["decisions"], v["decisions_per_second"]))

# ---------------------------------------------------------------- safety backstop (main comparison budget)
f = os.path.join(R, "har_results.json")
if os.path.exists(f):
    bs = json.load(open(f)); BK = "TriMedAI + safety backstop"
    w("\n## Safety backstop (main comparison budget: %d seeds, %d users, %d interactions; file results/har_results.json)\n"
      % (len(bs["seeds"]) if isinstance(bs["seeds"], list) else bs["seeds"], bs["n_users"], bs["steps_per_user"]))
    w("The framework unchanged, except that a caregiver is always called once severity reaches the band at which the "
      "fixed-threshold rule escalates. Simulated last in every seed, so every other method reproduces exactly.\n")
    w("| Method | Normalised score (95% CI) | Deteriorations /100 | Call-outs /100 | Regret /100 | Users with ≥1 deterioration (%) | Interactions to settle |\n|---|---|---|---|---|---|---|")
    mn = lambda m, k: float(np.mean(bs["main"][m][k]) if m != "oracle" else np.mean(bs["oracle"][k]))
    for m, lab in [("TriMedAI", "Framework"), (BK, "**Framework + safety backstop**"), ("NEWS2", "Fixed-threshold rule"),
                   ("QMDP", "QMDP"), ("oracle", "Reference policy")]:
        sc = "100" if m == "oracle" else "%.2f (%.2f–%.2f)" % (bs["normalised"][m]["mean"], bs["normalised"][m]["lo"], bs["normalised"][m]["hi"])
        w("| %s | %s | %.3f | %.3f | %.3f | %.2f | %.1f |" % (lab, sc, mn(m, "crisis_per_100"), mn(m, "alerts_per_100"),
          0.0 if m == "oracle" else mn(m, "regret_per_100"), mn(m, "users_with_crisis"), mn(m, "steps_to_adapt")))
    nb = bs["normalised"][BK]["per_seed"]
    d1, p1 = test(nb, bs["normalised"]["NEWS2"]["per_seed"])
    d2, p2 = test(bs["main"][BK]["alerts_per_100"], bs["main"]["NEWS2"]["alerts_per_100"])
    d3, p3 = test(nb, bs["normalised"]["TriMedAI"]["per_seed"])
    w("\nPaired Wilcoxon over seeds: backstop vs threshold rule %+.2f points (p = %.3g); call-outs %+.3f per 100 (p = %.3g); "
      "backstop vs framework %+.2f points (p = %.3g)." % (d1, p1, d2, p2, d3, p3))

# ---------------------------------------------------------------- operating curve over the planning discount
f = os.path.join(OLD, "har_frontier.json")
if os.path.exists(f):
    fr = json.load(open(f))
    w("\n## Operating curve over the planning discount (results/har_frontier.json; %d seeds, %d users, %d interactions)\n"
      % (len(fr["seeds"]), fr["n_users"], fr["steps_per_user"]))
    w("| Planning discount | Score | Deteriorations /100 | Call-outs /100 | Regret /100 |\n|---|---|---|---|---|")
    for g in fr["gammas"]:
        t = fr["points"][str(g)]["TriMedAI"]
        w("| %s%s | %.2f | %.2f | %.2f | %.3f |" % (g, " (main setting)" if abs(g - 0.9) < 1e-9 else "", t["norm"], t["crisis"], t["alerts"], t["regret"]))
    for k in ("Fixed threshold", "Oracle"):
        t = fr[k]
        if isinstance(t, dict) and "norm" in t:
            w("| %s (same run) | %.2f | %.2f | %.2f | %s |" % (k, t["norm"], t["crisis"], t["alerts"], "%.3f" % t["regret"] if "regret" in t else "—"))

out = os.path.join(ROOT, "RESULTS_SUMMARY.md")
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written", out, len(L), "lines")
