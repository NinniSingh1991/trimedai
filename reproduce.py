"""Regenerate every number and figure reported in the article.

    python reproduce.py            # everything at the published budget (30 seeds); about 20 hours on a 14-thread laptop CPU
    python reproduce.py --quick    # 3 seeds, 20 users, 3 recogniser seeds: a smoke test of the whole pipeline

The UCI HAR corpus and the three external corpora (HAR70+, HARTH, KU-HAR) are downloaded on first run.
Results are written to results/, figures to figures/, and RESULTS_SUMMARY.md collects every number.
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
QUICK = "--quick" in sys.argv
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
if QUICK:
    ENV.update(N_SEEDS="3", N_USERS="20", TRAIN_SEEDS="0-2")

DECISION = [("uci9_main", "uniform"), ("uci9_acc", "uniform"), ("pooled110", "uniform"),
            ("perfect", "uniform"), ("perfect", "observed"),
            ("har70_uci", "uniform"), ("har70_uci", "observed"), ("har70_harth", "uniform"), ("har70_harth", "observed"),
            ("har70_older", "uniform"), ("har70_older", "observed"), ("har70_pooled", "uniform"), ("har70_pooled", "observed")]

STEPS = [("download the UCI HAR corpus", ["get_har.py"]),
         ("download HAR70+, HARTH and KU-HAR", ["get_external.py"]),
         ("main comparison, ablation, backup rule and safety backstop (Tables 4, 5)", ["run_main.py"]),
         ("held-out latent settings and per-setting results (Table 6, Table S5)", ["run_extended.py"]),
         ("meta-initialisation by onboarding window (Table 5, panel C)", ["onboarding_check.py"]),
         ("learning curves (Figure 3)", ["triage_curves.py"]),
         ("operating curve over the planning discount (Figure 5)", ["frontier.py"]),
         ("fixed-threshold deteriorations (Figure 2)", ["thresholds30.py"])] + \
        [("sensitivity sweep, constant %d of 6 (Table S6)" % (k + 1), ["run_sweep30.py", str(k)]) for k in range(6)] + \
        [("misspecification and signal noise (Table 6, Table S8)", ["misspec.py"]),
         ("continuous population (Table 6)", ["continuous_population.py"]),
         ("recognition on older adults, HAR70+ (Table 7)", ["e1_older_adults.py"]),
         ("age-versus-device control, HARTH (Table 7)", ["e1h_harth_control.py"]),
         ("recognition on 110 people, UCI HAR + KU-HAR (Table 7)", ["e4_more_subjects.py"])] + \
        [("decision layer: %s, %s context mix (Table 7, Table S7)" % c, ["e1d_decision_layer.py", c[0], c[1]]) for c in DECISION] + \
        [("decision latency under multi-user load (Table 8)", ["e3_latency_load.py"]),
         ("summary of every number (RESULTS_SUMMARY.md)", ["compile_results.py"]),
         ("every figure", ["make_figures.py"])]

t0 = time.perf_counter()
for label, cmd in STEPS:
    print("\n=== %s (%s)" % (label, " ".join(cmd)), flush=True)
    r = subprocess.run([sys.executable, "-u", os.path.join(SRC, cmd[0])] + cmd[1:], cwd=SRC, env=ENV)
    if r.returncode != 0:
        sys.exit("failed at %s" % " ".join(cmd))
print("\ncomplete in %.0f s" % (time.perf_counter() - t0))
