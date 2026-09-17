"""Regenerate every number and figure reported in the article.

    python reproduce.py            # everything, roughly 70 minutes on four cores
    python reproduce.py --quick    # reduced seeds, roughly 6 minutes, for a smoke test

The corpus is downloaded on first run. Results are written to results/ and
figures to figures/.
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
QUICK = "--quick" in sys.argv
ENV = dict(os.environ)
if QUICK:
    ENV["N_SEEDS"], ENV["N_USERS"] = "3", "20"

STEPS = [("download the wearable corpus", "get_har.py"),
         ("main comparison, stage ablation, backup-rule ablation", "run_main.py"),
         ("per-setting performance and out-of-family robustness", "run_extended.py"),
         ("effect of the meta-initialisation by window", "onboarding_check.py"),
         ("learning curves with per-seed dispersion", "triage_curves.py"),
         ("harm against caregiver burden, by planning discount", "frontier.py"),
         ("sensitivity to the care-process constants", "sensitivity.py"),
         ("misspecification of the model class, and observation noise",
          "misspec.py"),
         ("a population drawn from a continuous prior",
          "continuous_population.py"),
         ("every figure in the article", "make_figures.py")]

t0 = time.perf_counter()
for label, script in STEPS:
    print("\n=== %s (%s)" % (label, script), flush=True)
    r = subprocess.run([sys.executable, "-u", os.path.join(SRC, script)],
                       cwd=SRC, env=ENV)
    if r.returncode != 0:
        sys.exit("failed at %s" % script)
print("\ncomplete in %.0f s" % (time.perf_counter() - t0))
