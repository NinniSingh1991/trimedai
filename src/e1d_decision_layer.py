"""What recognition quality does to the decision layer.

The comparison of sensitivity.run_setting is re-run at the main constants, 30 seeds and 50 users per seed, with only
the source of the context stream changed. Every method sees the same windows and the same recogniser output; the
reference policy acts on the true context and the floor acts at random, as in every experiment.

Conditions
  uci9_main       UCI HAR test subjects, the main nine-channel CNN and MLP (as in the main experiments)
  uci9_acc        UCI HAR test subjects, the accelerometer-only model of E1 (trained on 21 UCI subjects)
  har70_uci       HAR70+ older adults, the same accelerometer-only model trained on young adults only
  har70_older     HAR70+ older adults, the accelerometer-only model trained on other older adults (leave-3-out)
  har70_pooled    HAR70+ older adults, trained on UCI HAR plus the other older adults
  har70_harth     HAR70+ older adults, trained on HARTH (younger adults, same sensor and setting)
  pooled110       all 110 people of UCI HAR and KU-HAR, the six-channel model of E4 (subject-grouped 5-fold)
  perfect         every context recognised correctly (an upper reference for recognition)
Simulation seed s uses the recogniser trained with seed s (30 recognisers per condition).
"""
import json, os, sys, time
os.environ.setdefault("N_SEEDS", "30"); os.environ.setdefault("N_USERS", "50")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE); OUT = os.path.join(ROOT, "results")
import sensitivity as S
import perception as PC
from triage_env import reset_config

cond = sys.argv[1]
mix = sys.argv[2] if len(sys.argv) > 2 else "uniform"     # "uniform" or "observed" (context frequencies of the windows)
_e1 = os.path.join(OUT, "e1_predictions.npz"); e1 = np.load(_e1) if os.path.exists(_e1) else None
if cond == "uci9_main":
    Xtr, ytr, Xte, yte = PC.load(S.SCRATCH)
    bb = {n: PC.train(n, Xtr, ytr, Xte, yte, seed=0) for n in ("CNN", "MLP")}
    y = yte; acc = {n: bb[n]["accuracy"] for n in bb}
else:
    if cond == "uci9_acc":
        y, p = e1["yb"], e1["uci_test_A21"]
    elif cond == "har70_uci":
        y, p = e1["yh"], e1["har70_A21"]
    elif cond == "har70_older":
        y, p = e1["yh"], e1["har70_B"]
    elif cond == "har70_pooled":
        y, p = e1["yh"], e1["har70_C"]
    elif cond == "har70_harth":
        eh = np.load(os.path.join(OUT, "e1h_predictions.npz")); y, p = eh["yh"], eh["har70_harth"]
    elif cond == "pooled110":
        e4 = np.load(os.path.join(OUT, "e4_predictions.npz")); y, p = e4["y"], e4["pred"]
    elif cond == "perfect":
        y = e1["yh"]; p = np.tile(y, (len(S.SEEDS), 1))
    else:
        raise SystemExit("unknown condition " + cond)
    p = np.asarray(p)
    assert p.ndim == 2 and p.shape[0] >= len(S.SEEDS), p.shape      # one recogniser per training seed
    S.PRED_BY_SEED = {s: p[s] for s in S.SEEDS}                      # simulation seed s uses recogniser seed s
    bb = {"CNN": {"pred": p[0]}, "MLP": {"pred": p[0]}}
    accs = [(p[s] == np.asarray(y)).mean() for s in S.SEEDS]
    acc = {"recogniser": float(np.mean(accs)), "recogniser_sd": float(np.std(accs, ddof=1))}
if mix == "observed":
    S.CONTEXT_PROBS = np.bincount(np.asarray(y), minlength=6) / len(y)
t0 = time.perf_counter()
reset_config()
r = S.run_setting(bb, np.asarray(y))
json.dump({"condition": cond, "context_mix": mix, "context_probs": (None if S.CONTEXT_PROBS is None else S.CONTEXT_PROBS.tolist()), "recognition_accuracy": acc, "windows": int(len(y)),
           "windows_per_class": np.bincount(np.asarray(y), minlength=6).tolist(), "results": r,
           "_budget": {"seeds": len(S.SEEDS), "n_users": S.N_USERS, "steps_per_user": S.T_STEPS,
                       "evaluation_from_step": S.EVAL_FROM}, "seconds": time.perf_counter() - t0},
          open(os.path.join(OUT, "e1d_%s%s.json" % (cond, "" if mix == "uniform" else "_observed")), "w"), indent=1)
print(cond, mix, acc, {m: round(r[m]["normalised"], 2) for m in S.SHOWN}, "%.0fs" % (time.perf_counter() - t0))
