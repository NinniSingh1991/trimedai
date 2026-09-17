"""Does the meta-learned initialisation earn its place during onboarding?

The ablation in the main experiment measures interactions 100 to 200, which is
settled behaviour. The stage was never claimed to help there; it was claimed to
help a new user in their first interactions. This measures exactly that window,
and reports the crisis rate within it, because a component that protects people
while the belief is still forming would be worth keeping even if it changes
nothing later.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA as SCRATCH, RESULTS, FIGURES

from triage_env import (solve_average, REQUESTS, ACTIONS, PROFILE_NAMES, CORE_NAMES, ALERT, ASK,
                        N_STATES, sid, step, solve, COMFORT)
from triage_agents import ParticleBelief, PlanningPolicy, meta_initialisation
import perception as PC

SEEDS = list(range(int(os.environ.get("N_SEEDS", "30"))))
N_USERS, T_STEPS = int(os.environ.get("N_USERS", "120")), 200
WINDOWS = [(0, 20), (0, 50), (50, 100), (100, 200)]
N_ACT, N_REQ = len(ACTIONS), len(REQUESTS)


def main():
    Xtr, ytr, Xte, yte = PC.load(SCRATCH)
    cnn = PC.train("CNN", Xtr, ytr, Xte, yte, seed=0)
    oracle_pi = {p: solve_average(p)[0] for p in CORE_NAMES}
    by_class = {c: np.where(yte == c)[0] for c in range(N_REQ)}

    got = {k: {w: {"reward": [], "crisis": [], "n": 0}
               for w in WINDOWS} for k in ("with meta", "without meta")}
    t0 = time.perf_counter()
    for seed in SEEDS:
        rng = np.random.default_rng(5000 + seed)
        meta_Q = meta_initialisation(seed=seed)
        users = [CORE_NAMES[int(rng.integers(len(CORE_NAMES)))]
                 for _ in range(N_USERS)]
        streams = []
        for _ in users:
            cls = rng.integers(0, N_REQ, size=T_STEPS)
            idx = np.array([rng.choice(by_class[int(c)]) for c in cls])
            streams.append((cls, idx))

        for key, init in (("with meta", meta_Q),
                          ("without meta", np.zeros((N_STATES, N_ACT)))):
            acc = {w: {"r": [], "c": 0, "n": 0} for w in WINDOWS}
            for u, (cls, idx) in zip(users, streams):
                ui = PROFILE_NAMES.index(u)
                belief = ParticleBelief(rng=np.random.default_rng(seed * 977 + 11))
                pol = PlanningPolicy(init=init.copy())
                sev = 0
                for t in range(T_STEPS):
                    rq = int(cls[t])
                    rp = int(cnn["pred"][idx[t]])
                    a = pol.act(sid(rp, sev))
                    r, nsev, crisis, worsened = step(u, rq, sev, a, rng)
                    v = 4.0 * COMFORT[ui, rq, :].copy()
                    v[ASK] = -1e9
                    v = np.exp(v - v.max()); v /= v.sum()
                    belief.observe_preference(rp, int(rng.choice(N_ACT, p=v)))
                    belief.observe_transition(rp, worsened)
                    pol.observe(belief)
                    for w in WINDOWS:
                        if w[0] <= t < w[1]:
                            acc[w]["r"].append(r)
                            acc[w]["c"] += int(crisis)
                            acc[w]["n"] += 1
                    sev = nsev
            for w in WINDOWS:
                got[key][w]["reward"].append(float(np.mean(acc[w]["r"])))
                got[key][w]["crisis"].append(100.0 * acc[w]["c"] / max(acc[w]["n"], 1))
        print("  seed %d done (%.0fs)" % (seed, time.perf_counter() - t0), flush=True)

    import json
    from scipy import stats
    out = {"seeds": SEEDS, "n_users": N_USERS, "steps_per_user": T_STEPS,
           "windows": ["%d-%d" % w for w in WINDOWS], "rows": []}
    print("\n%-14s %18s %18s %14s"
          % ("window", "reward with meta", "reward without", "p (Wilcoxon)"))
    for w in WINDOWS:
        a = np.array(got["with meta"][w]["reward"])
        b = np.array(got["without meta"][w]["reward"])
        _, p = stats.wilcoxon(a, b, alternative="two-sided")
        ca = np.array(got["with meta"][w]["crisis"])
        cb = np.array(got["without meta"][w]["crisis"])
        try:
            _, pc = stats.wilcoxon(ca, cb, alternative="two-sided")
        except ValueError:
            pc = float("nan")
        out["rows"].append({"window": "%d-%d" % w,
                            "reward_with": float(a.mean()),
                            "reward_without": float(b.mean()),
                            "reward_sd_with": float(a.std()),
                            "reward_sd_without": float(b.std()),
                            "p_reward": float(p),
                            "crisis_with": float(ca.mean()),
                            "crisis_without": float(cb.mean()),
                            "p_crisis": float(pc)})
        print("%-14s %18.4f %18.4f %14.4f" % ("%d-%d" % w, a.mean(), b.mean(), p))
    print("\n%-14s %18s %18s %14s"
          % ("window", "crises with meta", "crises without", "p (Wilcoxon)"))
    for r in out["rows"]:
        print("%-14s %18.3f %18.3f %14.4f"
              % (r["window"], r["crisis_with"], r["crisis_without"], r["p_crisis"]))
    json.dump(out, open(os.path.join(RESULTS, "har_onboarding.json"), "w"), indent=1)
    print("\nsaved har_onboarding.json")


if __name__ == "__main__":
    main()
