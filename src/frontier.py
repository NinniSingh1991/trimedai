"""Where the framework sits on the harm-against-burden curve, and what moves it.

With the reference policy computed for the criterion actually reported -- the
undiscounted mean reward per interaction -- the reference no longer allows the
same rate of deterioration as the framework. It allows less, and it buys that by
summoning a caregiver more often. The two are at different points on the same
curve, and the thing that decides which point is the planning discount: a policy
that discounts the future at 0.90 weights a deterioration several interactions
away less heavily than one that barely discounts at all.

This script traces that curve. The framework is run at a range of discounts with
everything else held fixed, and each run is recorded by the caregiver call-outs
it raises and the deteriorations it allows. The fixed-threshold rule and the
reference are placed on the same axes. The result is an operating characteristic
rather than a single number: how much caregiver time a service is prepared to
spend decides where on the curve it should sit, and that is a service decision
rather than a technical one.

A second run isolates the cause. A variant that is handed the person's true
latent and plans at the same discount is included at every point, so the gap
between the framework and the reference can be attributed to the discount rather
than to the inference.

    python frontier.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA, RESULTS

import triage_env as TE
from triage_env import (REQUESTS, ACTIONS, CORE_NAMES, N_CORE, N_SEV, N_STATES,
                        ALERT, ASK, COMFORT, PROFILE_NAMES, sid, step,
                        solve_average, latent_of, model_for, plan, news2_policy)
from triage_agents import ParticleBelief, PlanningPolicy, meta_initialisation
import perception as PC

N_ACT, N_REQ = len(ACTIONS), len(REQUESTS)
SEEDS = list(range(int(os.environ.get("N_SEEDS", "15"))))
N_USERS = int(os.environ.get("N_USERS", "60"))
T_STEPS, EVAL_FROM = 200, 100
GAMMAS = [0.70, 0.80, 0.90, 0.95, 0.98, 0.995]


def run_gamma(gamma, pred, yte, oracle_pi, oracle_Q, meta_by_seed, fixed_pi):
    by_class = {c: np.where(yte == c)[0] for c in range(N_REQ)}
    out = {m: {"reward": [], "crisis": [], "alerts": [], "regret": [], "sens": []}
           for m in ("TriMedAI", "LatentGiven", "Fixed threshold", "Oracle",
                     "Random")}
    for seed in SEEDS:
        rng = np.random.default_rng(9000 + seed)
        users = [CORE_NAMES[int(rng.integers(N_CORE))] for _ in range(N_USERS)]
        streams = []
        for _ in users:
            cls = rng.integers(0, N_REQ, size=T_STEPS)
            idx = np.array([rng.choice(by_class[int(c)]) for c in cls])
            streams.append((cls, idx))
        meta_Q = meta_by_seed[seed]
        acc = {m: dict(rew=[], reg=[], cri=0, al=0, n=0, tp=0, fn=0)
               for m in out}
        for m in out:
            for u, (cls, idx) in zip(users, streams):
                ui = PROFILE_NAMES.index(u)
                opi, oq = oracle_pi[u], oracle_Q[u]
                belief = (ParticleBelief(rng=np.random.default_rng(seed * 977 + 21))
                          if m == "TriMedAI" else None)
                pol = None
                if m == "TriMedAI":
                    pol = PlanningPolicy(init=meta_Q.copy())
                elif m == "LatentGiven":
                    fr, pref = latent_of(u)
                    R, T = model_for(fr, pref)
                    pol = PlanningPolicy(init=plan(R, T, gamma=gamma, sweeps=900))
                sev = 0
                for t in range(T_STEPS):
                    rq = int(cls[t])
                    rp = rq if m in ("Oracle",) else int(pred[idx[t]])
                    s_p, s_t = sid(rp, sev), sid(rq, sev)
                    if m == "Oracle":
                        a = int(opi[s_t])
                    elif m == "Random":
                        a = int(rng.integers(N_ACT))
                    elif m == "Fixed threshold":
                        a = int(fixed_pi[s_p])
                    else:
                        a = pol.act(s_p)
                    r, nsev, crisis, worsened = step(u, rq, sev, a, rng)
                    if belief is not None:
                        v = 4.0 * COMFORT[ui, rq, :].copy()
                        v[ASK] = -1e9
                        v = np.exp(v - v.max())
                        v /= v.sum()
                        belief.observe_preference(rp, int(rng.choice(N_ACT, p=v)))
                        belief.observe_transition(rp, worsened)
                        pol.observe(belief)
                    if t >= EVAL_FROM:
                        g = acc[m]
                        g["rew"].append(r)
                        g["reg"].append(float(np.max(oq[s_t]) - oq[s_t, a]))
                        g["cri"] += int(crisis)
                        g["al"] += int(a == ALERT)
                        g["n"] += 1
                        if int(opi[s_t]) == ALERT:
                            g["tp" if a == ALERT else "fn"] += 1
                    sev = nsev
        for m in out:
            g = acc[m]
            n = max(g["n"], 1)
            out[m]["reward"].append(float(np.mean(g["rew"])))
            out[m]["crisis"].append(100.0 * g["cri"] / n)
            out[m]["alerts"].append(100.0 * g["al"] / n)
            out[m]["regret"].append(100.0 * float(np.mean(g["reg"])))
            out[m]["sens"].append(100.0 * g["tp"] / max(g["tp"] + g["fn"], 1))
    o = np.asarray(out["Oracle"]["reward"])
    f = np.asarray(out["Random"]["reward"])
    res = {}
    for m in out:
        x = np.asarray(out[m]["reward"])
        res[m] = {"reward": float(np.mean(x)),
                  "norm": float(100.0 * np.mean((x - f) / (o - f))),
                  "norm_sd": float(100.0 * np.std((x - f) / (o - f), ddof=1)),
                  "crisis": float(np.mean(out[m]["crisis"])),
                  "crisis_sd": float(np.std(out[m]["crisis"], ddof=1)),
                  "alerts": float(np.mean(out[m]["alerts"])),
                  "alerts_sd": float(np.std(out[m]["alerts"], ddof=1)),
                  "regret": float(np.mean(out[m]["regret"])),
                  "escalation_recall": float(np.mean(out[m]["sens"]))}
    return res


def main():
    Xtr, ytr, Xte, yte = PC.load(DATA)
    cnn = PC.train("CNN", Xtr, ytr, Xte, yte, seed=0)
    print("context recognition %.2f%%" % (100 * cnn["accuracy"]), flush=True)
    oracle_pi, oracle_Q = {}, {}
    for p in PROFILE_NAMES:
        pi, Q, _ = solve_average(p)
        oracle_pi[p], oracle_Q[p] = pi, Q
    fixed_pi = news2_policy()

    blob = {"seeds": SEEDS, "n_users": N_USERS, "steps_per_user": T_STEPS,
            "evaluation_from_step": EVAL_FROM, "gammas": GAMMAS, "points": {}}
    t0 = time.perf_counter()
    print("\n%-8s %-16s %8s %9s %9s %9s"
          % ("gamma", "method", "norm", "call-outs", "crises", "recall"))
    for g in GAMMAS:
        TE.reset_config()
        TE.reconfigure(GAMMA=g)
        meta_by_seed = {s: meta_initialisation(seed=s) for s in SEEDS}
        res = run_gamma(g, cnn["pred"], yte, oracle_pi, oracle_Q, meta_by_seed,
                        fixed_pi)
        blob["points"][str(g)] = res
        for m in ("TriMedAI", "LatentGiven"):
            print("%-8s %-16s %8.1f %9.2f %9.2f %9.1f"
                  % (g if m == "TriMedAI" else "", m, res[m]["norm"],
                     res[m]["alerts"], res[m]["crisis"],
                     res[m]["escalation_recall"]), flush=True)
    TE.reset_config()
    ref = blob["points"][str(GAMMAS[0])]
    for m in ("Fixed threshold", "Oracle", "Random"):
        blob[m] = ref[m]
        print("%-8s %-16s %8.1f %9.2f %9.2f %9.1f"
              % ("-", m, ref[m]["norm"], ref[m]["alerts"], ref[m]["crisis"],
                 ref[m]["escalation_recall"]))
    json.dump(blob, open(os.path.join(RESULTS, "har_frontier.json"), "w"), indent=1)
    print("\nwritten to results/har_frontier.json, %.0f s"
          % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
