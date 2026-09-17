"""Sensitivity of the conclusions to the constants of the care process.

The wearable corpus is real, but the care process above it is simulated and its
constants were chosen by the authors, so a reader is entitled to ask whether they
were chosen until the proposed method won. Each constant is swept either side of
the value used in the main experiment and the comparison is re-run at every
setting. What matters is not whether the numbers move, which they will, but
whether the ordering of the methods survives.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA as SCRATCH, RESULTS, FIGURES

import triage_env as ENV
from triage_env import (solve_average, REQUESTS, ACTIONS, PROFILE_NAMES, CORE_NAMES, ALERT,
                        ASK, sid, step, solve, reconfigure, reset_config,
                        news2_policy)
from triage_agents import (ParticleBelief, PlanningPolicy, meta_initialisation,
                           global_policy, population_reward_table)
import perception as PC

SEEDS = list(range(int(os.environ.get("N_SEEDS", "5"))))
N_USERS = int(os.environ.get("N_USERS", "50"))
T_STEPS, EVAL_FROM = 160, 80
N_ACT, N_REQ = len(ACTIONS), len(REQUESTS)
METHODS = ["CNN", "NEWS2", "RLHF", "QMDP", "TriMedAI", "Random"]
FLOOR = "Random"
SHOWN = ["CNN", "NEWS2", "RLHF", "QMDP", "TriMedAI"]   # the floor is not ranked

GRID = [
    ("crisis penalty", "CRISIS_PENALTY", [1.0, 1.75, 2.5, 3.5, 5.0]),
    ("caregiver call-out cost", "ESCALATION_COST", [0.30, 0.45, 0.60, 0.85, 1.20]),
    ("relief from a supported context", "RECOVER", [0.02, 0.05, 0.10, 0.20, 0.35]),
    ("urgency scale", "urgency_scale", [0.5, 0.75, 1.0, 1.3, 1.6]),
    ("frailty spread", "frailty_spread", [0.0, 0.5, 1.0, 1.5, 2.0]),
    ("discount factor", "GAMMA", [0.70, 0.80, 0.90, 0.95, 0.98]),
]


def _proj(belief):
    from triage_env import latent_of
    w = belief.weights
    fr = w @ belief.frailty
    pref = w @ belief.pref
    d = np.array([np.hypot(fr - latent_of(p)[0],
                           np.mean(np.abs(pref - latent_of(p)[1])))
                  for p in CORE_NAMES])
    s = np.exp(-8.0 * d)
    return s / s.sum()


def run_setting(backbones, yte):
    oracle_pi, oracle_Q = {}, {}
    for p in CORE_NAMES:
        pi, Q, _ = solve_average(p)
        oracle_pi[p], oracle_Q[p] = pi, Q
    QSTAR = np.stack([oracle_Q[p] for p in CORE_NAMES])
    pop_R = population_reward_table()
    news2_pi = news2_policy()
    by_class = {c: np.where(yte == c)[0] for c in range(N_REQ)}
    got = {m: {"reward": [], "crisis": []} for m in METHODS + ["Oracle"]}

    for seed in SEEDS:
        rng = np.random.default_rng(2000 + seed)
        meta_Q = meta_initialisation(seed=seed, n_draws=16)
        glob_Q = global_policy(steps=40000, seed=seed)
        users = [CORE_NAMES[int(rng.integers(len(CORE_NAMES)))]
                 for _ in range(N_USERS)]
        streams = []
        for _ in users:
            cls = rng.integers(0, N_REQ, size=T_STEPS)
            idx = np.array([rng.choice(by_class[int(c)]) for c in cls])
            streams.append((cls, idx))

        for m in METHODS + ["Oracle"]:
            rew, cri, n = [], 0, 0
            pred = backbones["MLP" if m == "RLHF" else "CNN"]["pred"]
            for u, (cls, idx) in zip(users, streams):
                ui = PROFILE_NAMES.index(u)
                opi = oracle_pi[u]
                belief = ParticleBelief(rng=np.random.default_rng(seed * 977 + 7)) \
                    if m in ("QMDP", "TriMedAI") else None
                pol = PlanningPolicy(init=meta_Q.copy()) if m == "TriMedAI" else None
                sev = 0
                for t in range(T_STEPS):
                    rq = int(cls[t])
                    rp = rq if m == "Oracle" else int(pred[idx[t]])
                    s_p = sid(rp, sev)
                    if m == "Oracle":
                        a = int(opi[sid(rq, sev)])
                    elif m == "CNN":
                        a = int(np.argmax(pop_R[s_p]))
                    elif m == "NEWS2":
                        a = int(news2_pi[s_p])
                    elif m == "RLHF":
                        a = int(np.argmax(glob_Q[s_p]))
                    elif m == "Random":
                        a = int(rng.integers(N_ACT))
                    elif m == "QMDP":
                        a = int(np.argmax(_proj(belief) @ QSTAR[:, s_p, :]))
                    else:
                        a = pol.act(s_p)
                    r, nsev, crisis, worsened = step(u, rq, sev, a, rng)
                    if belief is not None:
                        v = 4.0 * ENV.COMFORT[ui, rq, :].copy()
                        v[ASK] = -1e9
                        v = np.exp(v - v.max()); v /= v.sum()
                        belief.observe_preference(rp, int(rng.choice(N_ACT, p=v)))
                        belief.observe_transition(rp, worsened)
                        if pol is not None:
                            pol.observe(belief)
                    if t >= EVAL_FROM:
                        rew.append(r); cri += int(crisis); n += 1
                    sev = nsev
            got[m]["reward"].append(float(np.mean(rew)))
            got[m]["crisis"].append(100.0 * cri / max(n, 1))

    import numpy as _np
    o_seed = _np.asarray(got["Oracle"]["reward"], float)
    f_seed = _np.asarray(got[FLOOR]["reward"], float)
    span = o_seed - f_seed
    orc = float(_np.mean(o_seed))
    out = {}
    for m in got:
        x = _np.asarray(got[m]["reward"], float)
        z = 100.0 * (x - f_seed) / span
        out[m] = {"reward": float(_np.mean(x)),
                  "pct_oracle": 100.0 * float(_np.mean(x)) / orc,
                  "normalised": float(_np.mean(z)),
                  "normalised_sd": float(_np.std(z, ddof=1)) if len(z) > 1 else 0.0,
                  "crisis": float(_np.mean(got[m]["crisis"]))}
    out["_floor_reward"] = float(_np.mean(f_seed))
    return out


def main():
    Xtr, ytr, Xte, yte = PC.load(SCRATCH)
    backbones = {n: PC.train(n, Xtr, ytr, Xte, yte, seed=0) for n in ("CNN", "MLP")}
    print("perception ready\n", flush=True)

    out, t0 = {}, time.perf_counter()
    for label, key, values in GRID:
        out[label] = {"parameter": key, "values": values, "results": []}
        print("=== %s (%s)" % (label, key), flush=True)
        print("   %-8s %s" % ("value", "  ".join("%-16s" % m for m in SHOWN)),
              flush=True)
        for v in values:
            reset_config()
            reconfigure(**{key: v})
            res = run_setting(backbones, yte)
            out[label]["results"].append({"value": v, **res})
            cells = ["%5.1f%% / %4.2f" % (res[m]["normalised"], res[m]["crisis"])
                     for m in SHOWN]
            best = max(SHOWN, key=lambda m: res[m]["normalised"])
            print("   %-8s %s   best: %s"
                  % (v, "  ".join("%-16s" % c for c in cells), best), flush=True)
        print(flush=True)
    reset_config()

    blob = dict(out)
    blob["_budget"] = {"seeds": len(SEEDS), "n_users": N_USERS,
                       "steps_per_user": T_STEPS, "evaluation_from_step": EVAL_FROM,
                       "methods": METHODS}
    json.dump(blob, open(os.path.join(RESULTS, "har_sensitivity.json"), "w"), indent=1)
    n_set = sum(len(v) for _, _, v in GRID)
    wins = sum(1 for lab in out for row in out[lab]["results"]
               if max(SHOWN, key=lambda m: row[m]["normalised"]) == "TriMedAI")
    beats = {m: sum(1 for lab in out for row in out[lab]["results"]
                    if row["TriMedAI"]["normalised"] > row[m]["normalised"])
             for m in SHOWN if m != "TriMedAI"}
    print("TriMedAI best in %d of %d settings" % (wins, n_set))
    for m, c in beats.items():
        print("   ahead of %-8s in %d of %d" % (m, c, n_set))
    print("total %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
