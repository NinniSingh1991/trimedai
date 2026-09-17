"""Equity across profiles, and behaviour when the person is not a known type.

With the belief carried over a continuous latent, a person who matches none of
the named profiles is no longer outside the agent's hypothesis space; they are
simply another point in it. The comparators that require an enumerable set of
profiles have no such recourse, and this experiment measures what that costs
them. Performance is also broken out per profile, because a system that averages
well while serving one disability group badly is not acceptable in assistive
care and an aggregate figure cannot show the difference.
"""
import json, os, sys, time
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA as SCRATCH, RESULTS, FIGURES

from triage_env import (solve_average, REQUESTS, ACTIONS, PROFILE_NAMES, CORE_NAMES,
                        HELD_OUT_NAMES, N_CORE, N_SEV, N_STATES, ALERT, ASK,
                        CRISIS, PROFILES, sid, step, solve, latent_of,
                        model_for, plan, news2_policy, immediate_reward,
                        population_latent, COMFORT)
from triage_agents import (ParticleBelief, PlanningPolicy, meta_initialisation,
                           global_policy, population_reward_table, mcdm_policy)
import perception as PC

SEEDS = list(range(int(os.environ.get("N_SEEDS", "20"))))
N_USERS, T_STEPS, EVAL_FROM = 60, 200, 100
N_ACT, N_REQ = len(ACTIONS), len(REQUESTS)
METHODS = ["CNN", "NEWS2", "RLHF", "QMDP", "TriMedAI", "Random"]
# Random is the floor policy the normalised scores are anchored on; it is
# placed last so that every other method draws the same random stream it
# drew before this comparator was added.
FLOOR = "Random"


def acc():
    return dict(reward=[], crisis=0, steps=0, agree=[], regret=[], users=0,
                users_hit=0, fr_err=[], pf_err=[], alerts=0,
                tp=0, fn=0, tn=0, fp=0)


def run_phase(pool, backbones, yte, oracle_pi, oracle_Q, pop_R, news2_pi,
              tag):
    QSTAR = np.stack([oracle_Q[p] for p in CORE_NAMES])
    by_class = {c: np.where(yte == c)[0] for c in range(N_REQ)}
    per_seed = []

    for seed in SEEDS:
        rng = np.random.default_rng(3000 + seed)
        meta_Q = meta_initialisation(seed=seed)
        glob_Q = global_policy(seed=seed)
        users = [pool[int(rng.integers(len(pool)))] for _ in range(N_USERS)]
        streams = []
        for _ in users:
            cls = rng.integers(0, N_REQ, size=T_STEPS)
            idx = np.array([rng.choice(by_class[int(c)]) for c in cls])
            streams.append((cls, idx))

        store = {m: {"all": acc()} for m in METHODS + ["Oracle"]}
        for m in store:
            for p in pool:
                store[m][p] = acc()

        for m in METHODS + ["Oracle"]:
            pred = backbones["MLP" if m == "RLHF" else "CNN"]["pred"]
            for u, (cls, idx) in zip(users, streams):
                ui = PROFILE_NAMES.index(u)
                opi, oq = oracle_pi[u], oracle_Q[u]
                belief = (ParticleBelief(rng=np.random.default_rng(seed * 977 + 5))
                          if m in ("QMDP", "TriMedAI") else None)
                pol = (PlanningPolicy(init=meta_Q.copy())
                       if m == "TriMedAI" else None)
                sev, hit = 0, False
                for t in range(T_STEPS):
                    rq = int(cls[t])
                    rp = rq if m == "Oracle" else int(pred[idx[t]])
                    s_p, s_t = sid(rp, sev), sid(rq, sev)
                    if m == "Oracle":
                        a = int(opi[s_t])
                    elif m == "CNN":
                        a = int(np.argmax(pop_R[s_p]))
                    elif m == "NEWS2":
                        a = int(news2_pi[s_p])
                    elif m == "RLHF":
                        a = int(np.argmax(glob_Q[s_p]))
                    elif m == "Random":
                        a = int(rng.integers(N_ACT))
                    elif m == "QMDP":
                        post = _proj(belief)
                        a = int(np.argmax(post @ QSTAR[:, s_p, :]))
                    else:
                        a = pol.act(s_p)
                    r, nsev, crisis, worsened = step(u, rq, sev, a, rng)
                    if belief is not None:
                        v = 4.0 * COMFORT[ui, rq, :].copy()
                        v[ASK] = -1e9
                        v = np.exp(v - v.max()); v /= v.sum()
                        belief.observe_preference(rp, int(rng.choice(N_ACT, p=v)))
                        belief.observe_transition(rp, worsened)
                        if pol is not None:
                            pol.observe(belief)
                    want = int(opi[s_t])
                    if t >= EVAL_FROM:
                        for key in ("all", u):
                            g = store[m][key]
                            g["reward"].append(r)
                            g["crisis"] += int(crisis)
                            g["steps"] += 1
                            g["alerts"] += int(a == ALERT)
                            g["agree"].append(a == want)
                            g["regret"].append(float(np.max(oq[s_t]) - oq[s_t, a]))
                            if want == ALERT:
                                g["tp" if a == ALERT else "fn"] += 1
                            else:
                                g["fp" if a == ALERT else "tn"] += 1
                        if crisis:
                            hit = True
                    sev = nsev
                if belief is not None:
                    fe, pe = belief.latent_error(u)
                    for key in ("all", u):
                        store[m][key]["fr_err"].append(fe)
                        store[m][key]["pf_err"].append(pe)
                for key in ("all", u):
                    store[m][key]["users"] += 1
                    store[m][key]["users_hit"] += int(hit)

        out = {}
        for m, byk in store.items():
            out[m] = {}
            for key, g in byk.items():
                n = max(g["steps"], 1)
                out[m][key] = dict(
                    reward=float(np.mean(g["reward"])) if g["reward"] else 0.0,
                    crisis_per_100=100.0 * g["crisis"] / n,
                    alerts_per_100=100.0 * g["alerts"] / n,
                    users_with_crisis=100.0 * g["users_hit"] / max(g["users"], 1),
                    agreement=100.0 * float(np.mean(g["agree"])) if g["agree"] else 0.0,
                    regret_per_100=100.0 * float(np.mean(g["regret"])) if g["regret"] else 0.0,
                    sensitivity=100.0 * g["tp"] / max(g["tp"] + g["fn"], 1),
                    specificity=100.0 * g["tn"] / max(g["tn"] + g["fp"], 1),
                    frailty_error=float(np.mean(g["fr_err"])) if g["fr_err"] else None,
                    preference_error=float(np.mean(g["pf_err"])) if g["pf_err"] else None)
        per_seed.append(out)
        if (seed + 1) % 5 == 0:
            print("   %s %d/%d" % (tag, seed + 1, len(SEEDS)), flush=True)
    return per_seed


def _proj(belief):
    """QMDP can only represent a distribution over the named profiles, so the
    continuous posterior is projected onto them"""
    w = belief.weights
    fr = w @ belief.frailty
    pref = w @ belief.pref
    d = np.array([np.hypot(fr - latent_of(p)[0],
                           np.mean(np.abs(pref - latent_of(p)[1])))
                  for p in CORE_NAMES])
    s = np.exp(-8.0 * d)
    return s / s.sum()


def agg(ps, m, key, f):
    vals = [s[m][key][f] for s in ps if s[m][key][f] is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def main():
    Xtr, ytr, Xte, yte = PC.load(SCRATCH)
    backbones = {n: PC.train(n, Xtr, ytr, Xte, yte, seed=0) for n in ("CNN", "MLP")}
    print("perception ready", flush=True)
    oracle_pi, oracle_Q = {}, {}
    for p in PROFILE_NAMES:
        pi, Q, _ = solve_average(p)
        oracle_pi[p], oracle_Q[p] = pi, Q
    pop_R = population_reward_table()
    news2_pi = news2_policy()

    t0 = time.perf_counter()
    print("\nphase A: people who match one of the four named profiles", flush=True)
    inf = run_phase(CORE_NAMES, backbones, yte, oracle_pi, oracle_Q, pop_R,
                    news2_pi, "in-family")
    print("\nphase B: people who match none of them", flush=True)
    oof = run_phase(HELD_OUT_NAMES, backbones, yte, oracle_pi, oracle_Q, pop_R,
                    news2_pi, "out-of-family")

    FIELDS = ("reward", "crisis_per_100", "alerts_per_100", "users_with_crisis",
              "agreement", "regret_per_100", "sensitivity", "specificity",
              "frailty_error", "preference_error")
    res = {"seeds": SEEDS, "n_users": N_USERS, "steps_per_user": T_STEPS,
           "core_profiles": CORE_NAMES, "held_out_profiles": HELD_OUT_NAMES,
           "frailty": {p: PROFILES[p]["frailty"] for p in PROFILE_NAMES},
           "in_family": {}, "out_of_family": {}}
    for tag, ps in (("in_family", inf), ("out_of_family", oof)):
        pool = CORE_NAMES if tag == "in_family" else HELD_OUT_NAMES
        for m in METHODS + ["Oracle"]:
            res[tag][m] = {}
            for key in ["all"] + pool:
                d = {}
                for f in FIELDS:
                    mu, sd = agg(ps, m, key, f)
                    d[f] = mu
                    d[f + "_sd"] = sd
                res[tag][m][key] = d
        a = np.array([s["TriMedAI"]["all"]["reward"] for s in ps])
        b = np.array([s["QMDP"]["all"]["reward"] for s in ps])
        w, p = stats.wilcoxon(a, b, alternative="two-sided")
        diff = a - b
        res[tag]["_test_trimedai_vs_qmdp"] = {
            "mean_difference": float(np.mean(diff)), "p": float(p),
            "cohens_dz": float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))}

    json.dump(res, open(os.path.join(RESULTS, "har_extended.json"), "w"), indent=1)

    for tag in ("in_family", "out_of_family"):
        pool = CORE_NAMES if tag == "in_family" else HELD_OUT_NAMES
        orc = res[tag]["Oracle"]["all"]["reward"]
        print("\n===== %s  (oracle %.4f) =====" % (tag, orc))
        print("%-10s %7s %8s %8s %9s %8s %8s %8s"
              % ("method", "%orc", "regret", "crisis", "users hit", "sens",
                 "spec", "alerts"))
        for m in METHODS:
            g = res[tag][m]["all"]
            print("%-10s %7.1f %8.3f %8.2f %9.1f %8.1f %8.1f %8.2f"
                  % (m, 100 * g["reward"] / orc, g["regret_per_100"],
                     g["crisis_per_100"], g["users_with_crisis"],
                     g["sensitivity"], g["specificity"], g["alerts_per_100"]))
        t = res[tag]["_test_trimedai_vs_qmdp"]
        print("  TriMedAI vs QMDP: d=%+.4f p=%.2e dz=%+.2f"
              % (t["mean_difference"], t["p"], t["cohens_dz"]))
        fe = res[tag]["TriMedAI"]["all"]["frailty_error"]
        print("  latent recovery: frailty error %.3f, preference error %.3f"
              % (fe, res[tag]["TriMedAI"]["all"]["preference_error"]))
        print("\n  per profile, share of that profile's own oracle:")
        print("  %-28s %7s %8s %8s %8s" % ("profile", "frailty", "CNN", "QMDP",
                                           "TriMedAI"))
        for p in pool:
            po = res[tag]["Oracle"][p]["reward"]
            print("  %-28s %7.2f %8.1f %8.1f %8.1f"
                  % (p, res["frailty"][p],
                     100 * res[tag]["CNN"][p]["reward"] / po,
                     100 * res[tag]["QMDP"][p]["reward"] / po,
                     100 * res[tag]["TriMedAI"][p]["reward"] / po))
    print("\ntotal %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
