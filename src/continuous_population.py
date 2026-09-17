"""Personalisation on a continuum, rather than among four named profiles.

A reviewer observed that although the framework describes a continuous latent,
the population in the main experiment is generated from four named profiles, so
what the experiment shows could be interpolation among a small synthetic set
rather than personalisation over a continuum.

This script removes the named profiles from the generative process altogether.
Each simulated person is drawn independently from the prior the agent carries:
a frailty uniform on [0,1] and a weight uniform on [0,1] for each delivery
channel. No two users share a latent, none of them is one of the four named
profiles, and the reference policy is solved separately for every individual.

Three things are measured.

  * whether the framework still attains the same share of what is attainable
    when nobody in the population is a named type;
  * how the error in the recovered frailty behaves across the whole range of
    frailty, rather than at four points;
  * how each method's performance depends on the distance from the person's
    latent to the nearest named profile. A method that carries a discrete
    hypothesis set should degrade with that distance and a method that carries a
    continuous latent should not. That contrast is the evidence the reviewer
    asked for, and it cannot be produced from a population of four types.

    python continuous_population.py
    N_SEEDS=3 N_USERS=20 python continuous_population.py
"""
import json, os, sys, time
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA, RESULTS

import triage_env as TE
from triage_env import (REQUESTS, ACTIONS, CORE_NAMES, N_CORE, N_SEV, N_STATES,
                        ALERT, ASK, sid, latent_of, model_for, plan,
                        plan_average, news2_policy, population_latent)
from triage_agents import (ParticleBelief, PlanningPolicy, meta_initialisation,
                           population_reward_table)
import perception as PC

N_ACT, N_REQ = len(ACTIONS), len(REQUESTS)
SEEDS = list(range(int(os.environ.get("N_SEEDS", "15"))))
N_USERS = int(os.environ.get("N_USERS", "60"))
T_STEPS, EVAL_FROM = 200, 100
METHODS = ["Oracle", "Random", "Population rule", "Fixed threshold",
           "QMDP", "TriMedAI"]


def register(name, frailty, pref):
    """add a synthetic individual to the environment's profile dictionary so
    that the sampler can be driven from an arbitrary continuous latent"""
    TE.PROFILES[name] = dict(
        pref={a: float(pref[i]) for i, a in enumerate(ACTIONS)},
        frailty=float(frailty))
    return name


def user_tables(name):
    """sampling table for one individual, built from triage_env.outcomes so that
    the simulator and the exact solver cannot drift apart"""
    tab = {}
    for rq in range(N_REQ):
        for sv in range(N_SEV):
            for a in range(N_ACT):
                o = TE.outcomes(name, rq, sv, a)
                cum = np.cumsum([x[0] for x in o])
                cum = cum / cum[-1]
                tab[(rq, sv, a)] = (cum,
                                    np.array([x[1] for x in o]),
                                    np.array([x[2] for x in o], dtype=np.int64),
                                    np.array([x[3] for x in o], dtype=bool),
                                    np.array([x[4] for x in o], dtype=bool))
    return tab


def sample(tab, rq, sv, a, rng):
    cum, rew, nxt, cri, wor = tab[(rq, sv, a)]
    i = min(int(np.searchsorted(cum, rng.random())), len(rew) - 1)
    return float(rew[i]), int(nxt[i]), bool(cri[i]), bool(wor[i])


def nearest_named(frailty, pref):
    """distance from a latent to the closest of the four named profiles, on the
    same scale the QMDP projection uses"""
    d = [np.hypot(frailty - latent_of(p)[0],
                  float(np.mean(np.abs(pref - latent_of(p)[1]))))
         for p in CORE_NAMES]
    return float(min(d))


def main():
    Xtr, ytr, Xte, yte = PC.load(DATA)
    cnn = PC.train("CNN", Xtr, ytr, Xte, yte, seed=0)
    pred = cnn["pred"]
    print("context recognition %.2f%% on held-out subjects"
          % (100 * cnn["accuracy"]), flush=True)

    QSTAR = np.stack([plan(*model_for(*latent_of(p)), sweeps=3000)
                      for p in CORE_NAMES])
    pop_R = population_reward_table()
    fixed_pi = news2_policy()
    by_class = {c: np.where(yte == c)[0] for c in range(N_REQ)}

    rows = []        # one record per simulated person
    per_seed = {m: {"reward": [], "regret": [], "crisis": []} for m in METHODS}
    t0 = time.perf_counter()

    for seed in SEEDS:
        rng = np.random.default_rng(7000 + seed)
        meta_Q = meta_initialisation(seed=seed)
        acc = {m: dict(rew=[], reg=[], cri=0, alerts=0, n=0) for m in METHODS}

        for k in range(N_USERS):
            fr = float(rng.random())
            pref = rng.random(N_ACT)
            name = register("__c%d_%d__" % (seed, k), fr, pref)
            tab = user_tables(name)
            R, T = model_for(fr, pref)
            oq, gain = plan_average(R, T)
            opi = oq.argmax(axis=1)
            dist = nearest_named(fr, pref)

            cls = rng.integers(0, N_REQ, size=T_STEPS)
            idx = np.array([rng.choice(by_class[int(c)]) for c in cls])

            rec = {"frailty": fr, "distance_to_nearest_named": dist,
                   "oracle_gain": float(gain)}
            for m in METHODS:
                belief = (ParticleBelief(rng=np.random.default_rng(seed * 977 + k))
                          if m in ("TriMedAI", "QMDP") else None)
                pol = PlanningPolicy(init=meta_Q.copy()) if m == "TriMedAI" else None
                sev = 0
                rew, reg, cri, alerts, n = [], [], 0, 0, 0
                for t in range(T_STEPS):
                    rq = int(cls[t])
                    rp = rq if m == "Oracle" else int(pred[idx[t]])
                    s_p, s_t = sid(rp, sev), sid(rq, sev)
                    if m == "Oracle":
                        a = int(opi[s_t])
                    elif m == "Random":
                        a = int(rng.integers(N_ACT))
                    elif m == "Population rule":
                        a = int(np.argmax(pop_R[s_p]))
                    elif m == "Fixed threshold":
                        a = int(fixed_pi[s_p])
                    elif m == "QMDP":
                        w = belief.weights
                        bf, bp = float(w @ belief.frailty), w @ belief.pref
                        d = np.array([np.hypot(bf - latent_of(p)[0],
                                               np.mean(np.abs(bp - latent_of(p)[1])))
                                      for p in CORE_NAMES])
                        post = np.exp(-8.0 * d)
                        post /= post.sum()
                        a = int(np.argmax(post @ QSTAR[:, s_p, :]))
                    else:
                        a = pol.act(s_p)

                    r, nsev, crisis, worsened = sample(tab, rq, sev, a, rng)
                    if belief is not None:
                        c = TE._AFF[rq] * (0.35 + 0.65 * pref)
                        v = 4.0 * c
                        v[ASK] = -1e9
                        v = np.exp(v - v.max())
                        v /= v.sum()
                        belief.observe_preference(rp, int(rng.choice(N_ACT, p=v)))
                        belief.observe_transition(rp, worsened)
                        if pol is not None:
                            pol.observe(belief)
                    if t >= EVAL_FROM:
                        rew.append(r)
                        reg.append(float(np.max(oq[s_t]) - oq[s_t, a]))
                        cri += int(crisis)
                        alerts += int(a == ALERT)
                        n += 1
                    sev = nsev
                rec[m] = {"reward": float(np.mean(rew)),
                          "regret": 100.0 * float(np.mean(reg)),
                          "crisis": 100.0 * cri / max(n, 1),
                          "alerts": 100.0 * alerts / max(n, 1)}
                if belief is not None:
                    bf = float(belief.weights @ belief.frailty)
                    bp = belief.weights @ belief.pref
                    rec[m]["frailty_error"] = abs(bf - fr)
                    rec[m]["frailty_estimate"] = bf
                    rec[m]["preference_error"] = float(np.mean(np.abs(bp - pref)))
                acc[m]["rew"].extend(rew)
                acc[m]["reg"].extend(reg)
                acc[m]["cri"] += cri
                acc[m]["alerts"] += alerts
                acc[m]["n"] += n
            rows.append(rec)
            del TE.PROFILES[name]

        for m in METHODS:
            a = acc[m]
            per_seed[m]["reward"].append(float(np.mean(a["rew"])))
            per_seed[m]["regret"].append(100.0 * float(np.mean(a["reg"])))
            per_seed[m]["crisis"].append(100.0 * a["cri"] / max(a["n"], 1))
        print("  seed %d/%d, %.0fs" % (seed + 1, len(SEEDS),
                                       time.perf_counter() - t0), flush=True)

    o = np.asarray(per_seed["Oracle"]["reward"])
    f = np.asarray(per_seed["Random"]["reward"])
    span = o - f
    summary = {}
    for m in METHODS:
        x = np.asarray(per_seed[m]["reward"])
        z = 100.0 * (x - f) / span
        summary[m] = {"reward": float(np.mean(x)),
                      "norm": float(np.mean(z)), "norm_sd": float(np.std(z, ddof=1)),
                      "regret": float(np.mean(per_seed[m]["regret"])),
                      "regret_sd": float(np.std(per_seed[m]["regret"], ddof=1)),
                      "crisis": float(np.mean(per_seed[m]["crisis"]))}

    # per-person normalised score, so that performance can be related to how far
    # the person sits from the nearest named profile
    fl = {m: None for m in METHODS}
    dist = np.array([r["distance_to_nearest_named"] for r in rows])
    per_user = {}
    orc_u = np.array([r["Oracle"]["reward"] for r in rows])
    flr_u = np.array([r["Random"]["reward"] for r in rows])
    sp_u = orc_u - flr_u
    for m in METHODS:
        xu = np.array([r[m]["reward"] for r in rows])
        per_user[m] = 100.0 * (xu - flr_u) / sp_u

    tert = np.quantile(dist, [1 / 3, 2 / 3])
    band = np.digitize(dist, tert)
    bands = {}
    for m in ("Population rule", "Fixed threshold", "QMDP", "TriMedAI"):
        bands[m] = [float(np.mean(per_user[m][band == b])) for b in range(3)]
    slopes = {}
    for m in ("QMDP", "TriMedAI"):
        sl, ic, r_, p_, se = stats.linregress(dist, per_user[m])
        slopes[m] = {"slope_per_unit_distance": float(sl), "p": float(p_),
                     "r": float(r_)}

    fr_all = np.array([r["frailty"] for r in rows])
    fr_hat = np.array([r["TriMedAI"]["frailty_estimate"] for r in rows])
    fr_err = np.array([r["TriMedAI"]["frailty_error"] for r in rows])
    fq = np.quantile(fr_all, [0.2, 0.4, 0.6, 0.8])
    fb = np.digitize(fr_all, fq)

    blob = {"seeds": SEEDS, "n_users_per_seed": N_USERS,
            "steps_per_user": T_STEPS, "evaluation_from_step": EVAL_FROM,
            "population": "latent drawn independently per person: frailty ~ U(0,1), "
                          "channel weights ~ U(0,1) each; no named profiles",
            "summary": summary,
            "distance_tertiles": [float(v) for v in tert],
            "normalised_by_distance_tertile": bands,
            "distance_slope": slopes,
            "frailty_recovery": {
                "correlation": float(np.corrcoef(fr_all, fr_hat)[0, 1]),
                "mean_absolute_error": float(np.mean(fr_err)),
                "by_quintile": [{"frailty_range": [float(fr_all[fb == b].min()),
                                                   float(fr_all[fb == b].max())],
                                 "mean_absolute_error": float(np.mean(fr_err[fb == b]))}
                                for b in range(5)]},
            "n_people": len(rows)}
    json.dump(blob, open(os.path.join(RESULTS, "har_continuous.json"), "w"), indent=1)

    print("\n%d people, each with their own latent and their own reference policy"
          % len(rows))
    print("%-18s %8s %8s %8s %8s" % ("method", "reward", "norm", "regret", "crisis"))
    for m in METHODS:
        s = summary[m]
        print("%-18s %8.4f %8.1f %8.3f %8.2f"
              % (m, s["reward"], s["norm"], s["regret"], s["crisis"]))
    print("\nnormalised score by distance from the nearest named profile")
    print("%-18s %10s %10s %10s" % ("method", "near", "middle", "far"))
    for m, v in bands.items():
        print("%-18s %10.1f %10.1f %10.1f" % (m, v[0], v[1], v[2]))
    for m, v in slopes.items():
        print("  %-10s slope %+.2f points per unit distance (p = %.2g)"
              % (m, v["slope_per_unit_distance"], v["p"]))
    print("\nfrailty recovered across the whole range: r = %.3f, mean error %.3f"
          % (blob["frailty_recovery"]["correlation"],
             blob["frailty_recovery"]["mean_absolute_error"]))
    for q in blob["frailty_recovery"]["by_quintile"]:
        print("   frailty %.2f-%.2f  error %.3f"
              % (q["frailty_range"][0], q["frailty_range"][1],
                 q["mean_absolute_error"]))
    print("\nwritten to results/har_continuous.json, %.0f s"
          % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
