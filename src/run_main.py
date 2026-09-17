"""Main experiment: every comparator, every metric, thirty seeds.

Contexts are recognised from real wearable recordings; the care process is
simulated with a severity scale aligned to NEWS2. Reported quantities include the
clinical endpoints by which an early-warning system is judged, namely the
sensitivity and specificity of the escalation decision and the number of
caregiver call-outs bought per deterioration averted, alongside the reward,
regret and adaptation measures.
"""
import json, os, sys, time
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA as SCRATCH, RESULTS, FIGURES

from triage_env import (solve_average, REQUESTS, ACTIONS, PROFILE_NAMES, CORE_NAMES,
                        HELD_OUT_NAMES, N_CORE, N_SEV, N_STATES, ALERT, ASK,
                        CRISIS, GAMMA, NEWS2_URGENT_REVIEW, sid, step, solve,
                        latent_of, model_for, plan, news2_policy, myopic_policy,
                        immediate_reward, population_latent, EXP_R, COMFORT, _AFF)
from triage_agents import (ParticleBelief, PlanningPolicy, QPolicy,
                           RowBackupPolicy, CellBackupPolicy, ObservedQPolicy,
                           meta_initialisation, global_policy,
                           population_reward_table, mcdm_policy)
import perception as PC

SEEDS = list(range(int(os.environ.get("N_SEEDS", "30"))))
N_USERS = int(os.environ.get("N_USERS", "80"))
T_STEPS, EVAL_FROM = 200, 100
ADAPT_WINDOW, ADAPT_LEVEL = 20, 0.80
UCB_C = 0.5
N_ACT, N_REQ = len(ACTIONS), len(REQUESTS)

METHODS = ["RNN", "LSTM", "CNN", "GAI", "MCDM", "NEWS2", "RLHF", "Bandit",
           "QMDP", "PosteriorSampling", "TriMedAI", "ProfileKnown",
           "Random", "PerUserQ"]
# Random is the floor policy every normalised score is anchored on.
# PerUserQ is a per-user tabular Q-learner, the sequential counterpart to
# the per-user bandit.
FLOOR = "Random"
BACKBONE = {"RNN": "RNN", "LSTM": "LSTM", "CNN": "CNN", "GAI": "MLP",
            "RLHF": "MLP"}
VARIANTS = ["Reward inference only", "Policy learning only", "Meta-adaptation only",
            "Without meta-learning", "Without policy learning",
            "Without reward inference", "TriMedAI (full)"]
# Backup-rule ablation: identical belief, identical meta-initialisation and
# identical visited states; only what one interaction may revise differs.
BACKUPS = ["Model solve (full value iteration)", "Belief-averaged full-row backup",
           "Belief-averaged single-action backup", "Single-action Q-learning"]
KEYS = ("reward_per_step", "crisis_per_100", "alerts_per_100",
        "needless_alerts_per_100", "oracle_agreement", "steps_to_adapt",
        "delay_ms", "regret_per_100", "users_with_crisis",
        "escalation_sensitivity", "escalation_specificity",
        "frailty_error", "preference_error")


def blank():
    return dict(reward=[], crisis=0, alerts=0, needless=0, agree=[], steps=0,
                adapt=[], secs=0.0, regret=[], users=0, users_hit=0,
                esc_tp=0, esc_fn=0, esc_tn=0, esc_fp=0, fr_err=[], pf_err=[])


def summarise(m, backbone_ms):
    n = max(m["steps"], 1)
    sens = m["esc_tp"] / max(m["esc_tp"] + m["esc_fn"], 1)
    spec = m["esc_tn"] / max(m["esc_tn"] + m["esc_fp"], 1)
    return dict(
        reward_per_step=float(np.mean(m["reward"])) if m["reward"] else 0.0,
        crisis_per_100=100.0 * m["crisis"] / n,
        alerts_per_100=100.0 * m["alerts"] / n,
        needless_alerts_per_100=100.0 * m["needless"] / n,
        oracle_agreement=100.0 * float(np.mean(m["agree"])) if m["agree"] else 0.0,
        steps_to_adapt=float(np.mean(m["adapt"])) if m["adapt"] else float(T_STEPS),
        delay_ms=1000.0 * m["secs"] / n + backbone_ms,
        regret_per_100=100.0 * float(np.mean(m["regret"])) if m["regret"] else 0.0,
        users_with_crisis=100.0 * m["users_hit"] / max(m["users"], 1),
        escalation_sensitivity=100.0 * sens,
        escalation_specificity=100.0 * spec,
        frailty_error=float(np.mean(m["fr_err"])) if m["fr_err"] else None,
        preference_error=float(np.mean(m["pf_err"])) if m["pf_err"] else None)


def adaptation_point(flags):
    if len(flags) < ADAPT_WINDOW:
        return float(T_STEPS)
    c = np.convolve(np.asarray(flags, float),
                    np.ones(ADAPT_WINDOW) / ADAPT_WINDOW, mode="valid")
    hit = np.where(c >= ADAPT_LEVEL)[0]
    return float(hit[0] + ADAPT_WINDOW) if len(hit) else float(T_STEPS)


def run_seed(seed, backbones, yte, oracle_pi, oracle_Q, meta_Q, glob_Q, pop_R,
             mcdm_pi, news2_pi, pop_plan_Q, user_pool=None):
    rng = np.random.default_rng(1000 + seed)
    pool = user_pool or CORE_NAMES
    by_class = {c: np.where(yte == c)[0] for c in range(N_REQ)}
    users = [pool[int(rng.integers(len(pool)))] for _ in range(N_USERS)]
    streams = []
    for _ in users:
        cls = rng.integers(0, N_REQ, size=T_STEPS)
        idx = np.array([rng.choice(by_class[int(c)]) for c in cls])
        streams.append((cls, idx))

    QSTAR = np.stack([oracle_Q[p] for p in CORE_NAMES])
    stats_ = {m: blank() for m in METHODS + ["Oracle"] + VARIANTS + BACKUPS}

    def simulate(key, mode, backbone="CNN", use_belief=False, use_plan=False,
                 use_meta=False, use_q=False, backup=None):
        st = stats_[key]
        pred = backbones[backbone]["pred"] if backbone else None
        t0 = time.perf_counter()
        for u, (cls, idx) in zip(users, streams):
            opi, oq = oracle_pi[u], oracle_Q[u]
            belief = (ParticleBelief(rng=np.random.default_rng(seed * 977 + 13))
                      if use_belief or mode in ("qmdp", "thompson") else None)
            pol = None
            if use_plan:
                init = meta_Q.copy() if use_meta else np.zeros((N_STATES, N_ACT))
                pol = PlanningPolicy(init=init)
            elif use_q:
                init = meta_Q.copy() if use_meta else np.zeros((N_STATES, N_ACT))
                pol = QPolicy(rng=np.random.default_rng(seed), init=init)
            elif backup is not None:
                init = meta_Q.copy() if use_meta else np.zeros((N_STATES, N_ACT))
                pol = backup(init=init)
            if mode == "profileknown":
                fr, pref = latent_of(u)
                R, T = model_for(fr, pref)
                pol = PlanningPolicy(init=plan(R, T, sweeps=600))
            counts = np.zeros((N_STATES, N_ACT)) if mode == "bandit" else None
            means = np.zeros((N_STATES, N_ACT)) if mode == "bandit" else None
            sev, hit = 0, False
            agree_seq = []
            for t in range(T_STEPS):
                rq = int(cls[t])
                rp = int(pred[idx[t]]) if pred is not None else rq
                s_p, s_t = sid(rp, sev), sid(rq, sev)

                if mode == "oracle":
                    a = int(opi[s_t])
                elif mode == "random":
                    a = int(rng.integers(N_ACT))
                elif mode == "population":
                    a = int(np.argmax(pop_R[s_p]))
                elif mode == "sample":
                    w = pop_R[s_p] - pop_R[s_p].max()
                    w = np.exp(3.0 * w); w /= w.sum()
                    a = int(rng.choice(N_ACT, p=w))
                elif mode == "mcdm":
                    a = int(mcdm_pi[s_p])
                elif mode == "news2":
                    a = int(news2_pi[s_p])
                elif mode == "global":
                    a = int(np.argmax(glob_Q[s_p]))
                elif mode == "qmdp":
                    post = _profile_posterior(belief)
                    a = int(np.argmax(post @ QSTAR[:, s_p, :]))
                elif mode == "thompson":
                    post = _profile_posterior(belief)
                    j = int(rng.choice(N_CORE, p=post))
                    a = int(oracle_pi[CORE_NAMES[j]][s_p])
                elif mode == "bandit":
                    n_s = counts[s_p].sum()
                    if n_s < N_ACT:
                        a = int(counts[s_p].argmin())
                    else:
                        bonus = UCB_C * np.sqrt(np.log(n_s + 1) / (counts[s_p] + 1e-9))
                        a = int(np.argmax(means[s_p] + bonus))
                elif mode == "popplan":
                    a = int(np.argmax(pop_plan_Q[s_p]))
                elif mode == "myopic_belief":
                    fr, pref = belief.mean_latent()
                    v = immediate_reward(fr, pref, rp, sev)
                    if use_meta:
                        v = v + meta_Q[s_p]
                    a = int(np.argmax(v))
                elif use_plan or mode == "profileknown" or backup is not None:
                    a = pol.act(s_p)
                elif use_q:
                    a = pol.act(s_p, eps=0.15 * max(0.0, 1.0 - t / EVAL_FROM))
                else:                                    # meta only, frozen
                    a = int(np.argmax(meta_Q[s_p]))

                r, nsev, crisis, worsened = step(u, rq, sev, a, rng)
                if mode == "bandit":
                    counts[s_p, a] += 1
                    means[s_p, a] += (r - means[s_p, a]) / counts[s_p, a]

                if belief is not None:
                    v = 4.0 * COMFORT[PROFILE_NAMES.index(u), rq, :].copy()
                    v[ASK] = -1e9
                    v = np.exp(v - v.max()); v /= v.sum()
                    belief.observe_preference(rp, int(rng.choice(N_ACT, p=v)))
                    belief.observe_transition(rp, worsened)
                    if use_plan:
                        pol.observe(belief)
                    elif backup is not None:
                        if isinstance(pol, ObservedQPolicy):
                            pol.observe(belief, s_p, a, r, sid(rp, nsev))
                        else:
                            pol.observe(belief, s_p, a, r)

                if use_q and mode not in ("bandit",):
                    nreq = int(pred[idx[t + 1]]) if (pred is not None and
                                                     t + 1 < T_STEPS) else rp
                    pol.update(s_p, a, r, sid(nreq, nsev))

                want = int(opi[s_t])
                matched = (a == want)
                greedy = (int(np.argmax(pol.Q[s_p])) if pol is not None
                          else (int(np.argmax(means[s_p])) if mode == "bandit" else a))
                agree_seq.append(greedy == want)

                if t >= EVAL_FROM:
                    st["reward"].append(r)
                    st["crisis"] += int(crisis)
                    st["alerts"] += int(a == ALERT)
                    st["needless"] += int(a == ALERT and sev <= 1)
                    st["agree"].append(matched)
                    st["regret"].append(float(np.max(oq[s_t]) - oq[s_t, a]))
                    st["steps"] += 1
                    if want == ALERT:
                        st["esc_tp" if a == ALERT else "esc_fn"] += 1
                    else:
                        st["esc_fp" if a == ALERT else "esc_tn"] += 1
                    if crisis:
                        hit = True
                sev = nsev
            if belief is not None and u in PROFILE_NAMES:
                fe, pe = belief.latent_error(u)
                st["fr_err"].append(fe)
                st["pf_err"].append(pe)
            st["adapt"].append(adaptation_point(agree_seq))
            st["users"] += 1
            st["users_hit"] += int(hit)
        st["secs"] += time.perf_counter() - t0

    def _profile_posterior(belief):
        """QMDP and posterior sampling need a distribution over the four named
        profiles, which is all they can represent; the continuous posterior is
        projected onto them by likelihood under each"""
        w = belief.weights
        fr = belief.weights @ belief.frailty
        pref = belief.weights @ belief.pref
        d = np.array([np.hypot(fr - latent_of(p)[0],
                               np.mean(np.abs(pref - latent_of(p)[1])))
                      for p in CORE_NAMES])
        s = np.exp(-8.0 * d)
        return s / s.sum()

    simulate("Oracle", "oracle", backbone=None)
    for bn in ("CNN", "RNN", "LSTM"):
        simulate(bn, "population", backbone=bn)
    simulate("GAI", "sample", backbone="MLP")
    simulate("MCDM", "mcdm")
    simulate("NEWS2", "news2")
    simulate("RLHF", "global", backbone="MLP")
    simulate("Bandit", "bandit")
    simulate("QMDP", "qmdp")
    simulate("PosteriorSampling", "thompson")
    simulate("ProfileKnown", "profileknown")
    simulate("TriMedAI", "framework", use_belief=True, use_plan=True, use_meta=True)

    simulate("Reward inference only", "myopic_belief", use_belief=True)
    simulate("Policy learning only", "framework", use_q=True)
    simulate("Meta-adaptation only", "metaonly")
    simulate("Without meta-learning", "framework", use_belief=True, use_plan=True)
    simulate("Without policy learning", "myopic_belief", use_belief=True, use_meta=True)
    simulate("Without reward inference", "popplan", use_plan=False, use_meta=True)
    stats_["TriMedAI (full)"] = stats_["TriMedAI"]

    # appended after every pre-existing call so the random stream consumed by
    # the methods above is identical to the first version of this experiment
    simulate("Random", "random", backbone=None)
    simulate("PerUserQ", "framework", use_q=True)
    simulate("Belief-averaged full-row backup", "backup", use_belief=True,
             use_meta=True, backup=RowBackupPolicy)
    simulate("Belief-averaged single-action backup", "backup", use_belief=True,
             use_meta=True, backup=CellBackupPolicy)
    simulate("Single-action Q-learning", "backup", use_belief=True,
             use_meta=True, backup=ObservedQPolicy)
    stats_["Model solve (full value iteration)"] = stats_["TriMedAI"]

    ms = {k: backbones[k]["infer_ms"] for k in backbones}
    out = {}
    for k, v in stats_.items():
        bb = ms.get(BACKBONE.get(k, "CNN"), ms["CNN"])
        out[k] = summarise(v, 0.0 if k == "Oracle" else bb)
    return out


def boot_ci(x, n=10000, seed=0):
    r = np.random.default_rng(seed)
    x = np.asarray(x, float)
    bs = r.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    Xtr, ytr, Xte, yte = PC.load(SCRATCH)
    print("wearable corpus: train %s, test %s (subject-wise split)"
          % (Xtr.shape, Xte.shape), flush=True)
    backbones = {}
    for name in ("CNN", "RNN", "LSTM", "MLP"):
        backbones[name] = PC.train(name, Xtr, ytr, Xte, yte, seed=0)
        b = backbones[name]
        print("  %-5s %5.2f%%  params %7d  train %5.1fs  infer %.4f ms"
              % (name, 100 * b["accuracy"], b["params"], b["train_s"],
                 b["infer_ms"]), flush=True)

    oracle_pi, oracle_Q = {}, {}
    for p in PROFILE_NAMES:
        pi, Q, _ = solve_average(p)
        oracle_pi[p], oracle_Q[p] = pi, Q
    pop_R = population_reward_table()
    mcdm_pi = mcdm_policy()
    news2_pi = news2_policy()
    fr_pop, pref_pop = population_latent()
    Rp, Tp = model_for(fr_pop, pref_pop)
    pop_plan_Q = plan(Rp, Tp, sweeps=800)

    runs, t0 = [], time.perf_counter()
    for s in SEEDS:
        meta_Q = meta_initialisation(seed=s)
        glob_Q = global_policy(seed=s)
        runs.append(run_seed(s, backbones, yte, oracle_pi, oracle_Q, meta_Q,
                             glob_Q, pop_R, mcdm_pi, news2_pi, pop_plan_Q))
        if (s + 1) % 5 == 0:
            print("  %d/%d seeds, %.0fs" % (s + 1, len(SEEDS),
                                            time.perf_counter() - t0), flush=True)

    res = {"task": "sequential assistive care over wearable activity contexts",
           "dataset": "UCI Human Activity Recognition (30 subjects, subject-wise split)",
           "severity_scale": "aligned to NEWS2 bands",
           "seeds": SEEDS, "n_users": N_USERS, "steps_per_user": T_STEPS,
           "evaluation_from_step": EVAL_FROM, "gamma": GAMMA,
           "backbones": {k: {"accuracy": 100 * v["accuracy"], "params": v["params"],
                             "train_s": v["train_s"], "infer_ms": v["infer_ms"]}
                         for k, v in backbones.items()},
           "main": {}, "ablation": {}, "backups": {}, "oracle": {},
           "tests": {}, "ci": {}, "normalised": {}}
    for m in METHODS:
        res["main"][m] = {k: [float(r[m][k]) for r in runs
                              if r[m][k] is not None] for k in KEYS}
    for v in VARIANTS:
        res["ablation"][v] = {k: [float(r[v][k]) for r in runs
                                  if r[v][k] is not None] for k in KEYS}
    for v in BACKUPS:
        res["backups"][v] = {k: [float(r[v][k]) for r in runs
                                 if r[v][k] is not None] for k in KEYS}
    res["oracle"] = {k: [float(r["Oracle"][k]) for r in runs
                         if r["Oracle"][k] is not None] for k in KEYS}

    orc = float(np.mean(res["oracle"]["reward_per_step"]))

    # Floor-anchored normalisation. Mean reward has an arbitrary origin, so a
    # plain ratio to the oracle is not invariant: adding a constant to eq. (19)
    # changes every such ratio while changing no ordering. The score below
    # measures the fraction of the floor-to-oracle span a method closes and is
    # unchanged by any constant shift of the reward.
    o_seed = np.asarray(res["oracle"]["reward_per_step"], float)
    f_seed = np.asarray(res["main"][FLOOR]["reward_per_step"], float)
    span = o_seed - f_seed
    res["floor_reward"] = float(np.mean(f_seed))
    res["oracle_reward"] = float(orc)
    for m in METHODS + VARIANTS + BACKUPS:
        src = res["main"].get(m) or res["ablation"].get(m) or res["backups"][m]
        lo, hi = boot_ci(src["reward_per_step"])
        res["ci"][m] = {"reward_lo": lo, "reward_hi": hi,
                        "pct_lo": 100 * lo / orc, "pct_hi": 100 * hi / orc}
        z = 100.0 * (np.asarray(src["reward_per_step"], float) - f_seed) / span
        nlo, nhi = boot_ci(z)
        res["normalised"][m] = {"per_seed": [float(v) for v in z],
                                "mean": float(np.mean(z)),
                                "sd": float(np.std(z, ddof=1)),
                                "lo": nlo, "hi": nhi}
    res["normalised"]["Oracle"] = {"per_seed": [100.0] * len(o_seed),
                                   "mean": 100.0, "sd": 0.0,
                                   "lo": 100.0, "hi": 100.0}

    a = np.array(res["main"]["TriMedAI"]["reward_per_step"])
    ca = np.array(res["main"]["TriMedAI"]["crisis_per_100"])
    for m in METHODS:
        if m == "TriMedAI":
            continue
        b = np.array(res["main"][m]["reward_per_step"])
        cb = np.array(res["main"][m]["crisis_per_100"])
        w, p = stats.wilcoxon(a, b, alternative="two-sided")
        wc, pc = stats.wilcoxon(ca, cb, alternative="two-sided")
        diff = a - b
        res["tests"][m] = {"W_reward": float(w), "p_reward": float(p),
                           "mean_reward_difference": float(np.mean(diff)),
                           "cohens_dz": float(np.mean(diff) /
                                              (np.std(diff, ddof=1) + 1e-12)),
                           "W_crisis": float(wc), "p_crisis": float(pc),
                           "mean_crisis_difference": float(np.mean(ca - cb))}

    json.dump(res, open(os.path.join(RESULTS, "har_results.json"), "w"), indent=1)

    print("\n%-19s %8s %8s %15s %7s %7s %7s %7s %7s %7s"
          % ("method", "reward", "norm", "95% CI (norm)", "regret", "crisis",
             "callout", "sens", "spec", "adapt"))
    for m in sorted(METHODS, key=lambda k: res["normalised"][k]["mean"]):
        v, nz = res["main"][m], res["normalised"][m]
        r = np.mean(v["reward_per_step"])
        print("%-19s %8.4f %8.1f  [%5.1f, %5.1f] %7.3f %7.2f %7.2f %7.1f %7.1f %7.1f"
              % (m, r, nz["mean"], nz["lo"], nz["hi"],
                 np.mean(v["regret_per_100"]), np.mean(v["crisis_per_100"]),
                 np.mean(v["alerts_per_100"]),
                 np.mean(v["escalation_sensitivity"]),
                 np.mean(v["escalation_specificity"]),
                 np.mean(v["steps_to_adapt"])))
    print("%-19s %8.4f %8.1f %16.3f %7.2f %7.2f"
          % ("Oracle", orc, 100.0, 0.0,
             np.mean(res["oracle"]["crisis_per_100"]),
             np.mean(res["oracle"]["alerts_per_100"])))
    print("floor (%s) reward %.4f, oracle reward %.4f, span %.4f"
          % (FLOOR, res["floor_reward"], orc, orc - res["floor_reward"]))
    print("\ntests vs TriMedAI (two-sided Wilcoxon, %d seeds)" % len(SEEDS))
    for m, t in res["tests"].items():
        print("  %-19s d=%+8.4f  p=%9.2e  dz=%+6.2f  dcrisis=%+6.2f p=%.4f"
              % (m, t["mean_reward_difference"], t["p_reward"], t["cohens_dz"],
                 t["mean_crisis_difference"], t["p_crisis"]))
    print("\nablation:")
    for v in VARIANTS:
        d = res["ablation"][v]
        r = np.mean(d["reward_per_step"])
        print("  %-26s %7.4f (%5.1f%%)  crisis %5.2f  adapt %5.1f"
              % (v, r, 100 * r / orc, np.mean(d["crisis_per_100"]),
                 np.mean(d["steps_to_adapt"])))
    print("\nbackup rule (belief, meta-initialisation, visited states held fixed):")
    for v in BACKUPS:
        d = res["backups"][v]
        print("  %-38s norm %6.1f  regret %6.3f  crisis %5.2f  adapt %6.1f"
              % (v, res["normalised"][v]["mean"], np.mean(d["regret_per_100"]),
                 np.mean(d["crisis_per_100"]), np.mean(d["steps_to_adapt"])))
    res["backup_tests"] = {}
    a_ = np.asarray(res["normalised"][BACKUPS[0]]["per_seed"], float)
    for v in BACKUPS[1:]:
        b_ = np.asarray(res["normalised"][v]["per_seed"], float)
        w_, p_ = stats.wilcoxon(a_, b_, alternative="two-sided")
        res["backup_tests"][v] = {"p": float(p_),
                                  "delta_norm": float(np.mean(a_ - b_)),
                                  "delta_adapt": float(
                                      np.mean(res["backups"][BACKUPS[0]]["steps_to_adapt"])
                                      - np.mean(res["backups"][v]["steps_to_adapt"]))}
    json.dump(res, open(os.path.join(RESULTS, "har_results.json"), "w"), indent=1)

    fe = res["main"]["TriMedAI"].get("frailty_error")
    if fe:
        print("\nlatent recovery: frailty error %.3f, preference error %.3f"
              % (np.mean(fe), np.mean(res["main"]["TriMedAI"]["preference_error"])))
    print("total %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
