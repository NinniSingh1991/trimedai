"""Per-interaction learning curves on the wearable assistive-care task."""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA as SCRATCH, RESULTS, FIGURES

from triage_env import (solve_average, REQUESTS, ACTIONS, PROFILE_NAMES, CORE_NAMES,
                        ALERT, ASK, sid, step, solve, news2_policy, COMFORT)
from triage_agents import (ParticleBelief, PlanningPolicy, meta_initialisation,
                           global_policy, population_reward_table, mcdm_policy)
import perception as PC

SEEDS = list(range(int(os.environ.get("N_SEEDS", "30"))))
N_USERS = int(os.environ.get("N_USERS", "60"))
BLOCK, N_BLOCKS = 10, 21
T_STEPS = BLOCK * N_BLOCKS
N_ACT, N_REQ = len(ACTIONS), len(REQUESTS)
METHODS = ["CNN", "LSTM", "RNN", "MCDM", "NEWS2", "RLHF", "TriMedAI", "Random"]
FLOOR = "Random"          # the anchor of the normalised score
BACK = {"CNN": "CNN", "LSTM": "LSTM", "RNN": "RNN", "MCDM": "CNN",
        "NEWS2": "CNN", "RLHF": "MLP", "TriMedAI": "CNN", "Random": "CNN"}


def main():
    Xtr, ytr, Xte, yte = PC.load(SCRATCH)
    backbones = {n: PC.train(n, Xtr, ytr, Xte, yte, seed=0)
                 for n in ("CNN", "RNN", "LSTM", "MLP")}
    for n, b in backbones.items():
        print("  %-5s %.2f%%" % (n, 100 * b["accuracy"]), flush=True)

    oracle_pi, oracle_Q = {}, {}
    for p in CORE_NAMES:
        pi, Q, _ = solve_average(p)
        oracle_pi[p], oracle_Q[p] = pi, Q
    pop_R = population_reward_table()
    mcdm_pi = mcdm_policy()
    news2_pi = news2_policy()
    by_class = {c: np.where(yte == c)[0] for c in range(N_REQ)}

    curves = {m: {k: np.zeros((len(SEEDS), N_BLOCKS))
                  for k in ("reward", "agreement", "crisis")} for m in METHODS}
    oracle_curve = np.zeros((len(SEEDS), N_BLOCKS))

    t0 = time.perf_counter()
    for si, seed in enumerate(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        meta_Q = meta_initialisation(seed=seed)
        glob_Q = global_policy(seed=seed)
        users = [CORE_NAMES[int(rng.integers(len(CORE_NAMES)))]
                 for _ in range(N_USERS)]
        streams = []
        for _ in users:
            cls = rng.integers(0, N_REQ, size=T_STEPS)
            idx = np.array([rng.choice(by_class[int(c)]) for c in cls])
            streams.append((cls, idx))

        for m in METHODS + ["Oracle"]:
            rew = np.zeros(N_BLOCKS); agr = np.zeros(N_BLOCKS)
            cri = np.zeros(N_BLOCKS)
            pred = backbones[BACK.get(m, "CNN")]["pred"] if m != "Oracle" else None
            for u, (cls, idx) in zip(users, streams):
                ui = PROFILE_NAMES.index(u)
                opi = oracle_pi[u]
                belief = ParticleBelief(rng=np.random.default_rng(seed * 977 + 3)) \
                    if m == "TriMedAI" else None
                pol = PlanningPolicy(init=meta_Q.copy()) if m == "TriMedAI" else None
                sev = 0
                for t in range(T_STEPS):
                    rq = int(cls[t])
                    rp = rq if m == "Oracle" else int(pred[idx[t]])
                    s_p, s_t = sid(rp, sev), sid(rq, sev)
                    if m == "Oracle":
                        a = int(opi[s_t])
                    elif m in ("CNN", "LSTM", "RNN"):
                        a = int(np.argmax(pop_R[s_p]))
                    elif m == "MCDM":
                        a = int(mcdm_pi[s_p])
                    elif m == "NEWS2":
                        a = int(news2_pi[s_p])
                    elif m == "RLHF":
                        a = int(np.argmax(glob_Q[s_p]))
                    elif m == "Random":
                        a = int(rng.integers(N_ACT))
                    else:
                        a = pol.act(s_p)
                    r, nsev, crisis, worsened = step(u, rq, sev, a, rng)
                    if belief is not None:
                        v = 4.0 * COMFORT[ui, rq, :].copy()
                        v[ASK] = -1e9
                        v = np.exp(v - v.max()); v /= v.sum()
                        belief.observe_preference(rp, int(rng.choice(N_ACT, p=v)))
                        belief.observe_transition(rp, worsened)
                        pol.observe(belief)
                    b = t // BLOCK
                    rew[b] += r
                    cri[b] += int(crisis)
                    agr[b] += int(a == int(opi[s_t]))
                    sev = nsev
            n = N_USERS * BLOCK
            if m == "Oracle":
                oracle_curve[si] = rew / n
            else:
                curves[m]["reward"][si] = rew / n
                curves[m]["agreement"][si] = 100.0 * agr / n
                curves[m]["crisis"][si] = 100.0 * cri / n
        if (si + 1) % 5 == 0:
            print("  %d/%d seeds, %.0fs" % (si + 1, len(SEEDS),
                                            time.perf_counter() - t0), flush=True)

    # floor-anchored normalised score, paired by seed and by block
    span = oracle_curve - curves[FLOOR]["reward"]
    norm = {m: 100.0 * (curves[m]["reward"] - curves[FLOOR]["reward"]) / span
            for m in METHODS}
    out = {"block_size": BLOCK, "n_blocks": N_BLOCKS,
           "interaction_index": [i * BLOCK for i in range(N_BLOCKS)],
           "methods": METHODS, "floor": FLOOR, "seeds": len(SEEDS),
           "n_users": N_USERS,
           "oracle_reward": [round(float(v), 4) for v in oracle_curve.mean(axis=0)],
           "oracle_reward_sd": [round(float(v), 4) for v in oracle_curve.std(axis=0, ddof=1)],
           "curves": {m: {k: [round(float(v), 4) for v in curves[m][k].mean(axis=0)]
                          for k in curves[m]} for m in METHODS},
           "curves_sd": {m: {k: [round(float(v), 4)
                                 for v in curves[m][k].std(axis=0, ddof=1)]
                             for k in curves[m]} for m in METHODS},
           "normalised": {m: [round(float(v), 3) for v in norm[m].mean(axis=0)]
                          for m in METHODS},
           "normalised_sd": {m: [round(float(v), 3)
                                 for v in norm[m].std(axis=0, ddof=1)]
                             for m in METHODS}}
    json.dump(out, open(os.path.join(RESULTS, "har_curves.json"), "w"), indent=1)

    print("\nreward per interaction, by block")
    cols = range(0, N_BLOCKS, 4)
    print("%-10s %s" % ("block", " ".join("%7d" % (i * BLOCK) for i in cols)))
    for m in METHODS:
        print("%-10s %s" % (m, " ".join("%7.3f" % out["curves"][m]["reward"][i]
                                        for i in cols)))
    print("%-10s %s" % ("Oracle", " ".join("%7.3f" % out["oracle_reward"][i]
                                           for i in cols)))


if __name__ == "__main__":
    main()
