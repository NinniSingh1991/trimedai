"""Decision latency and capacity under multi-user load.

One service process holds N users at once. Each user has the framework's own state: a 256-particle posterior, a
warm-started value table and a re-plan every tenth interaction, exactly as in the main experiments. A round serves
one interaction to every user in turn (round robin): recognise the context window, choose the action, observe the
two signals, update the posterior and, when due, re-solve the model. Every decision is timed individually.
Recognition is timed separately as one batched forward pass of the main CNN over the round's N windows.

Reported for N = 1, 10, 100, 1,000 and 10,000 users: mean, median, 95th and 99th percentile decision latency,
recognition time per window, end-to-end time per round, the memory held per user, and the number of users one
process can serve in real time when each user produces one window every 2.56 s. A second part runs P = 1, 2, 4, 8
and 12 worker processes, each serving its own share of 1,000 users, to measure how throughput scales across cores.
Run on an otherwise idle machine; nothing else was running.
"""
import json, os, sys, time, platform
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE); OUT = os.path.join(ROOT, "results")
import triage_env as ENV
from triage_env import REQUESTS, ACTIONS, PROFILE_NAMES, CORE_NAMES, ASK, sid, step
from triage_agents import ParticleBelief, PlanningPolicy, meta_initialisation

N_ACT, N_REQ = len(ACTIONS), len(REQUESTS)
WINDOW_S = 2.56


def serve(n_users, rounds, seed=0, meta_Q=None, time_each=True):
    rng = np.random.default_rng(seed)
    meta_Q = meta_initialisation(seed=0, n_draws=16) if meta_Q is None else meta_Q
    users = [CORE_NAMES[int(rng.integers(len(CORE_NAMES)))] for _ in range(n_users)]
    beliefs = [ParticleBelief(rng=np.random.default_rng(seed * 100003 + i)) for i in range(n_users)]
    pols = [PlanningPolicy(init=meta_Q.copy()) for _ in range(n_users)]
    sev = np.zeros(n_users, int)
    lat = []
    t_all = time.perf_counter()
    for _ in range(rounds):
        ctx = rng.integers(0, N_REQ, size=n_users)
        for i in range(n_users):
            u = users[i]; ui = PROFILE_NAMES.index(u); rq = int(ctx[i])
            # the person's response is simulated outside the timed region
            t0 = time.perf_counter()
            a = pols[i].act(sid(rq, sev[i]))
            t1 = time.perf_counter()
            r, nsev, crisis, worsened = step(u, rq, sev[i], a, rng)
            v = 4.0 * ENV.COMFORT[ui, rq, :].copy(); v[ASK] = -1e9
            v = np.exp(v - v.max()); v /= v.sum(); pref = int(rng.choice(N_ACT, p=v))
            t2 = time.perf_counter()
            beliefs[i].observe_preference(rq, pref)
            beliefs[i].observe_transition(rq, worsened)
            pols[i].observe(beliefs[i])
            t3 = time.perf_counter()
            if time_each:
                lat.append((t1 - t0) + (t3 - t2))
            sev[i] = nsev
    wall = time.perf_counter() - t_all
    per_user_bytes = (beliefs[0].frailty.nbytes + beliefs[0].pref.nbytes + beliefs[0].logw.nbytes + pols[0].Q.nbytes)
    return np.array(lat), wall, per_user_bytes


def recognition_cost(batch_sizes):
    import torch, perception as PC
    torch.set_num_threads(1)
    net = PC.CNN().eval()
    out = {}
    for n in batch_sizes:
        x = torch.randn(n, 128, 9)
        with torch.no_grad():
            net(x[: min(n, 8)])
            reps = max(3, int(2000 / max(n, 1)))
            t0 = time.perf_counter()
            for _ in range(reps):
                net(x)
            out[n] = (time.perf_counter() - t0) / reps / n * 1000.0      # ms per window
    return out


def _worker(args):
    n, rounds, seed = args
    import numpy as _np
    meta_Q = meta_initialisation(seed=0, n_draws=16)
    t0 = time.perf_counter(); lat, wall, _ = serve(n, rounds, seed=seed, meta_Q=meta_Q, time_each=False)
    return n * rounds, time.perf_counter() - t0


def main():
    res = {"machine": {"processor": platform.processor(), "python": platform.python_version(),
                       "logical_cores": os.cpu_count()}, "single_process": {}, "multi_process": {}}
    meta_Q = meta_initialisation(seed=0, n_draws=16)
    rec = recognition_cost([1, 10, 100, 1000, 10000])
    res["recognition_ms_per_window"] = rec
    for n, rounds in [(1, 2000), (10, 300), (100, 60), (1000, 20), (10000, 10)]:
        lat, wall, b = serve(n, rounds, meta_Q=meta_Q)
        ms = lat * 1000.0
        dec = float(ms.mean()); recog = rec[min(rec, key=lambda k: abs(k - n))]
        per_user_ms = dec + recog
        res["single_process"][str(n)] = {
            "rounds": rounds, "decisions": int(len(ms)),
            "decision_ms_mean": dec, "decision_ms_median": float(np.median(ms)),
            "decision_ms_p95": float(np.percentile(ms, 95)), "decision_ms_p99": float(np.percentile(ms, 99)),
            "decision_ms_max": float(ms.max()),
            "recognition_ms_per_window": recog,
            "round_seconds_decisions_only": float(ms.sum() / 1000.0 / rounds),
            "round_seconds_end_to_end_estimate": float(n * per_user_ms / 1000.0),
            "memory_kb_per_user": b / 1024.0,
            "users_in_real_time_per_process": int(WINDOW_S * 1000.0 / per_user_ms)}
        print(n, json.dumps(res["single_process"][str(n)]), flush=True)
    from multiprocessing import Pool
    total, rounds = 1000, 20
    for P in [1, 2, 4, 8, 12]:
        with Pool(P) as pool:
            t0 = time.perf_counter()
            parts = pool.map(_worker, [(total // P, rounds, 7 + k) for k in range(P)])
            wall = time.perf_counter() - t0
        n_dec = sum(p[0] for p in parts)
        res["multi_process"][str(P)] = {"users": total, "rounds": rounds, "decisions": n_dec,
                                        "wall_seconds_incl_start": wall,
                                        "decisions_per_second": n_dec / max(max(p[1] for p in parts), 1e-9)}
        print("P", P, res["multi_process"][str(P)], flush=True)
    json.dump(res, open(os.path.join(OUT, "e3_latency_load.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
