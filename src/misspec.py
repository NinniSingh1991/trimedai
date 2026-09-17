"""Misspecification of the model class, and noise on the deterioration signal.

The robustness experiment of the first submission varied the *value* of the
latent: users were drawn from profiles the agent had never been given, but the
functional form of the environment was the one the agent assumes. A reviewer
observed, correctly, that this never makes the model class wrong, so the
evaluation was close to self-referential. This script closes that gap in two
ways.

Form-level misspecification. Three properties of the simulator that the agent's
model of equations (14), (17) and (21) asserts are replaced by properties it
cannot express, one at a time and then together:

    product          p_theta(x) = phi_theta * u(x)          (what the agent assumes)
    proportional     p_theta(x) = 1 - (1 - phi_theta)^(c u(x))
    boltzmann        stated preference ~ softmax(beta * channel value)
    epsilon-greedy   stated preference = best channel with prob 1 - eps, else uniform
    constant relief  an attending caregiver always resolves three severity bands
    graded relief    the caregiver resolves 3, 2 or 1 bands with probability .5/.3/.2

The proportional-hazard form is calibrated so that the population-mean
probability of deterioration matches the product form, so what changes is the
shape of the dependence and not the overall difficulty of the task.

Observation noise. The deterioration flag w of equation (22) is what makes
frailty identifiable and therefore what sets the escalation threshold, and the
first submission assumed it arrived without error. Three one-at-a-time sweeps
relax that: the flag is missed with probability fnr, raised spuriously with
probability fpr, and the context urgency u(x) the agent plans with is perturbed
away from the true one.

In every configuration the oracle, the floor policy and the regret reference are
recomputed on the *true* dynamics, so the ceiling is the right ceiling and the
agent is the only thing that is wrong.

    python misspec.py            # 10 seeds, 40 users
    N_SEEDS=3 N_USERS=20 python misspec.py
"""
import json, os, sys, time
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA, RESULTS

import triage_env as TE
from triage_env import (REQUESTS, ACTIONS, PROFILES, PROFILE_NAMES, CORE_NAMES,
                        N_CORE, N_SEV, N_STATES, ALERT, ASK, CRISIS, GAMMA,
                        W_COMFORT, ESCALATION_COST, ASK_COST, CRISIS_PENALTY,
                        RECOVER, NEWS2_URGENT_REVIEW, AFFINITY, URGENCY,
                        sid, latent_of, comfort, model_for, plan, plan_average,
                        news2_policy)
from triage_agents import ParticleBelief, PlanningPolicy
import perception as PC

N_ACT, N_REQ = len(ACTIONS), len(REQUESTS)
SEEDS = list(range(int(os.environ.get("N_SEEDS", "10"))))
N_USERS = int(os.environ.get("N_USERS", "40"))
T_STEPS, EVAL_FROM = 200, 100
_URG_TRUE = np.array([URGENCY[r] for r in range(N_REQ)])
_AFF_TRUE = np.array([[AFFINITY[r].get(a, 0.0) for a in ACTIONS]
                      for r in range(N_REQ)])


# ===========================================================  true dynamics ==
def _calibrate_proportional():
    """choose c so that the mean deterioration probability over the population
    and the contexts equals the one the product form produces"""
    fr = np.array([PROFILES[p]["frailty"] for p in CORE_NAMES])
    target = float(np.mean(fr[:, None] * _URG_TRUE[None, :]))

    def mean_for(c):
        return float(np.mean(1.0 - (1.0 - fr[:, None]) ** (c * _URG_TRUE[None, :])))

    lo, hi = 1e-4, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_for(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


C_PROP = _calibrate_proportional()

RELIEF_GRADED = ((0.5, 3), (0.3, 2), (0.2, 1))


class World:
    """the true environment, which need not belong to the agent's model class"""

    def __init__(self, name, deterioration="product", preference="boltzmann",
                 relief="constant", eps=0.30, beta=4.0):
        self.name = name
        self.deterioration, self.preference, self.relief = (
            deterioration, preference, relief)
        self.eps, self.beta = eps, beta

    # ---- deterioration -----------------------------------------------------
    def p_worse(self, frailty, request):
        u = _URG_TRUE[request]
        if self.deterioration == "product":
            return float(np.clip(frailty * u, 0.0, 1.0))
        if self.deterioration == "proportional":
            return float(np.clip(1.0 - (1.0 - frailty) ** (C_PROP * u), 0.0, 1.0))
        raise ValueError(self.deterioration)

    # ---- what an attending caregiver resolves ------------------------------
    def relief_branches(self, sev):
        if self.relief == "constant":
            return ((1.0, max(0, sev - 3)),)
        return tuple((p, max(0, sev - d)) for p, d in RELIEF_GRADED)

    # ---- how the person states the channel they would have preferred -------
    def state_preference(self, profile, request, rng):
        c = np.array([comfort(profile, request, a) for a in range(N_ACT)])
        c[ASK] = -1e9
        if self.preference == "boltzmann":
            v = self.beta * c
            v = np.exp(v - v.max())
            v /= v.sum()
            return int(rng.choice(N_ACT, p=v))
        # epsilon-greedy: no temperature, no graded trade-off between channels
        if rng.random() < self.eps:
            cand = [a for a in range(N_ACT) if a != ASK]
            return int(cand[int(rng.integers(len(cand)))])
        return int(np.argmax(c))

    # ---- one interaction, as a distribution --------------------------------
    def outcomes(self, profile, request, severity, action_idx):
        pw = self.p_worse(PROFILES[profile]["frailty"], request)
        demanding = _URG_TRUE[request] >= 0.25
        cf = comfort(profile, request, action_idx)
        if action_idx == ALERT:
            base = W_COMFORT * cf - ESCALATION_COST
        elif action_idx == ASK:
            base = -ASK_COST
        else:
            base = W_COMFORT * cf
        out = []
        for p, worsened in ((pw, True), (1.0 - pw, False)):
            if p <= 0.0:
                continue
            sev = severity + 1 if worsened else severity
            if sev >= CRISIS:
                out.append((p, base - CRISIS_PENALTY, 0, True, worsened))
                continue
            if action_idx == ALERT:
                for q, nxt in self.relief_branches(sev):
                    out.append((p * q, base, nxt, False, worsened))
            elif action_idx == ASK:
                out.append((p, base, sev, False, worsened))
            else:
                if cf > 0.5 and sev > 0 and not demanding:
                    out.append((p * RECOVER, base, sev - 1, False, worsened))
                    out.append((p * (1.0 - RECOVER), base, sev, False, worsened))
                else:
                    out.append((p, base, sev, False, worsened))
        return out

    # ---- sampling tables and the exact solution ----------------------------
    def build(self):
        self.table, self.R, self.T = {}, {}, {}
        for pf in PROFILE_NAMES:
            R = np.zeros((N_STATES, N_ACT))
            T = np.zeros((N_STATES, N_ACT, N_SEV))
            for rq in range(N_REQ):
                for sv in range(N_SEV):
                    s = sid(rq, sv)
                    for a in range(N_ACT):
                        o = self.outcomes(pf, rq, sv, a)
                        cum = np.cumsum([x[0] for x in o])
                        cum = cum / cum[-1]
                        self.table[(pf, rq, sv, a)] = (
                            cum,
                            np.array([x[1] for x in o]),
                            np.array([x[2] for x in o], dtype=np.int64),
                            np.array([x[3] for x in o], dtype=bool),
                            np.array([x[4] for x in o], dtype=bool))
                        R[s, a] = sum(x[0] * x[1] for x in o)
                        for x in o:
                            T[s, a, x[2]] += x[0]
            self.R[pf], self.T[pf] = R, T
        self.Q, self.pi, self.gain = {}, {}, {}
        for pf in PROFILE_NAMES:
            Q, g = plan_average(self.R[pf], self.T[pf])
            self.Q[pf], self.pi[pf], self.gain[pf] = Q, Q.argmax(axis=1), g
        return self

    def step(self, profile, request, severity, action_idx, rng):
        cum, rew, nxt, cri, wor = self.table[(profile, request, severity, action_idx)]
        i = int(np.searchsorted(cum, rng.random()))
        i = min(i, len(rew) - 1)
        return float(rew[i]), int(nxt[i]), bool(cri[i]), bool(wor[i])


WORLDS = [
    World("product / Boltzmann / constant relief (the agent's model class)"),
    World("proportional-hazard deterioration", deterioration="proportional"),
    World("epsilon-greedy stated preference", preference="epsgreedy"),
    World("graded caregiver relief", relief="graded"),
    World("all three together", deterioration="proportional",
          preference="epsgreedy", relief="graded"),
]


# ==================================================================  agents ==
def population_rule_policy():
    """the fixed immediate-value rule, averaged over the four named latents,
    computed in the agent's own model class"""
    R = np.zeros((N_STATES, N_ACT))
    for pf in CORE_NAMES:
        fr, pref = latent_of(pf)
        Rp, _ = model_for(fr, pref)
        R += Rp / N_CORE
    return R.argmax(axis=1)


def qmdp_tables():
    """QMDP is given the true latent of every named profile but, like the
    framework, the agent's functional form for the dynamics"""
    out = []
    for pf in CORE_NAMES:
        fr, pref = latent_of(pf)
        R, T = model_for(fr, pref)
        out.append(plan(R, T, sweeps=3000))
    return np.stack(out)


def profile_posterior(belief):
    fr = belief.weights @ belief.frailty
    pref = belief.weights @ belief.pref
    d = np.array([np.hypot(fr - latent_of(p)[0],
                           np.mean(np.abs(pref - latent_of(p)[1])))
                  for p in CORE_NAMES])
    s = np.exp(-8.0 * d)
    return s / s.sum()


# ===============================================================  one config ==
def run_config(world, seed, pred, yte, meta_Q, QSTAR, pop_pi, fixed_pi,
               fnr=0.0, fpr=0.0, urg_sd=0.0):
    """One seed of one configuration. `pred` is the cached CNN prediction over
    the test windows, so a recognition error propagates as it does elsewhere."""
    rng = np.random.default_rng(1000 + seed)
    by_class = {c: np.where(yte == c)[0] for c in range(N_REQ)}
    users = [CORE_NAMES[int(rng.integers(N_CORE))] for _ in range(N_USERS)]
    streams = []
    for _ in users:
        cls = rng.integers(0, N_REQ, size=T_STEPS)
        idx = np.array([rng.choice(by_class[int(c)]) for c in cls])
        streams.append((cls, idx))

    acc = {m: dict(rew=[], reg=[], crisis=0, alerts=0, n=0, fr=[],
                   tp=0, fn=0, tn=0, fp=0)
           for m in ("Oracle", "Random", "Fixed threshold", "QMDP", "TriMedAI")}

    for u, (cls, idx) in zip(users, streams):
        oq, opi = world.Q[u], world.pi[u]
        for method in acc:
            st = acc[method]
            belief = pol = None
            if method in ("TriMedAI", "QMDP"):
                belief = ParticleBelief(rng=np.random.default_rng(seed * 977 + 13))
            if method == "TriMedAI":
                pol = PlanningPolicy(init=meta_Q.copy())
            sev = 0
            for t in range(T_STEPS):
                rq = int(cls[t])
                rp = int(pred[idx[t]])
                s_p, s_t = sid(rp, sev), sid(rq, sev)
                if method == "Oracle":
                    a = int(opi[s_t])
                elif method == "Random":
                    a = int(rng.integers(N_ACT))
                elif method == "Fixed threshold":
                    a = int(fixed_pi[s_p])
                elif method == "QMDP":
                    a = int(np.argmax(profile_posterior(belief) @ QSTAR[:, s_p, :]))
                else:
                    a = pol.act(s_p)

                r, nsev, crisis, worsened = world.step(u, rq, sev, a, rng)

                if belief is not None:
                    belief.observe_preference(
                        rp, world.state_preference(u, rq, rng))
                    w_obs = worsened
                    if worsened and rng.random() < fnr:
                        w_obs = False
                    elif (not worsened) and rng.random() < fpr:
                        w_obs = True
                    belief.observe_transition(rp, w_obs)
                    if pol is not None:
                        pol.observe(belief)

                if t >= EVAL_FROM:
                    st["rew"].append(r)
                    st["reg"].append(float(np.max(oq[s_t]) - oq[s_t, a]))
                    st["crisis"] += int(crisis)
                    st["alerts"] += int(a == ALERT)
                    st["n"] += 1
                    want = int(opi[s_t])
                    if want == ALERT:
                        st["tp" if a == ALERT else "fn"] += 1
                    else:
                        st["fp" if a == ALERT else "tn"] += 1
                sev = nsev
            if belief is not None:
                st["fr"].append(belief.latent_error(u)[0])

    out = {}
    for m, st in acc.items():
        n = max(st["n"], 1)
        out[m] = dict(reward=float(np.mean(st["rew"])),
                      regret=100.0 * float(np.mean(st["reg"])),
                      crisis=100.0 * st["crisis"] / n,
                      alerts=100.0 * st["alerts"] / n,
                      sens=100.0 * st["tp"] / max(st["tp"] + st["fn"], 1),
                      spec=100.0 * st["tn"] / max(st["tn"] + st["fp"], 1),
                      frailty_error=float(np.mean(st["fr"])) if st["fr"] else None)
    return out


def normalise(rows):
    """floor-anchored score, paired by seed"""
    o = np.array([r["Oracle"]["reward"] for r in rows])
    f = np.array([r["Random"]["reward"] for r in rows])
    span = o - f
    out = {}
    for m in rows[0]:
        x = np.array([r[m]["reward"] for r in rows])
        z = 100.0 * (x - f) / span
        out[m] = dict(norm=float(np.mean(z)), norm_sd=float(np.std(z, ddof=1)),
                      per_seed=[float(v) for v in z])
        for k in ("reward", "regret", "crisis", "alerts", "sens", "spec"):
            vals = [r[m][k] for r in rows]
            out[m][k] = float(np.mean(vals))
            out[m][k + "_sd"] = float(np.std(vals, ddof=1))
        fe = [r[m]["frailty_error"] for r in rows if r[m]["frailty_error"] is not None]
        out[m]["frailty_error"] = float(np.mean(fe)) if fe else None
    return out


def main():
    from triage_agents import meta_initialisation
    Xtr, ytr, Xte, yte = PC.load(DATA)
    cnn = PC.train("CNN", Xtr, ytr, Xte, yte, seed=0)
    pred = cnn["pred"]
    print("context recognition %.2f%% on held-out subjects" % (100 * cnn["accuracy"]),
          flush=True)
    print("proportional-hazard exponent c = %.4f (calibrated to the product form's "
          "population mean)" % C_PROP, flush=True)

    QSTAR = qmdp_tables()
    pop_pi = population_rule_policy()
    fixed_pi = news2_policy()
    meta = {s: meta_initialisation(seed=s) for s in SEEDS}

    blob = {"seeds": SEEDS, "n_users": N_USERS, "steps_per_user": T_STEPS,
            "evaluation_from_step": EVAL_FROM,
            "proportional_hazard_c": C_PROP,
            "form": {}, "noise": {}}

    t0 = time.perf_counter()
    print("\n--- form-level misspecification "
          "(the agent's model class is held fixed) ---", flush=True)
    print("%-52s %8s %8s %8s %8s %8s"
          % ("true environment", "method", "norm", "regret", "crisis", "callout"))
    for w in WORLDS:
        w.build()
        rows = [run_config(w, s, pred, yte, meta[s], QSTAR, pop_pi, fixed_pi)
                for s in SEEDS]
        res = normalise(rows)
        blob["form"][w.name] = res
        for m in ("Oracle", "Fixed threshold", "QMDP", "TriMedAI"):
            print("%-52s %8s %8.1f %8.3f %8.2f %8.2f"
                  % (w.name if m == "Oracle" else "", m, res[m]["norm"],
                     res[m]["regret"], res[m]["crisis"], res[m]["alerts"]),
                  flush=True)
        print("", flush=True)

    base = WORLDS[0]
    print("--- noise on the deterioration signal, and on the assumed urgency ---",
          flush=True)
    print("%-34s %8s %8s %8s %8s %10s"
          % ("configuration", "method", "norm", "regret", "crisis", "frailty err"))
    sweeps = ([("missed deterioration reports (fnr)", dict(fnr=v))
               for v in (0.0, 0.1, 0.2, 0.3, 0.5)] +
              [("spurious deterioration reports (fpr)", dict(fpr=v))
               for v in (0.05, 0.1, 0.2, 0.3)] +
              [("both, symmetric", dict(fnr=v, fpr=v)) for v in (0.1, 0.2, 0.3)] +
              [("error in the assumed urgency u(x)", dict(urg_sd=v))
               for v in (0.1, 0.2, 0.3, 0.5)])
    for label, kw in sweeps:
        key = "%s | %s" % (label, ", ".join("%s=%.2f" % (k, v)
                                            for k, v in kw.items()))
        urg_sd = kw.pop("urg_sd", 0.0)
        rows = []
        for s in SEEDS:
            if urg_sd > 0.0:
                g = np.random.default_rng(5000 + s)
                pert = np.clip(_URG_TRUE * np.exp(g.normal(0, urg_sd, N_REQ)),
                               0.0, 0.98)
                TE._URG[:] = pert                      # the agent now plans wrong
            else:
                TE._URG[:] = _URG_TRUE
            rows.append(run_config(base, s, pred, yte, meta[s], QSTAR, pop_pi,
                                   fixed_pi, **kw))
        TE._URG[:] = _URG_TRUE
        res = normalise(rows)
        blob["noise"][key] = res
        for m in ("TriMedAI", "QMDP"):
            print("%-34s %8s %8.1f %8.3f %8.2f %10s"
                  % (key[:34] if m == "TriMedAI" else "", m, res[m]["norm"],
                     res[m]["regret"], res[m]["crisis"],
                     "%.3f" % res[m]["frailty_error"]
                     if res[m]["frailty_error"] is not None else "n/a"),
                  flush=True)

    json.dump(blob, open(os.path.join(RESULTS, "har_misspec.json"), "w"), indent=1)
    print("\nwritten to results/har_misspec.json, %.0f s"
          % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
