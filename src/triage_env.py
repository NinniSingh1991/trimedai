"""Sequential assistive-care environment over real wearable activity contexts.

The context a person is in is recognised from the UCI Human Activity Recognition
corpus: thirty volunteers wearing a waist-mounted smartphone, inertial signals at
50 Hz segmented into 2.56 second windows, six activities of daily living. The
train and test partitions are split by subject, so the perception layer is always
evaluated on people it has never seen.

Above that sits the care process, which is simulated. A person carries a severity
state that persists between interactions and worsens when a developing problem is
not addressed. The scale is aligned with the National Early Warning Score 2, so
the levels and the escalation thresholds follow published clinical guidance rather
than being chosen by the authors:

    severity 0   NEWS2 0          routine monitoring
    severity 1   NEWS2 1 to 4     low risk, increased observation frequency
    severity 2   NEWS2 5 to 6     medium risk, urgent review by a clinician
    severity 3   NEWS2 7 or more  high risk, emergency assessment
    severity 4   deterioration    unplanned critical care, resuscitation call

NEWS2 directs an urgent review at the medium band, which is severity 2 here. That
threshold is used to define the clinical-protocol comparator, so the framework is
measured against the standard of care and not only against machine-learning
baselines.

Three properties make each stage of the framework necessary. Summoning a
caregiver costs more on the interaction where it is issued than it returns, and
repays only through the severity the person is left in. A deterioration event
carries a large penalty that arrives several interactions after the decisions
that permitted it. And people differ in how rapidly they deteriorate as well as
in the channel they can receive, so the severity at which escalation becomes
correct is a property of the individual.
"""
import numpy as np

# --- activity contexts recognised from the wearable signal --------------------
# order matches the UCI HAR label order
REQUESTS = ["Walking", "Ascending stairs", "Descending stairs",
            "Sitting", "Standing", "Lying down"]

# --- assistance actions ------------------------------------------------------
ACTIONS = ["voice prompt", "visual display", "haptic cue", "mobility assistance",
           "ask clarifying question", "caregiver alert"]
ASK = ACTIONS.index("ask clarifying question")
ALERT = ACTIONS.index("caregiver alert")

# --- severity, aligned with the NEWS2 bands described above ------------------
N_SEV = 5
CRISIS = 4
NEWS2_BANDS = ["NEWS2 0 (routine)", "NEWS2 1-4 (low)", "NEWS2 5-6 (medium)",
               "NEWS2 >=7 (high)", "deterioration event"]
NEWS2_URGENT_REVIEW = 2      # the band at which NEWS2 directs an urgent review

# --- how strongly each context tends to deteriorate if unsupported -----------
# stair descent carries the highest fall risk, ascent the highest exertion, and
# recumbent rest the lowest; sitting is treated as the quiescent reference
URGENCY = {0: 0.30,    # walking
           1: 0.70,    # ascending stairs
           2: 0.80,    # descending stairs
           3: 0.05,    # sitting
           4: 0.35,    # standing
           5: 0.00}    # lying down

# a well-supported quiescent context gives some relief, but no delivery channel
# resolves a developing problem; only an attending caregiver does
RECOVER = 0.10

# --- which assistance channel suits which context, independent of the person --
# a person who is walking or on stairs cannot read a screen, so visual output is
# poor there and haptic or spoken output is preferred
AFFINITY = {
    0: {"voice prompt": 0.90, "visual display": 0.20, "haptic cue": 0.85,
        "mobility assistance": 1.00, "caregiver alert": 0.30},
    1: {"voice prompt": 0.70, "visual display": 0.10, "haptic cue": 0.95,
        "mobility assistance": 0.80, "caregiver alert": 1.00},
    2: {"voice prompt": 0.65, "visual display": 0.10, "haptic cue": 1.00,
        "mobility assistance": 0.75, "caregiver alert": 1.00},
    3: {"voice prompt": 0.95, "visual display": 1.00, "haptic cue": 0.40,
        "mobility assistance": 0.10, "caregiver alert": 0.20},
    4: {"voice prompt": 1.00, "visual display": 0.70, "haptic cue": 0.60,
        "mobility assistance": 0.50, "caregiver alert": 0.40},
    5: {"voice prompt": 1.00, "visual display": 0.55, "haptic cue": 0.35,
        "mobility assistance": 0.05, "caregiver alert": 0.25},
}

# --- user profiles: a channel preference and a frailty -----------------------
PROFILES = {
    "visually impaired": dict(
        pref={"voice prompt": 1.0, "visual display": 0.0, "haptic cue": 0.7,
              "mobility assistance": 0.5, "caregiver alert": 0.5},
        frailty=0.45),
    "hearing impaired": dict(
        pref={"voice prompt": 0.0, "visual display": 1.0, "haptic cue": 0.8,
              "mobility assistance": 0.5, "caregiver alert": 0.5},
        frailty=0.45),
    "mobility impaired": dict(
        pref={"voice prompt": 0.6, "visual display": 0.6, "haptic cue": 0.4,
              "mobility assistance": 1.0, "caregiver alert": 0.6},
        frailty=0.80),
    "cognitively impaired": dict(
        pref={"voice prompt": 0.8, "visual display": 0.5, "haptic cue": 0.5,
              "mobility assistance": 0.3, "caregiver alert": 1.0},
        frailty=0.95),
}
N_CORE = len(PROFILES)

# Three further profiles the environment can generate but the agent is never
# told about, used by the robustness experiment.
HELD_OUT = {
    "mixed sensory (held out)": dict(
        pref={"voice prompt": 0.55, "visual display": 0.55, "haptic cue": 0.65,
              "mobility assistance": 0.45, "caregiver alert": 0.50},
        frailty=0.62),
    "severely frail (held out)": dict(
        pref={"voice prompt": 0.70, "visual display": 0.40, "haptic cue": 0.55,
              "mobility assistance": 0.55, "caregiver alert": 0.90},
        frailty=0.99),
    "haptic dominant (held out)": dict(
        pref={"voice prompt": 0.20, "visual display": 0.20, "haptic cue": 1.00,
              "mobility assistance": 0.35, "caregiver alert": 0.40},
        frailty=0.25),
}
PROFILES.update(HELD_OUT)
PROFILE_NAMES = list(PROFILES)
CORE_NAMES = PROFILE_NAMES[:N_CORE]
HELD_OUT_NAMES = PROFILE_NAMES[N_CORE:]

# --- reward weights ----------------------------------------------------------
W_COMFORT = 0.35        # value of supporting the person through a suitable channel
ESCALATION_COST = 0.60  # caregiver time is scarce, so a call-out is not free
ASK_COST = 0.12         # asking costs the person effort and delays the response
CRISIS_PENALTY = 2.50   # a deterioration event
GAMMA = 0.90

N_STATES = len(REQUESTS) * N_SEV


def sid(request, severity):
    return request * N_SEV + severity


def comfort(profile, request, action_idx):
    """immediate value of the chosen channel to this individual"""
    a = ACTIONS[action_idx]
    if a not in AFFINITY[request]:
        return 0.0
    fit = PROFILES[profile]["pref"][a]
    return AFFINITY[request][a] * (0.35 + 0.65 * fit)


def outcomes(profile, request, severity, action_idx):
    """The full transition for one interaction, as a list of

        (probability, immediate reward, next severity, crisis, worsened)

    Sampling and the exact solver both read this function, so the simulator and
    the optimal-policy oracle cannot drift apart.

    The condition develops before the chosen action takes effect. A summoned
    caregiver has to travel and an assistant that is still speaking has not yet
    helped, so an alert raised at high severity does not avert a deterioration
    already under way; the value of escalating lies in the severity the person is
    left in, which is a future quantity.
    """
    p_worse = PROFILES[profile]["frailty"] * URGENCY[request]
    demanding = URGENCY[request] >= 0.25

    if action_idx == ALERT:
        base = W_COMFORT * comfort(profile, request, action_idx) - ESCALATION_COST
    elif action_idx == ASK:
        base = -ASK_COST
    else:
        base = W_COMFORT * comfort(profile, request, action_idx)

    final = []
    for p, worsened in ((p_worse, True), (1.0 - p_worse, False)):
        if p <= 0.0:
            continue
        sev = severity + 1 if worsened else severity
        if sev >= CRISIS:
            final.append((p, base - CRISIS_PENALTY, 0, True, worsened))
            continue
        if action_idx == ALERT:
            final.append((p, base, max(0, sev - 3), False, worsened))
        elif action_idx == ASK:
            final.append((p, base, sev, False, worsened))
        else:
            fit = comfort(profile, request, action_idx)
            if fit > 0.5 and sev > 0 and not demanding:
                final.append((p * RECOVER, base, sev - 1, False, worsened))
                final.append((p * (1.0 - RECOVER), base, sev, False, worsened))
            else:
                final.append((p, base, sev, False, worsened))
    return final


_TABLE = {}


def _tabulate():
    for pf in PROFILE_NAMES:
        for req in range(len(REQUESTS)):
            for sev in range(N_SEV):
                for a in range(len(ACTIONS)):
                    outs = outcomes(pf, req, sev, a)
                    cum = np.cumsum([o[0] for o in outs])
                    cum = cum / cum[-1]
                    _TABLE[(pf, req, sev, a)] = (
                        cum,
                        np.array([o[1] for o in outs]),
                        np.array([o[2] for o in outs], dtype=np.int64),
                        np.array([o[3] for o in outs], dtype=bool),
                        np.array([o[4] for o in outs], dtype=bool))


_tabulate()

EXP_R = np.zeros((len(PROFILE_NAMES), N_STATES, len(ACTIONS)))
TRANS = np.zeros((len(PROFILE_NAMES), N_STATES, len(ACTIONS), N_SEV))
P_WORSE = np.zeros((len(PROFILE_NAMES), len(REQUESTS)))
COMFORT = np.zeros((len(PROFILE_NAMES), len(REQUESTS), len(ACTIONS)))


def _derive():
    for pi, pf in enumerate(PROFILE_NAMES):
        for rq in range(len(REQUESTS)):
            for sv in range(N_SEV):
                s = sid(rq, sv)
                for a in range(len(ACTIONS)):
                    outs = outcomes(pf, rq, sv, a)
                    EXP_R[pi, s, a] = sum(p * r for p, r, _, _, _ in outs)
                    TRANS[pi, s, a, :] = 0.0
                    for p, r, nxt, _, _ in outs:
                        TRANS[pi, s, a, nxt] += p
            P_WORSE[pi, rq] = PROFILES[pf]["frailty"] * URGENCY[rq]
            for a in range(len(ACTIONS)):
                COMFORT[pi, rq, a] = comfort(pf, rq, a)


_derive()


# ---------------------------------------------------------------------------
# The environment above is written in terms of named profiles because that is
# how the population is generated. The agent, however, does not get to assume
# that people come in a small number of named kinds. The functions below express
# the same dynamics for an arbitrary continuous latent, a frailty in [0,1] and a
# preference weight per channel, so a policy can be derived for any point in the
# latent space rather than only for the four the environment happens to use.
# ---------------------------------------------------------------------------
CHANNELS = [a for a in ACTIONS if a not in (ACTIONS[ASK], ACTIONS[ALERT])] + \
           [ACTIONS[ALERT]]
_AFF = np.array([[AFFINITY[r].get(a, 0.0) for a in ACTIONS]
                 for r in range(len(REQUESTS))])          # contexts x actions
_URG = np.array([URGENCY[r] for r in range(len(REQUESTS))])


def latent_of(profile):
    """the continuous latent corresponding to one named profile"""
    pref = np.array([PROFILES[profile]["pref"].get(a, 0.0) for a in ACTIONS])
    return float(PROFILES[profile]["frailty"]), pref


def comfort_matrix(pref):
    """value of each channel in each context for a latent with this preference"""
    return _AFF * (0.35 + 0.65 * pref[None, :])


def model_for(frailty, pref):
    """Expected reward and severity transition for an arbitrary continuous latent.

    Returns R with shape (states, actions) and T with shape
    (states, actions, next severity). Vectorised, because the planner rebuilds
    this every time its belief moves.
    """
    nR, nA = len(REQUESTS), len(ACTIONS)
    C = comfort_matrix(pref)
    p_worse = np.clip(frailty * _URG, 0.0, 1.0)
    demanding = _URG >= 0.25

    R = np.zeros((N_STATES, nA))
    T = np.zeros((N_STATES, nA, N_SEV))
    for rq in range(nR):
        base = W_COMFORT * C[rq].copy()
        base[ASK] = -ASK_COST
        base[ALERT] = W_COMFORT * C[rq, ALERT] - ESCALATION_COST
        pw = p_worse[rq]
        for sv in range(N_SEV):
            s = sid(rq, sv)
            for branch_p, worsened in ((pw, True), (1.0 - pw, False)):
                if branch_p <= 0.0:
                    continue
                sev = sv + 1 if worsened else sv
                if sev >= CRISIS:
                    R[s] += branch_p * (base - CRISIS_PENALTY)
                    T[s, :, 0] += branch_p
                    continue
                R[s] += branch_p * base
                T[s, ALERT, max(0, sev - 3)] += branch_p
                T[s, ASK, sev] += branch_p
                for a in range(nA):
                    if a in (ASK, ALERT):
                        continue
                    if C[rq, a] > 0.5 and sev > 0 and not demanding[rq]:
                        T[s, a, sev - 1] += branch_p * RECOVER
                        T[s, a, sev] += branch_p * (1.0 - RECOVER)
                    else:
                        T[s, a, sev] += branch_p
    return R, T


def immediate_reward(frailty, pref, request, severity):
    """Expected immediate reward of every action at one state, for a continuous
    latent. Cheap enough to call every interaction, which the myopic variants do.

    Only the crisis term depends on severity: a deterioration matters now only
    when the person is one level below the top of the scale.
    """
    C = _AFF[request] * (0.35 + 0.65 * pref)
    base = W_COMFORT * C
    base[ASK] = -ASK_COST
    base[ALERT] = W_COMFORT * C[ALERT] - ESCALATION_COST
    if severity + 1 >= CRISIS:
        base = base - CRISIS_PENALTY * float(np.clip(frailty * _URG[request], 0, 1))
    return base


def population_latent(names=None):
    """the average latent of the population, which is what an assistant that does
    not personalise is implicitly assuming about everybody"""
    names = names or CORE_NAMES
    fr = float(np.mean([PROFILES[p]["frailty"] for p in names]))
    pref = np.mean([np.array([PROFILES[p]["pref"].get(a, 0.0) for a in ACTIONS])
                    for p in names], axis=0)
    return fr, pref


def plan(R, T, gamma=None, sweeps=60, Q0=None):
    """Value iteration on a given model, vectorised and warm-startable.

    The next context is drawn independently of the action taken, so the value of
    arriving at a severity is the average over the contexts that may accompany
    it. That collapses the backup to a matrix product and makes re-solving cheap
    enough to do whenever the belief moves.
    """
    if gamma is None:
        gamma = GAMMA
    nR, nA = len(REQUESTS), len(ACTIONS)
    Q = np.zeros((N_STATES, nA)) if Q0 is None else Q0.copy()
    for _ in range(sweeps):
        Vbar = Q.reshape(nR, N_SEV, nA).max(axis=2).mean(axis=0)   # (N_SEV,)
        Qn = R + gamma * (T @ Vbar)
        if np.max(np.abs(Qn - Q)) < 1e-9:
            Q = Qn
            break
        Q = Qn
    return Q


def step(profile, request, severity, action_idx, rng):
    """sample one interaction: (reward, next severity, crisis, worsened)"""
    cum, rew, nxt, cri, wor = _TABLE[(profile, request, severity, action_idx)]
    i = int(np.searchsorted(cum, rng.random()))
    if i >= len(rew):
        i = len(rew) - 1
    return float(rew[i]), int(nxt[i]), bool(cri[i]), bool(wor[i])


def reconfigure(**kw):
    """change the environment constants and rebuild every derived table in place"""
    g = globals()
    for name in ("W_COMFORT", "ESCALATION_COST", "ASK_COST", "CRISIS_PENALTY",
                 "RECOVER", "GAMMA"):
        if name in kw:
            g[name] = float(kw[name])
    if "urgency_scale" in kw:
        s = float(kw["urgency_scale"])
        for k in URGENCY:
            URGENCY[k] = min(0.98, _URGENCY_BASE[k] * s)
    if "frailty_spread" in kw:
        s = float(kw["frailty_spread"])
        mean = float(np.mean([_FRAILTY_BASE[p] for p in PROFILE_NAMES]))
        for p in PROFILE_NAMES:
            PROFILES[p]["frailty"] = float(
                np.clip(mean + s * (_FRAILTY_BASE[p] - mean), 0.02, 0.99))
    _TABLE.clear()
    _tabulate()
    _derive()


_URGENCY_BASE = dict(URGENCY)
_FRAILTY_BASE = {p: PROFILES[p]["frailty"] for p in PROFILE_NAMES}
_DEFAULTS = dict(W_COMFORT=W_COMFORT, ESCALATION_COST=ESCALATION_COST,
                 ASK_COST=ASK_COST, CRISIS_PENALTY=CRISIS_PENALTY,
                 RECOVER=RECOVER, GAMMA=GAMMA, urgency_scale=1.0,
                 frailty_spread=1.0)


def reset_config():
    reconfigure(**_DEFAULTS)


def solve(profile, gamma=None, iters=4000, tol=1e-10):
    """Exact optimal policy for a fully known user, by value iteration.

    Not available to any agent. It is the ceiling against which the learned
    policies are measured, so the comparison is against what the task permits
    rather than only against the other methods.
    """
    if gamma is None:
        gamma = GAMMA
    nR, nA = len(REQUESTS), len(ACTIONS)
    V = np.zeros(N_STATES)
    R = np.zeros((N_STATES, nA))
    T = [[[] for _ in range(nA)] for _ in range(N_STATES)]
    for req in range(nR):
        for sev in range(N_SEV):
            s = sid(req, sev)
            for a in range(nA):
                tot = 0.0
                for p, r, nxt, _, _ in outcomes(profile, req, sev, a):
                    tot += p * r
                    T[s][a].append((p, nxt))
                R[s, a] = tot

    for _ in range(iters):
        Vn = np.empty_like(V)
        for s in range(N_STATES):
            best = -1e18
            for a in range(nA):
                nxt_v = 0.0
                for p, nsev in T[s][a]:
                    nxt_v += p * np.mean([V[sid(rq, nsev)] for rq in range(nR)])
                q = R[s, a] + gamma * nxt_v
                if q > best:
                    best = q
            Vn[s] = best
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn

    pi = np.zeros(N_STATES, dtype=int)
    Q = np.zeros((N_STATES, nA))
    for s in range(N_STATES):
        for a in range(nA):
            nxt_v = 0.0
            for p, nsev in T[s][a]:
                nxt_v += p * np.mean([V[sid(rq, nsev)] for rq in range(nR)])
            Q[s, a] = R[s, a] + gamma * nxt_v
        pi[s] = int(np.argmax(Q[s]))
    return pi, Q, V


def myopic_policy(profile):
    """the policy that maximises immediate reward only, ignoring what follows"""
    pi = np.zeros(N_STATES, dtype=int)
    for req in range(len(REQUESTS)):
        for sev in range(N_SEV):
            vals = [sum(p * r for p, r, _, _, _ in outcomes(profile, req, sev, a))
                    for a in range(len(ACTIONS))]
            pi[sid(req, sev)] = int(np.argmax(vals))
    return pi


def news2_policy(profile=None):
    """The standard of care: escalate once the person reaches the NEWS2 medium
    band, at which the score directs an urgent clinical review. Identical for
    every person, because a published score does not know who it is scoring."""
    pi = np.zeros(N_STATES, dtype=int)
    for req in range(len(REQUESTS)):
        for sev in range(N_SEV):
            if sev >= NEWS2_URGENT_REVIEW:
                pi[sid(req, sev)] = ALERT
            else:
                vals = [AFFINITY[req].get(ACTIONS[a], 0.0)
                        if a not in (ASK, ALERT) else -9.0
                        for a in range(len(ACTIONS))]
                pi[sid(req, sev)] = int(np.argmax(vals))
    return pi


def _threshold(pi, request):
    got = [sev for sev in range(CRISIS) if pi[sid(request, sev)] == ALERT]
    return min(got) if got else None


if __name__ == "__main__":
    print("contexts:", len(REQUESTS), " actions:", len(ACTIONS),
          " states:", N_STATES)
    print("\nseverity scale")
    for i, b in enumerate(NEWS2_BANDS):
        print("   %d  %s" % (i, b))
    print("\nescalation threshold, lowest severity at which a caregiver is summoned\n")
    print("%-24s %8s   %s" % ("profile", "frailty",
                              "  ".join("%-12s" % r[:12] for r in REQUESTS)))
    for name in CORE_NAMES:
        pi, _, _ = solve(name)
        row = []
        for req in range(len(REQUESTS)):
            t = _threshold(pi, req)
            row.append("never" if t is None else str(t))
        print("%-24s %8.2f   %s" % (name, PROFILES[name]["frailty"],
                                    "  ".join("%-12s" % v for v in row)))
    print("\n%-24s %8s   %s" % ("myopic (immediate value)", "",
                                "  ".join("%-12s" % (
                                    "never" if _threshold(myopic_policy(CORE_NAMES[3]), r)
                                    is None else str(_threshold(
                                        myopic_policy(CORE_NAMES[3]), r)))
                                    for r in range(len(REQUESTS)))))
    n2 = news2_policy()
    print("%-24s %8s   %s" % ("NEWS2 protocol", "",
                              "  ".join("%-12s" % (
                                  "never" if _threshold(n2, r) is None
                                  else str(_threshold(n2, r)))
                                  for r in range(len(REQUESTS)))))


# ---------------------------------------------------------------------------
# The reference policy.
#
# The quantity reported throughout is undiscounted mean reward per interaction,
# but the first version of this code computed the reference by discounted value
# iteration at gamma = 0.90. The two criteria order policies almost identically
# at that discount and separate as it falls, which is why a few cells of the
# sensitivity sweep exceeded 100 percent of the reference: a policy optimised
# for a short horizon can beat a short-horizon-optimal reference on a long-run
# average. A reference that can be exceeded is not a bound.
#
# The reference below is optimal for the criterion actually reported. Relative
# value iteration returns the gain g, which is the best attainable mean reward
# per interaction, together with relative action values whose differences are
# the value forgone by departing from the optimal policy. Regret of eq. (27) is
# read off those differences and is unaffected by the arbitrary offset.
# ---------------------------------------------------------------------------
def plan_average(R, T, iters=200000, tol=1e-13):
    """Relative value iteration. Returns (Q, gain).

    Q is defined up to an additive constant, which is pinned by holding one cell
    at zero; gain is the optimal undiscounted mean reward per interaction.
    """
    nR, nA = len(REQUESTS), len(ACTIONS)
    Q = np.zeros((N_STATES, nA))
    g = 0.0
    for _ in range(iters):
        Vbar = Q.reshape(nR, N_SEV, nA).max(axis=2).mean(axis=0)
        Qraw = R + (T @ Vbar)
        g = float(Qraw[0, 0])
        Qn = Qraw - g
        d = Qn - Q
        if float(d.max() - d.min()) < tol:
            Q = Qn
            break
        Q = Qn
    return Q, g


def solve_average(profile):
    """Average-reward optimal policy, action values and gain for a known user."""
    pi_idx = PROFILE_NAMES.index(profile)
    Q, g = plan_average(EXP_R[pi_idx], TRANS[pi_idx])
    return Q.argmax(axis=1), Q, g


def solve_average_model(R, T):
    Q, g = plan_average(R, T)
    return Q.argmax(axis=1), Q, g
