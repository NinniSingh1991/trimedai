"""The three stages of TriMedAI, and the comparators they are measured against.

The framework is revised in three ways relative to its first formulation, each
addressing a limitation that the earlier version could not answer.

The belief is now over a continuous latent rather than a short list of named
profiles. A person is described by a frailty in [0,1] and a preference weight per
channel, and the posterior over that latent is carried by a particle filter. The
earlier design had to assume that every user was one of four known types, which
is both unrealistic and the reason a comparator supplied with a precomputed
policy for each type could outperform it.

The policy is now obtained by solving the inferred model rather than by nudging a
table towards it. Whenever the belief moves the agent rebuilds the reward and
transition functions implied by its posterior and solves them by value iteration,
warm-started from the previous solution and, at the outset, from the meta-learned
initialisation. Solving a model of this size costs under a millisecond, so there
is no reason to approximate.

Planning is risk-averse in the frailty direction. The agent plans against an
upper quantile of its frailty posterior rather than the mean, which amounts to
preparing for the frailer end of what the person might turn out to be. This costs
nothing and it moves the errors made while the belief is still wide from the
dangerous side to the cautious side.
"""
import numpy as np

from triage_env import (ACTIONS, REQUESTS, PROFILES, PROFILE_NAMES, CORE_NAMES,
                        N_CORE, N_SEV, N_STATES, ALERT, ASK, GAMMA, CRISIS,
                        NEWS2_URGENT_REVIEW, EXP_R, TRANS, P_WORSE, COMFORT,
                        _AFF, _URG, latent_of, comfort_matrix, model_for, plan,
                        sid, step)

N_ACT = len(ACTIONS)
N_REQ = len(REQUESTS)


# ============================================================ reward inference
class ParticleBelief:
    """Posterior over a continuous latent, carried by a particle filter.

    Two observation channels drive it. The channel the person indicates they
    would have preferred identifies how they can be supported; whether the
    monitoring layer reports that the condition worsened identifies how readily
    they deteriorate. The second has no counterpart in the standard cooperative
    formulation and is what makes the belief informative about escalation timing.
    """

    def __init__(self, n=256, beta=4.0, rng=None, jitter=0.03):
        self.rng = rng or np.random.default_rng(0)
        self.n, self.beta, self.jitter = n, beta, jitter
        self.frailty = self.rng.random(n)
        self.pref = self.rng.random((n, N_ACT))
        self.logw = np.zeros(n)

    @property
    def weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def observe_preference(self, context, chosen_action):
        C = _AFF[context][None, :] * (0.35 + 0.65 * self.pref)      # n x actions
        v = self.beta * C
        v[:, ASK] = -1e9                                            # not a channel
        v = v - v.max(axis=1, keepdims=True)
        lik = np.exp(v)
        lik /= lik.sum(axis=1, keepdims=True)
        self.logw += np.log(lik[:, chosen_action] + 1e-12)
        self._maybe_resample()

    def observe_transition(self, context, worsened):
        p = np.clip(self.frailty * _URG[context], 1e-6, 1 - 1e-6)
        self.logw += np.log(p if worsened else 1.0 - p)
        self._maybe_resample()

    def _maybe_resample(self):
        w = self.weights
        ess = 1.0 / np.sum(w ** 2)
        if ess >= 0.5 * self.n:
            return
        # systematic resampling, then a small jitter so the cloud does not collapse
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.searchsorted(np.cumsum(w), pos)
        idx = np.clip(idx, 0, self.n - 1)
        self.frailty = np.clip(self.frailty[idx] +
                               self.rng.normal(0, self.jitter, self.n), 0.0, 1.0)
        self.pref = np.clip(self.pref[idx] +
                            self.rng.normal(0, self.jitter, (self.n, N_ACT)), 0.0, 1.0)
        self.logw = np.zeros(self.n)

    def mean_latent(self):
        w = self.weights
        return float(w @ self.frailty), w @ self.pref

    def planning_latent(self, quantile=0.75):
        """mean preference, but an upper quantile of frailty, so that the errors
        made while the posterior is still wide fall on the cautious side"""
        w = self.weights
        order = np.argsort(self.frailty)
        cw = np.cumsum(w[order])
        fr = float(self.frailty[order][np.searchsorted(cw, quantile)
                                       if np.searchsorted(cw, quantile) < self.n
                                       else self.n - 1])
        return fr, w @ self.pref

    def frailty_sd(self):
        w = self.weights
        m = w @ self.frailty
        return float(np.sqrt(w @ (self.frailty - m) ** 2))

    def latent_error(self, profile):
        """distance from the true latent, used to report calibration directly"""
        fr_true, pref_true = latent_of(profile)
        fr, pref = self.mean_latent()
        return abs(fr - fr_true), float(np.mean(np.abs(pref - pref_true)))


# =============================================================== policy learning
class PlanningPolicy:
    """Solves the model implied by the current belief, warm-started."""

    def __init__(self, init=None, replan_every=10, sweeps=40, quantile=0.75):
        self.Q = np.zeros((N_STATES, N_ACT)) if init is None else init.copy()
        self.replan_every, self.sweeps, self.quantile = replan_every, sweeps, quantile
        self.k = 0

    def act(self, state):
        return int(np.argmax(self.Q[state]))

    def observe(self, belief, force=False):
        self.k += 1
        if not force and self.k % self.replan_every:
            return
        fr, pref = belief.planning_latent(self.quantile)
        R, T = model_for(fr, pref)
        self.Q = plan(R, T, sweeps=self.sweeps, Q0=self.Q)


class QPolicy:
    """Tabular Q-learning, retained for the ablation variants and the pooled
    comparator that learn without an inferred model."""

    def __init__(self, alpha=0.25, gamma=GAMMA, eps=0.15, rng=None, init=None):
        self.Q = np.zeros((N_STATES, N_ACT)) if init is None else init.copy()
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.rng = rng or np.random.default_rng(0)

    def act(self, state, explore=True, eps=None):
        e = self.eps if eps is None else eps
        if explore and e > 0.0 and self.rng.random() < e:
            return int(self.rng.integers(N_ACT))
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state):
        target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def severity_values(self):
        return self.Q.reshape(N_REQ, N_SEV, N_ACT).max(axis=2).mean(axis=0)

    def plan_update(self, state, reward_vector, transition_matrix):
        target = reward_vector + self.gamma * (transition_matrix @
                                               self.severity_values())
        self.Q[state] += self.alpha * (target - self.Q[state])


# ================================================================ meta-learning
def meta_initialisation(steps=None, seed=0, n_draws=24):
    """A starting policy averaged over the latent space, not over four names.

    Latents are drawn from the prior, each is solved exactly, and the solutions
    are averaged by the first-order outer step of Nichol et al. The result is a
    policy that is close to correct for whichever person arrives next and, in
    particular, already escalates in the severe states a new user has not yet
    reached.
    """
    rng = np.random.default_rng(seed)
    meta = np.zeros((N_STATES, N_ACT))
    for k in range(n_draws):
        fr = float(rng.random())
        pref = rng.random(N_ACT)
        R, T = model_for(fr, pref)
        Qk = plan(R, T, sweeps=400, Q0=meta)
        meta += (Qk - meta) / (k + 1)
    return meta


def global_policy(steps=60000, seed=0):
    """A single policy learned from feedback pooled across the population: the
    reinforcement-learning-from-human-feedback comparator. It learns the
    sequential structure and does escalate, but holds one policy for everyone."""
    rng = np.random.default_rng(seed)
    pol = QPolicy(rng=rng, eps=0.2)
    req, sev = int(rng.integers(N_REQ)), 0
    profile = CORE_NAMES[int(rng.integers(N_CORE))]
    for t in range(steps):
        s = sid(req, sev)
        a = pol.act(s)
        r, nsev, _, _ = step(profile, req, sev, a, rng)
        nreq = int(rng.integers(N_REQ))
        pol.update(s, a, r, sid(nreq, nsev))
        req, sev = nreq, nsev
        if t % 60 == 59:
            profile = CORE_NAMES[int(rng.integers(N_CORE))]
            sev = 0
    return pol.Q


# ==================================================================== baselines
def population_reward_table():
    """expected immediate reward averaged over the population, which is what a
    non-personalised assistant scores its options with"""
    return EXP_R[:N_CORE].mean(axis=0)


def mcdm_policy(weights=(0.45, 0.35, 0.20)):
    """A weighted multi-criteria rule, representing the decision-support family
    that the application literature in this venue most often adopts.

    Three criteria are scored on a common scale and combined linearly: how well
    the channel suits the context, how urgent the context is, and how severe the
    person's condition is. It is transparent and needs no learning, which is why
    it is popular, and it has no way to represent the fact that the value of
    escalating is realised later.
    """
    w_fit, w_urg, w_sev = weights
    pi = np.zeros(N_STATES, dtype=int)
    for rq in range(N_REQ):
        fit = _AFF[rq] / (_AFF[rq].max() + 1e-9)
        for sv in range(N_SEV):
            score = w_fit * fit.copy()
            score[ASK] = 0.15
            escalate = w_urg * _URG[rq] + w_sev * (sv / (CRISIS - 1))
            score[ALERT] = escalate
            pi[sid(rq, sv)] = int(np.argmax(score))
    return pi


# ====================================================== policy-update variants
# Reviewer request: the article claims that backing up every action at a visited
# state from the belief-implied model is worth several points of attained value
# and a large reduction in the interactions a new user must spend. That claim
# needs an ablation that changes the backup rule and nothing else. The three
# classes below share the belief, the meta-initialisation and the visited states
# and differ only in what one interaction is allowed to revise.

class RowBackupPolicy:
    """Equations (23) to (25) taken literally: at the visited state, average the
    reward and the transition kernel under the belief and back up every action.

    This is the update written in Table 4. It touches one state per interaction,
    unlike the full model solve used by the framework, so it isolates the
    belief-averaged full-row backup from the value iteration that surrounds it.
    """

    def __init__(self, init=None, eta=1.0, gamma=GAMMA, quantile=0.75):
        self.Q = np.zeros((N_STATES, N_ACT)) if init is None else init.copy()
        self.eta, self.gamma, self.quantile = eta, gamma, quantile

    def act(self, state):
        return int(np.argmax(self.Q[state]))

    def _vbar(self):
        return self.Q.reshape(N_REQ, N_SEV, N_ACT).max(axis=2).mean(axis=0)

    def observe(self, belief, state, action=None, reward=None):
        fr, pref = belief.planning_latent(self.quantile)
        R, T = model_for(fr, pref)
        target = R[state] + self.gamma * (T[state] @ self._vbar())
        self.Q[state] += self.eta * (target - self.Q[state])


class CellBackupPolicy(RowBackupPolicy):
    """The same belief-averaged target, applied to the action that was taken and
    to no other. Everything else is identical to RowBackupPolicy, so the
    difference between the two is the loop over actions in Table 4 line 2."""

    def observe(self, belief, state, action=None, reward=None):
        if action is None:
            return
        fr, pref = belief.planning_latent(self.quantile)
        R, T = model_for(fr, pref)
        target = R[state, action] + self.gamma * (T[state, action] @ self._vbar())
        self.Q[state, action] += self.eta * (target - self.Q[state, action])


class ObservedQPolicy(RowBackupPolicy):
    """Ordinary Q-learning: the single cell belonging to the action taken is
    moved towards the reward that was actually observed. The belief is available
    but unused, which is the point of the comparison."""

    def __init__(self, init=None, alpha=0.25, gamma=GAMMA):
        super().__init__(init=init, eta=alpha, gamma=gamma)
        self.alpha = alpha

    def observe(self, belief, state, action=None, reward=None, next_state=None):
        if action is None or reward is None:
            return
        nxt = state if next_state is None else next_state
        target = reward + self.gamma * np.max(self.Q[nxt])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
