"""Deteriorations under fixed escalation thresholds (Figure 2), at 30 seeds."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from triage_env import step, ACTIONS, AFFINITY, ASK, ALERT, REQUESTS, CORE_NAMES
from paths import RESULTS
SEEDS, USERS, STEPS = 30, 300, 220


def crises(threshold):
    out = []
    for pf in CORE_NAMES:
        tot, n = 0, 0
        for s in range(SEEDS):
            rng = np.random.default_rng(4000 + s)
            for _ in range(USERS):
                sev = 0
                for t in range(STEPS):
                    rq = int(rng.integers(len(REQUESTS)))
                    if sev >= threshold:
                        a = ALERT
                    else:
                        vals = [AFFINITY[rq].get(ACTIONS[i], 0.0) if i not in (ASK, ALERT) else -9.0
                                for i in range(len(ACTIONS))]
                        a = int(np.argmax(vals))
                    _, sev, cri, _ = step(pf, rq, sev, a, rng)
                    tot += int(cri); n += 1
        out.append(100.0 * tot / n)
    return out


store = {"profiles": list(CORE_NAMES), "seeds": SEEDS, "n_users": USERS, "steps_per_user": STEPS, "by_threshold": {}}
for lab, key in (("escalate at 3", 3), ("escalate at 2", 2), ("never escalate", 9)):
    store["by_threshold"][lab] = dict(zip(CORE_NAMES, crises(key)))
    print(lab, store["by_threshold"][lab], flush=True)
json.dump(store, open(os.path.join(RESULTS, "har_thresholds.json"), "w"), indent=1)
print("done")
