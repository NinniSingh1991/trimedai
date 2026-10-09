"""The sensitivity sweep of sensitivity.py at 30 seeds, one swept constant per process (row 0 to 5)."""
import json, os, sys, time
os.environ.setdefault("N_SEEDS", "30")
os.environ.setdefault("N_USERS", "50")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sensitivity as S
import perception as PC
from triage_env import reset_config, reconfigure

row = int(sys.argv[1])
label, key, values = S.GRID[row]
out_dir = os.path.join(os.path.dirname(HERE), "results"); os.makedirs(out_dir, exist_ok=True)
Xtr, ytr, Xte, yte = PC.load(S.SCRATCH)
bb = {n: PC.train(n, Xtr, ytr, Xte, yte, seed=0) for n in ("CNN", "MLP")}
t0 = time.perf_counter(); res = []
for v in values:
    reset_config(); reconfigure(**{key: v})
    r = S.run_setting(bb, yte); res.append({"value": v, **r})
    print(label, v, {m: round(r[m]["normalised"], 2) for m in S.SHOWN}, flush=True)
reset_config()
json.dump({"label": label, "parameter": key, "values": values, "results": res,
           "_budget": {"seeds": len(S.SEEDS), "n_users": S.N_USERS, "steps_per_user": S.T_STEPS,
                       "evaluation_from_step": S.EVAL_FROM}, "seconds": time.perf_counter() - t0},
          open(os.path.join(out_dir, "sens30_row%d.json" % row), "w"), indent=1)
print("done row", row, "%.0fs" % (time.perf_counter() - t0))
