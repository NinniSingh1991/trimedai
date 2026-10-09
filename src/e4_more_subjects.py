"""The recognition layer on 110 people (UCI HAR 30 + the 80 KU-HAR participants with the six activities).

KU-HAR (Sikder & Nahid, 2021, Pattern Recognition Letters; Mendeley Data 10.17632/45f952y38r.5, CC BY 4.0):
90 adults, smartphone in a waist bag, accelerometer with gravity removed (m/s^2) and gyroscope (rad/s), trimmed and
interpolated to 100 Hz. The six UCI HAR classes are present: Walk, Stair-up, Stair-down, Sit, Stand, Lay.

Because KU-HAR's accelerometer has gravity removed, the matching UCI HAR signals are body acceleration (g) and body
angular velocity (rad/s): six channels. KU-HAR acceleration is converted to g and both signals are decimated from
100 Hz to 50 Hz (mean of consecutive pairs), then cut into 128-sample windows with 50 % overlap within each trial,
as in UCI HAR. Inputs are standardised with training-fold statistics.

Protocol: subject-grouped 5-fold cross-validation over all 110 people (each fold holds out 6 UCI and 16 KU-HAR
subjects), so every person is tested exactly once by a model that never saw them, for 5 training seeds.
Reference: the same six-channel model trained on the 21 UCI training subjects and tested on the 9 UCI test subjects.
"""
import glob, json, os, sys, time, re
import numpy as np, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
UCI = os.path.join(ROOT, "data", "har_raw", "UCI HAR Dataset")
KU = os.path.join(ROOT, "external", "kuhar")
torch.set_num_threads(2)
CLASSES = ["walking", "stairs up", "stairs down", "sitting", "standing", "lying"]
KU_MAP = {"11.Walk": 0, "15.Stair-up": 1, "16.Stair-down": 2, "1.Sit": 3, "0.Stand": 4, "5.Lay": 5}
WIN, STEP, G = 128, 64, 9.80665
def _seed_range():
    r = os.environ.get("TRAIN_SEEDS", "0-29"); a, b = r.split("-")
    return list(range(int(a), int(b) + 1))


TRAIN_SEEDS = _seed_range()
SUFFIX = os.environ.get("CHUNK", "")
SIG = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]


def load_uci(split):
    X = np.stack([np.array(open(os.path.join(UCI, split, "Inertial Signals", "%s_%s.txt" % (s, split))).read().split(),
                           dtype=np.float32).reshape(-1, 128) for s in SIG], -1)
    y = np.array(open(os.path.join(UCI, split, "y_%s.txt" % split)).read().split(), dtype=np.int64) - 1
    s = np.array(open(os.path.join(UCI, split, "subject_%s.txt" % split)).read().split(), dtype=np.int64)
    return X, y, s


def load_ku():
    X, Y, S, trials = [], [], [], 0
    for folder, c in KU_MAP.items():
        for f in sorted(glob.glob(os.path.join(KU, folder, "*.csv"))):
            sid = int(re.match(r"(\d+)_", os.path.basename(f)).group(1))
            d = np.loadtxt(f, delimiter=",", dtype=np.float64)
            if d.ndim != 2 or d.shape[1] < 8:
                continue
            acc, gyr = d[:, 1:4] / G, d[:, 5:8]
            n = (len(d) // 2) * 2
            sig = np.concatenate([acc[:n], gyr[:n]], 1).reshape(n // 2, 2, 6).mean(1)   # 100 Hz -> 50 Hz
            trials += 1
            for k in range(max(0, (len(sig) - WIN) // STEP + 1)):
                X.append(sig[k * STEP: k * STEP + WIN].astype(np.float32)); Y.append(c); S.append(sid)
    return np.stack(X), np.array(Y), np.array(S), trials


class CNN6(nn.Module):
    """the main experiments' convolutional backbone, with six input channels"""
    def __init__(self, n_ch=6, n_cls=6):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(n_ch, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.f(x.transpose(1, 2))


def fit_predict(Xtr, ytr, tests, seed, epochs=12, bs=128, lr=1e-3):
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    torch.manual_seed(seed); np.random.seed(seed)
    net = CNN6(); opt = torch.optim.Adam(net.parameters(), lr=lr); lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor((Xtr - mu) / sd), torch.tensor(ytr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            loss = lossf(net(Xt[b]), yt[b]); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval(); out = []
    with torch.no_grad():
        for X in tests:
            xt = torch.tensor((X - mu) / sd)
            out.append(torch.cat([net(xt[i:i + 1024]) for i in range(0, len(xt), 1024)]).argmax(1).numpy())
    return out


def scores(y, p):
    cm = np.zeros((6, 6), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1); prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * recall / np.maximum(prec + recall, 1e-12)
    return dict(accuracy=float((y == p).mean()), balanced_accuracy=float(recall.mean()), macro_f1=float(f1.mean()),
                recall={CLASSES[i]: float(recall[i]) for i in range(6)}, confusion=cm.tolist())


def summarise(runs):
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "sd": float(np.std([r[k] for r in runs], ddof=1))}
            for k in ("accuracy", "balanced_accuracy", "macro_f1")} | {"per_seed": runs}


def main():
    t0 = time.perf_counter()
    Xa, ya, sa = load_uci("train"); Xb, yb, sb = load_uci("test")
    Xk, yk, sk, trials = load_ku()
    Xu, yu, su = np.concatenate([Xa, Xb]), np.concatenate([ya, yb]), np.concatenate([sa, sb])
    info = {"uci_subjects": int(len(np.unique(su))), "kuhar_subjects": int(len(np.unique(sk))),
            "kuhar_trials_used": int(trials), "kuhar_windows": int(len(yk)), "uci_windows": int(len(yu)),
            "kuhar_windows_per_class": {CLASSES[c]: int((yk == c).sum()) for c in range(6)}}
    print(json.dumps(info, indent=1), flush=True)
    rng = np.random.default_rng(12345)
    fu = np.array_split(rng.permutation(np.unique(su)), 5); fk = np.array_split(rng.permutation(np.unique(sk)), 5)
    X = np.concatenate([Xu, Xk]); y = np.concatenate([yu, yk])
    src = np.r_[np.zeros(len(yu), int), np.ones(len(yk), int)]
    subj = np.r_[su, sk + 100000]
    ref, pooled, pooled_uci, pooled_ku, preds = [], [], [], [], []
    for seed in TRAIN_SEEDS:
        (pr,) = fit_predict(Xa, ya, [Xb], seed); ref.append(scores(yb, pr))
        p = np.full(len(y), -1)
        for k in range(5):
            te = np.isin(subj, np.r_[fu[k], fk[k] + 100000]); tr = ~te
            (p[te],) = fit_predict(X[tr], y[tr], [X[te]], seed)
        pooled.append(scores(y, p)); pooled_uci.append(scores(y[src == 0], p[src == 0]))
        pooled_ku.append(scores(y[src == 1], p[src == 1])); preds.append(p)
        print("seed", seed, "UCI21->UCI9 %.4f  pooled 120 %.4f  (UCI part %.4f, KU-HAR part %.4f)  %.0fs" % (
            ref[-1]["accuracy"], pooled[-1]["accuracy"], pooled_uci[-1]["accuracy"], pooled_ku[-1]["accuracy"],
            time.perf_counter() - t0), flush=True)
    json.dump({"info": info, "reference_uci21_on_uci9": summarise(ref), "pooled_cv_120": summarise(pooled),
               "pooled_cv_uci_part": summarise(pooled_uci), "pooled_cv_kuhar_part": summarise(pooled_ku),
               "_budget": {"training_seeds": TRAIN_SEEDS, "folds": 5, "epochs": 12, "window": WIN, "step": STEP,
                           "channels": SIG}},
              open(os.path.join(OUT, "e4_more_subjects%s.json" % SUFFIX), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "e4_predictions%s.npz" % SUFFIX), y=y, src=src, subj=subj, pred=np.stack(preds))
    print("done %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
