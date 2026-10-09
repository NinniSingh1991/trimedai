"""The recognition layer on older adults (HAR70+), and what its errors do to the decision layer.

HAR70+ (Ustad et al., 2023; UCI dataset 780, CC BY 4.0): 18 adults aged 70-95, fit to frail, five using walking
aids, lower-back tri-axial accelerometer at 50 Hz in g, semi-structured free-living protocol, video-annotated.
It has no gyroscope, so every model here is accelerometer-only.

Input representation. The UCI HAR phone and the HAR70+ sensor are mounted with different axis conventions
(UCI x points up, HAR70+ back x points down) and the two devices are rotated about the vertical axis relative to each
other. A placement-robust three-channel input removes that difference without any fitting to HAR70+:
    vertical acceleration (UCI +x, HAR70+ -back_x), horizontal magnitude sqrt(y^2 + z^2), total magnitude.
Windows: 128 samples (2.56 s), 50 % overlap, within contiguous runs of a single label, as in UCI HAR.
Classes: walking, stairs up, stairs down, sitting, standing, lying (UCI order). HAR70+ "shuffling" has no UCI
counterpart and is excluded (its count is reported).

Parts
  A  external test: trained on UCI HAR (21 training subjects; and all 30), tested on all 18 HAR70+ subjects
  B  in-population reference: leave-three-subjects-out cross-validation within HAR70+ (6 folds)
  C  pooled: UCI HAR (30) + HAR70+ training folds, tested on the held-out HAR70+ subjects
  D  decision layer: the comparison of sensitivity.run_setting at the main constants, 30 seeds, with the context
     stream drawn from HAR70+ windows and recognised by the models of A and B (and UCI HAR as the reference).
Every model is trained under the main experiments' budget (12 epochs, Adam 1e-3, batch 128) for 5 training seeds.
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
UCI = os.path.join(ROOT, "data", "har_raw", "UCI HAR Dataset")
H70 = os.path.join(ROOT, "external", "har70", "har70plus")
torch.set_num_threads(2)
CLASSES = ["walking", "stairs up", "stairs down", "sitting", "standing", "lying"]
H70_MAP = {1: 0, 4: 1, 5: 2, 7: 3, 6: 4, 8: 5}          # HAR70+ label -> UCI class index
WIN, STEP = 128, 64
def _seed_range():
    r = os.environ.get("TRAIN_SEEDS", "0-29"); a, b = r.split("-")
    return list(range(int(a), int(b) + 1))


TRAIN_SEEDS = _seed_range()
SUFFIX = os.environ.get("CHUNK", "")


def feats(v, y, z):
    h = np.sqrt(y ** 2 + z ** 2)
    m = np.sqrt(v ** 2 + y ** 2 + z ** 2)
    return np.stack([v, h, m], -1).astype(np.float32)


def load_uci(split):
    acc = [np.array(open(os.path.join(UCI, split, "Inertial Signals", "total_acc_%s_%s.txt" % (a, split))).read().split(),
                    dtype=np.float32).reshape(-1, 128) for a in "xyz"]
    y = np.array(open(os.path.join(UCI, split, "y_%s.txt" % split)).read().split(), dtype=np.int64) - 1
    s = np.array(open(os.path.join(UCI, split, "subject_%s.txt" % split)).read().split(), dtype=np.int64)
    return feats(acc[0], acc[1], acc[2]), y, s


def load_h70():
    X, Y, S, excluded = [], [], [], 0
    for f in sorted(glob.glob(os.path.join(H70, "*.csv"))):
        sid = int(os.path.basename(f)[:3])
        d = pd.read_csv(f, usecols=["back_x", "back_y", "back_z", "label"])
        F = feats(-d.back_x.values, d.back_y.values, d.back_z.values)
        lab = d.label.values
        run_start = np.r_[0, np.flatnonzero(np.diff(lab)) + 1, len(lab)]
        for a, b in zip(run_start[:-1], run_start[1:]):
            l = int(lab[a])
            n_win = max(0, (b - a - WIN) // STEP + 1)
            if l not in H70_MAP:
                excluded += n_win; continue
            for k in range(n_win):
                X.append(F[a + k * STEP: a + k * STEP + WIN]); Y.append(H70_MAP[l]); S.append(sid)
    return np.stack(X), np.array(Y), np.array(S), excluded


class CNN3(nn.Module):
    """the main experiments' convolutional backbone, with three input channels"""
    def __init__(self, n_ch=3, n_cls=6):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(n_ch, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.f(x.transpose(1, 2))


def fit_predict(Xtr, ytr, tests, seed, epochs=12, bs=128, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = CNN3(); opt = torch.optim.Adam(net.parameters(), lr=lr); lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            loss = lossf(net(Xt[b]), yt[b]); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval(); out = []
    with torch.no_grad():
        for X in tests:
            xt = torch.tensor(X)
            out.append(torch.cat([net(xt[i:i + 1024]) for i in range(0, len(xt), 1024)]).argmax(1).numpy())
    return out


def scores(y, p):
    cm = np.zeros((6, 6), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    present = cm.sum(1) > 0
    recall = np.where(present, np.diag(cm) / np.maximum(cm.sum(1), 1), np.nan)
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = np.where(present, 2 * prec * recall / np.maximum(prec + recall, 1e-12), np.nan)
    return dict(accuracy=float((y == p).mean()), balanced_accuracy=float(np.nanmean(recall)),
                macro_f1=float(np.nanmean(f1)), recall={CLASSES[i]: (None if np.isnan(recall[i]) else float(recall[i]))
                                                       for i in range(6)}, confusion=cm.tolist())


def summarise(runs):
    keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "sd": float(np.std([r[k] for r in runs], ddof=1))}
            for k in keys} | {"per_seed": runs}


def main():
    t0 = time.perf_counter()
    Xa, ya, sa = load_uci("train"); Xb, yb, sb = load_uci("test")
    Xh, yh, sh, excl = load_h70()
    mu = Xa.reshape(-1, 3).mean(0); sd = Xa.reshape(-1, 3).std(0) + 1e-8      # UCI training statistics only
    nz = lambda X: (X - mu) / sd
    Xa, Xb, Xh = nz(Xa), nz(Xb), nz(Xh)
    subj_h = np.unique(sh)
    info = {"uci_train_windows": int(len(ya)), "uci_test_windows": int(len(yb)),
            "har70_windows": int(len(yh)), "har70_subjects": int(len(subj_h)),
            "har70_excluded_shuffling_windows": int(excl),
            "har70_windows_per_class": {CLASSES[c]: int((yh == c).sum()) for c in range(6)},
            "har70_subjects_per_class": {CLASSES[c]: int(len(np.unique(sh[yh == c]))) for c in range(6)}}
    print(json.dumps(info, indent=1), flush=True)
    res = {"info": info, "A_uci21": {"uci_test": [], "har70": []}, "A_uci30": {"har70": []},
           "B_har70_cv": [], "C_pooled": []}
    preds = {"uci_test_A21": [], "har70_A21": [], "har70_A30": [], "har70_B": [], "har70_C": []}
    X30, y30 = np.concatenate([Xa, Xb]), np.concatenate([ya, yb])
    folds = np.array_split(subj_h, 6)
    for seed in TRAIN_SEEDS:
        p_uci, p_h = fit_predict(Xa, ya, [Xb, Xh], seed)
        res["A_uci21"]["uci_test"].append(scores(yb, p_uci)); res["A_uci21"]["har70"].append(scores(yh, p_h))
        preds["uci_test_A21"].append(p_uci); preds["har70_A21"].append(p_h)
        (p_h30,) = fit_predict(X30, y30, [Xh], seed)
        res["A_uci30"]["har70"].append(scores(yh, p_h30)); preds["har70_A30"].append(p_h30)
        pB = np.full(len(yh), -1); pC = np.full(len(yh), -1)
        for fold in folds:
            te = np.isin(sh, fold); tr = ~te
            (pB[te],) = fit_predict(Xh[tr], yh[tr], [Xh[te]], seed)
            (pC[te],) = fit_predict(np.concatenate([X30, Xh[tr]]), np.concatenate([y30, yh[tr]]), [Xh[te]], seed)
        res["B_har70_cv"].append(scores(yh, pB)); res["C_pooled"].append(scores(yh, pC))
        preds["har70_B"].append(pB); preds["har70_C"].append(pC)
        print("seed", seed, "UCI->UCI %.4f  UCI21->H70 %.4f  UCI30->H70 %.4f  H70 CV %.4f  pooled %.4f  (%.0fs)" % (
            res["A_uci21"]["uci_test"][-1]["accuracy"], res["A_uci21"]["har70"][-1]["accuracy"],
            res["A_uci30"]["har70"][-1]["accuracy"], res["B_har70_cv"][-1]["accuracy"],
            res["C_pooled"][-1]["accuracy"], time.perf_counter() - t0), flush=True)
    out = {"info": info,
           "A_uci21_on_uci_test": summarise(res["A_uci21"]["uci_test"]),
           "A_uci21_on_har70": summarise(res["A_uci21"]["har70"]),
           "A_uci30_on_har70": summarise(res["A_uci30"]["har70"]),
           "B_har70_leave3out": summarise(res["B_har70_cv"]),
           "C_pooled_on_har70": summarise(res["C_pooled"]),
           "_budget": {"training_seeds": TRAIN_SEEDS, "epochs": 12, "window": WIN, "step": STEP,
                       "input": "vertical, horizontal magnitude, total magnitude of acceleration (g)"}}
    json.dump(out, open(os.path.join(OUT, "e1_recognition%s.json" % SUFFIX), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "e1_predictions%s.npz" % SUFFIX), yh=yh, sh=sh, yb=yb,
                        **{k: np.stack(v) for k, v in preds.items()})
    print("done %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
