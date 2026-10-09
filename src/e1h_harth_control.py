"""Age versus device and setting: is the drop on older adults caused by age, or by the change of device and setting?

HARTH (Logacjov et al., 2021, Sensors 21(23):7853; UCI dataset 779, CC BY 4.0): 22 adults aged 25-68 in free-living
conditions with the SAME lower-back Axivity sensor, axis convention, label scheme and annotation protocol as HAR70+.
One subject (S006) was recorded at 100 Hz and is decimated to 50 Hz (mean of consecutive pairs); the rest are 50 Hz.
Inputs, windows, classes, model and training budget are identical to e1_older_adults.py.

  young_to_old    trained on HARTH (younger, same sensor) -> tested on HAR70+ (older)       : age shift only
  uci_to_harth    trained on UCI HAR (smartphone, lab)     -> tested on HARTH (same age span): device/setting shift
  harth_cv        leave-subjects-out within HARTH (6 folds)                                   : in-domain reference
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import e1_older_adults as E

HARTH = os.path.join(E.ROOT, "external", "harth", "harth")


def load_harth():
    X, Y, S = [], [], []
    for f in sorted(glob.glob(os.path.join(HARTH, "*.csv"))):
        sid = int(os.path.basename(f)[1:4])
        d = pd.read_csv(f, usecols=["timestamp", "back_x", "back_y", "back_z", "label"])
        t = pd.to_datetime(d.timestamp.values[:2000])
        if np.median(np.diff(t.values).astype("timedelta64[us]").astype(float)) < 15000:   # 100 Hz file
            n = (len(d) // 2) * 2
            v = d[["back_x", "back_y", "back_z"]].values[:n].reshape(n // 2, 2, 3).mean(1)
            lab = d.label.values[:n:2]
        else:
            v = d[["back_x", "back_y", "back_z"]].values; lab = d.label.values
        F = E.feats(-v[:, 0], v[:, 1], v[:, 2])
        starts = np.r_[0, np.flatnonzero(np.diff(lab)) + 1, len(lab)]
        for a, b in zip(starts[:-1], starts[1:]):
            l = int(lab[a])
            if l not in E.H70_MAP:
                continue
            for k in range(max(0, (b - a - E.WIN) // E.STEP + 1)):
                X.append(F[a + k * E.STEP: a + k * E.STEP + E.WIN]); Y.append(E.H70_MAP[l]); S.append(sid)
    return np.stack(X), np.array(Y), np.array(S)


def main():
    t0 = time.perf_counter()
    Xa, ya, _ = E.load_uci("train"); Xb, yb, _ = E.load_uci("test")
    Xh, yh, sh, _ = E.load_h70(); Xt, yt, st = load_harth()
    mu = Xa.reshape(-1, 3).mean(0); sd = Xa.reshape(-1, 3).std(0) + 1e-8
    nz = lambda X: (X - mu) / sd
    Xa, Xb, Xh, Xt = nz(Xa), nz(Xb), nz(Xh), nz(Xt)
    X30, y30 = np.concatenate([Xa, Xb]), np.concatenate([ya, yb])
    info = {"harth_subjects": int(len(np.unique(st))), "harth_windows": int(len(yt)),
            "harth_windows_per_class": {E.CLASSES[c]: int((yt == c).sum()) for c in range(6)}}
    print(json.dumps(info, indent=1), flush=True)
    r = {"young_to_old": [], "uci_to_harth": [], "harth_cv": []}; preds = []
    folds = np.array_split(np.unique(st), 6)
    for seed in E.TRAIN_SEEDS:
        (p1,) = E.fit_predict(Xt, yt, [Xh], seed); r["young_to_old"].append(E.scores(yh, p1)); preds.append(p1)
        (p2,) = E.fit_predict(X30, y30, [Xt], seed); r["uci_to_harth"].append(E.scores(yt, p2))
        p3 = np.full(len(yt), -1)
        for fold in folds:
            te = np.isin(st, fold)
            (p3[te],) = E.fit_predict(Xt[~te], yt[~te], [Xt[te]], seed)
        r["harth_cv"].append(E.scores(yt, p3))
        print("seed", seed, "HARTH->HAR70+ %.4f  UCI30->HARTH %.4f  HARTH CV %.4f  (%.0fs)" % (
            r["young_to_old"][-1]["accuracy"], r["uci_to_harth"][-1]["accuracy"], r["harth_cv"][-1]["accuracy"],
            time.perf_counter() - t0), flush=True)
    json.dump({"info": info, **{k: E.summarise(v) for k, v in r.items()}},
              open(os.path.join(E.OUT, "e1h_harth_control%s.json" % E.SUFFIX), "w"), indent=1)
    np.savez_compressed(os.path.join(E.OUT, "e1h_predictions%s.npz" % E.SUFFIX), yh=yh, har70_harth=np.stack(preds))
    print("done %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
