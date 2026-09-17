"""Download the UCI Human Activity Recognition corpus and cache it as arrays.

Thirty volunteers wearing a waist-mounted smartphone, six activities of daily
living, inertial signals sampled at 50 Hz and segmented into 2.56 second windows.
The train and test partitions supplied with the corpus are split by subject, so a
model is evaluated on people it has never seen, which is the correct protocol for
a wearable system and is kept here.

Each window is 128 samples across 9 channels: body acceleration, body angular
velocity and total acceleration, each in three axes.
"""
import io, os, socket, time, urllib.request, zipfile
import numpy as np

socket.setdefaulttimeout(60)

from paths import DATA as SCRATCH, RESULTS, FIGURES
RAW = os.path.join(SCRATCH, "har_raw")  # under data/
URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
]
SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z",
           "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]
ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]


def fetch():
    os.makedirs(RAW, exist_ok=True)
    marker = os.path.join(RAW, "UCI HAR Dataset", "README.txt")
    if os.path.exists(marker):
        print("corpus already present")
        return
    blob = None
    for url in URLS:
        try:
            print("downloading", url, flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            chunks, got, t0 = [], 0, time.time()
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length") or 0)
                while True:
                    c = r.read(1 << 18)
                    if not c:
                        break
                    chunks.append(c)
                    got += len(c)
                    if got % (1 << 22) < (1 << 18):
                        print("   %6.1f MB%s  %5.0fs" %
                              (got / 1e6,
                               (" of %.1f" % (total / 1e6)) if total else "",
                               time.time() - t0), flush=True)
            blob = b"".join(chunks)
            print("  received %.1f MB in %.0fs" % (len(blob) / 1e6, time.time() - t0),
                  flush=True)
            break
        except Exception as e:
            print("  failed:", e, flush=True)
    if blob is None:
        raise SystemExit("could not download the corpus")
    z = zipfile.ZipFile(io.BytesIO(blob))
    z.extractall(RAW)
    # some mirrors nest a second archive inside the first
    inner = [n for n in os.listdir(RAW) if n.lower().endswith(".zip")]
    for n in inner:
        with zipfile.ZipFile(os.path.join(RAW, n)) as z2:
            z2.extractall(RAW)
    print("extracted to", RAW)


def _root():
    for dirpath, dirnames, filenames in os.walk(RAW):
        if "README.txt" in filenames and "train" in dirnames and "test" in dirnames:
            return dirpath
    raise SystemExit("could not locate the dataset root under " + RAW)


def _read_matrix(path, width):
    """whitespace-separated floats; far faster than loadtxt on files this size"""
    with open(path, "r") as f:
        vals = np.array(f.read().split(), dtype=np.float32)
    return vals.reshape(-1, width)


def load_split(root, split):
    sig = []
    for s in SIGNALS:
        path = os.path.join(root, split, "Inertial Signals", "%s_%s.txt" % (s, split))
        sig.append(_read_matrix(path, 128))
    X = np.stack(sig, axis=-1)                       # (N, 128, 9)
    with open(os.path.join(root, split, "y_%s.txt" % split)) as f:
        y = np.array(f.read().split(), dtype=np.int64) - 1
    with open(os.path.join(root, split, "subject_%s.txt" % split)) as f:
        subj = np.array(f.read().split(), dtype=np.int64)
    return X, y, subj


def main():
    fetch()
    root = _root()
    print("root:", root)
    Xtr, ytr, str_ = load_split(root, "train")
    Xte, yte, ste = load_split(root, "test")
    print("train %s  test %s" % (Xtr.shape, Xte.shape))
    print("train subjects %d, test subjects %d, overlap %d"
          % (len(set(str_)), len(set(ste)), len(set(str_) & set(ste))))
    print("\nclass balance (train / test):")
    for i, a in enumerate(ACTIVITIES):
        print("  %d %-20s %5d / %5d" % (i, a, int((ytr == i).sum()), int((yte == i).sum())))

    # standardise per channel using training statistics only
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    np.save(os.path.join(SCRATCH, "har_Xtr.npy"), Xtr)
    np.save(os.path.join(SCRATCH, "har_ytr.npy"), ytr)
    np.save(os.path.join(SCRATCH, "har_Xte.npy"), Xte)
    np.save(os.path.join(SCRATCH, "har_yte.npy"), yte)
    np.save(os.path.join(SCRATCH, "har_subj_te.npy"), ste)
    print("\ncached arrays written to", SCRATCH)


if __name__ == "__main__":
    main()
