"""Context-recognition backbones over wearable inertial signals.

Each window is 128 samples at 50 Hz across nine channels: body acceleration,
body angular velocity and total acceleration, in three axes each. Unlike the
image proxy used previously, this is a genuine multivariate time series, so the
recurrent models are the architecturally natural choice rather than being
handicapped by construction, and the convolutional model has to earn its place
with one-dimensional convolution over time.

Training and evaluation use the subject-wise partition supplied with the corpus,
so every model is assessed on people it has never seen.
"""
import numpy as np, time, torch, torch.nn as nn

torch.set_num_threads(4)
T_STEPS, N_CH, N_CLS = 128, 9, 6


class CNN(nn.Module):
    """one-dimensional convolution over time, shared across channels"""

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(N_CH, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, N_CLS))

    def forward(self, x):
        return self.f(x.transpose(1, 2))


class RNN(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.rnn = nn.RNN(N_CH, hidden, batch_first=True)
        self.out = nn.Linear(hidden, N_CLS)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h[:, -1])


class LSTM(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.rnn = nn.LSTM(N_CH, hidden, batch_first=True)
        self.out = nn.Linear(hidden, N_CLS)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h[:, -1])


class MLP(nn.Module):
    """no temporal structure at all, the reference against which it is judged"""

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Flatten(), nn.Linear(T_STEPS * N_CH, 256),
                               nn.ReLU(), nn.Linear(256, N_CLS))

    def forward(self, x):
        return self.f(x)


ARCHS = {"CNN": CNN, "RNN": RNN, "LSTM": LSTM, "MLP": MLP}


def train(name, Xtr, ytr, Xte, yte, epochs=12, bs=128, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    net = ARCHS[name]()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    t0 = time.perf_counter()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xtr_t[b]), ytr_t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
    train_s = time.perf_counter() - t0

    net.eval()
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    with torch.no_grad():
        t1 = time.perf_counter()
        logits = torch.cat([net(Xte_t[i:i + 512]) for i in range(0, len(Xte_t), 512)])
        infer_ms = (time.perf_counter() - t1) * 1000.0 / len(Xte)
        pred = logits.argmax(1).numpy()
    acc = float((pred == yte).mean())
    n_par = sum(p.numel() for p in net.parameters())
    return dict(pred=pred, accuracy=acc, infer_ms=infer_ms,
                train_s=train_s, params=int(n_par))


def load(scratch):
    """the cached subject-wise partition written by get_har.py"""
    import os
    Xtr = np.load(os.path.join(scratch, "har_Xtr.npy"))
    ytr = np.load(os.path.join(scratch, "har_ytr.npy"))
    Xte = np.load(os.path.join(scratch, "har_Xte.npy"))
    yte = np.load(os.path.join(scratch, "har_yte.npy"))
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    import os
    scratch = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Xtr, ytr, Xte, yte = load(scratch)
    print("train %s  test %s  (subject-wise split)" % (Xtr.shape, Xte.shape))
    print("\n%-6s %10s %10s %10s %12s" % ("model", "accuracy", "params",
                                          "train s", "infer ms"))
    for n in ("CNN", "RNN", "LSTM", "MLP"):
        b = train(n, Xtr, ytr, Xte, yte, seed=0)
        print("%-6s %9.2f%% %10d %10.1f %12.4f"
              % (n, 100 * b["accuracy"], b["params"], b["train_s"], b["infer_ms"]))
