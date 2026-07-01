"""
diagnostic_lag_alignment.py  —  Diagnostic D1 (see TRF_conv_DIAGNOSTICS.md)
────────────────────────────────────────────────────────────────────────────────
Settles hypothesis H1: do the linear Conv1d (TRF_conv_1.py) and the ridge Toeplitz
lag matrix (TRF_ridge_3.py) recover the SAME kernel from the SAME input with a known
impulse-response latency?

This is pure numpy + torch on synthetic data — no EEG, no eelbrain. It can run
anywhere torch is installed. If the conv kernel peak is shifted or flipped relative
to ridge's, H1 is confirmed and the padding/orientation in TRF_conv_1.py must be
fixed until this test passes BEFORE trusting any result on real data.

Run:
    python diagnostic_lag_alignment.py
"""

import numpy as np
import torch
import torch.nn as nn

# Mirror the exact TRF config (kept local so this file is standalone) ───────────
SFREQ = 64
TMIN, TMAX = -0.1, 0.6
N_LAGS  = int(round((TMAX - TMIN) * SFREQ)) + 1   # 46
LAG_MIN = int(round(TMIN * SFREQ))                # -6
LAG_MAX = LAG_MIN + N_LAGS - 1                    # 39

RIDGE_ALPHA = 1.0          # light reg; this is a fitting test, not a tuning test
TRUE_LATENCY_MS = 150.0    # where we inject the synthetic response peak
N_SAMPLES = 20000


# ─── Ridge lag matrix (copied from TRF_ridge_3.build_lag_matrix) ───────────────
def build_lag_matrix(x, tmin, tmax, sfreq):
    n_lags  = int(round((tmax - tmin) * sfreq)) + 1
    lag_min = int(round(tmin * sfreq))
    lag_max = lag_min + n_lags - 1
    n = len(x)
    x_pad = np.concatenate([np.zeros(lag_max), x, np.zeros(max(0, -lag_min))])
    wins  = np.lib.stride_tricks.sliding_window_view(x_pad, n_lags)
    return np.ascontiguousarray(wins[:n, ::-1])


# ─── Conv model (matches the 'linear' variant in TRF_conv_1.py) ────────────────
class CausalPad(nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.left, self.right = left, right
    def forward(self, x):
        return nn.functional.pad(x, (self.left, self.right))


def make_linear_conv(n_features=1, n_channels=1):
    # NOTE: bias=False here (unlike TRF_conv_1) to make the equivalence test clean.
    return nn.Sequential(
        CausalPad(LAG_MAX, max(0, -LAG_MIN)),
        nn.Conv1d(n_features, n_channels, kernel_size=N_LAGS, bias=False),
    )


def lag_axis_ms():
    """Lag value (ms) for each of the N_LAGS taps, on the ridge convention
    (column 0 → lag_min, column N_LAGS-1 → lag_max)."""
    lags = np.arange(LAG_MIN, LAG_MAX + 1)
    return lags / SFREQ * 1000.0


def main():
    rng = np.random.default_rng(0)

    # 1) Synthetic stimulus + a known TRF kernel (Gaussian bump at TRUE_LATENCY_MS)
    x = rng.standard_normal(N_SAMPLES)
    lags_ms = lag_axis_ms()
    true_kernel = np.exp(-0.5 * ((lags_ms - TRUE_LATENCY_MS) / 30.0) ** 2)

    # Synthetic EEG: y[t] = sum_lag kernel[lag] * x[t - lag]  (ridge convention)
    Phi = build_lag_matrix(x, TMIN, TMAX, SFREQ)        # (N, 46)
    y = Phi @ true_kernel
    y += 0.01 * rng.standard_normal(N_SAMPLES)          # tiny noise

    # 2) Ridge closed-form recovery
    p = Phi.shape[1]
    w_ridge = np.linalg.solve(Phi.T @ Phi + RIDGE_ALPHA * np.eye(p), Phi.T @ y)
    ridge_peak_ms = lags_ms[int(np.argmax(np.abs(w_ridge)))]

    # 3) Conv recovery (fit to convergence, no reg, no early stop)
    model = make_linear_conv()
    xt = torch.from_numpy(x[None, None].astype(np.float32))     # (1,1,N)
    yt = torch.from_numpy(y[None, None].astype(np.float32))     # (1,1,N)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3000):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xt), yt)
        loss.backward()
        opt.step()

    conv_w = model[1].weight.detach().numpy().reshape(-1)       # (46,)
    # PyTorch conv is cross-correlation: output[t] = sum_k w[k]*x_pad[t+k].
    # With left pad = LAG_MAX, tap k corresponds to stimulus lag (LAG_MAX - k).
    # Map conv taps onto the SAME lag axis as ridge to compare honestly.
    conv_tap_lag_ms = (LAG_MAX - np.arange(N_LAGS)) / SFREQ * 1000.0
    conv_peak_ms = conv_tap_lag_ms[int(np.argmax(np.abs(conv_w)))]

    # 4) Report
    print("─" * 70)
    print(f"Injected latency:        {TRUE_LATENCY_MS:.1f} ms")
    print(f"Ridge recovered peak:    {ridge_peak_ms:.1f} ms")
    print(f"Conv  recovered peak:    {conv_peak_ms:.1f} ms")
    print("─" * 70)

    ridge_ok = abs(ridge_peak_ms - TRUE_LATENCY_MS) <= 1000 / SFREQ + 1e-6
    conv_ok  = abs(conv_peak_ms  - TRUE_LATENCY_MS) <= 1000 / SFREQ + 1e-6
    agree    = abs(ridge_peak_ms - conv_peak_ms)    <= 1000 / SFREQ + 1e-6

    print(f"Ridge recovers latency:  {'PASS' if ridge_ok else 'FAIL'}")
    print(f"Conv  recovers latency:  {'PASS' if conv_ok else 'FAIL'}")
    print(f"Conv == Ridge alignment: {'PASS' if agree else 'FAIL  ← H1 CONFIRMED'}")

    if not agree:
        shift = conv_peak_ms - ridge_peak_ms
        print(f"\n  Conv peak is shifted {shift:+.1f} ms vs ridge.")
        print("  → Fix CausalPad / kernel orientation in TRF_conv_1.py so the")
        print("    tap→lag mapping matches build_lag_matrix's reversed-column")
        print("    convention, then re-run until this line PASSES.")
    else:
        print("\n  Conv and ridge agree on lag alignment. H1 is NOT the cause;")
        print("  proceed to D2 (shuffle/null test) and D3 (match regularization).")


if __name__ == "__main__":
    main()
