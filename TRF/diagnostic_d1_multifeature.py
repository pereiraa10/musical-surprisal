"""
diagnostic_d1_multifeature.py  —  Diagnostics D1 (extended) + D3 (synthetic)
────────────────────────────────────────────────────────────────────────────────
Addresses two open hypotheses from TRF_conv_DIAGNOSTICS.md:

  D1 extended (H1):  Prior D1 run was only 1 feature / 1 channel.  This extends
    to 6 features (matching the acoustic_and_surprisal condition) across multiple
    channels with a single long trial and matched light regularisation.
    Pass criterion: ridge and conv recover the same kernels (correlation > 0.99
    per feature/channel pair, peaks within 1 sample of each other).

  D3 synthetic (H2):  Does regularisation mismatch alone explain the ~3× r gap
    on Sub2?  Sweeps ridge alpha over the SAME logspace(1,7,25) range used by
    TRF_ridge_3.py, runs true LOOCV matching TRF_conv_1.py's protocol (inner
    val split, early stopping), and prints a ridge-r vs conv-r table.
    Key question: at what alpha does ridge r drop to the level seen on Sub2?

Run anywhere:
    python diagnostic_d1_multifeature.py
Requires:  numpy, torch  (no EEG data, no eelbrain)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

# ── Mirror TRF constants exactly ─────────────────────────────────────────────
SFREQ  = 64
TMIN   = -0.1
TMAX   = 0.600
N_LAGS  = int(round((TMAX - TMIN) * SFREQ)) + 1   # 46
LAG_MIN = int(round(TMIN * SFREQ))                 # -6
LAG_MAX = LAG_MIN + N_LAGS - 1                     # 39

# Exact hyperparameters from TRF_conv_1.py
WEIGHT_DECAY       = 1e-3
EPOCHS             = 50
LR                 = 1e-3
EARLY_STOP_PATIENCE = 25
RIDGE_ALPHAS        = np.logspace(1, 7, 25)   # from TRF_ridge_3.py

# Synthetic-data knobs
N_FEATURES  = 6     # acoustic (2) + surprisal (4)
N_CHANNELS  = 8     # reduced for speed; 64 on real data
N_TRIALS    = 8     # typical per-subject trial count
TRIAL_LEN   = 4000  # ~62 s at 64 Hz
SNR_LINEAR  = 0.05  # signal-fraction SNR: y = alpha*signal + sqrt(1-alpha²)*noise
                    # chosen so r_max ≈ 0.05, mimicking Sub2 ridge numbers
RNG_SEED    = 42


# ── Ridge lag matrix — verbatim from TRF_ridge_3.py ──────────────────────────
def build_lag_matrix(x, tmin, tmax, sfreq):
    n_lags  = int(round((tmax - tmin) * sfreq)) + 1
    lag_min = int(round(tmin * sfreq))
    lag_max = lag_min + n_lags - 1
    n = len(x)
    x_pad = np.concatenate([np.zeros(lag_max), x, np.zeros(max(0, -lag_min))])
    wins  = np.lib.stride_tricks.sliding_window_view(x_pad, n_lags)
    return np.ascontiguousarray(wins[:n, ::-1])


def build_design_matrix(X_feat, tmin, tmax, sfreq):
    """X_feat: (T, n_features) → (T, n_features * N_LAGS) Toeplitz stack."""
    return np.hstack([build_lag_matrix(X_feat[:, f], tmin, tmax, sfreq)
                      for f in range(X_feat.shape[1])])


def zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)


# ── Conv model — verbatim from TRF_conv_1.py (linear variant) ────────────────
class CausalPad(nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.left, self.right = left, right

    def forward(self, x):
        return nn.functional.pad(x, (self.left, self.right))


def make_linear_conv(n_features, n_channels, bias=True):
    pad_left  = LAG_MAX
    pad_right = max(0, -LAG_MIN)
    return nn.Sequential(
        CausalPad(pad_left, pad_right),
        nn.Conv1d(n_features, n_channels, kernel_size=N_LAGS, bias=bias),
    )


def _to_tensor(arr_2d):
    """(T, C) numpy → (1, C, T) float32 tensor."""
    return torch.from_numpy(
        np.ascontiguousarray(arr_2d.T[None].astype(np.float32)))


# ── Training (matches TRF_conv_1.train_one_fold exactly) ─────────────────────
def train_conv(X_tr, Y_tr, n_features, n_channels, weight_decay=WEIGHT_DECAY):
    model   = make_linear_conv(n_features, n_channels)
    opt     = torch.optim.Adam(model.parameters(), lr=LR,
                               weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    inner_val_idx = len(X_tr) - 1
    tr_idx        = list(range(inner_val_idx))
    Xv = _to_tensor(X_tr[inner_val_idx])
    Yv = _to_tensor(Y_tr[inner_val_idx])

    best_val, best_state, since = np.inf, None, 0
    for _ in range(EPOCHS):
        model.train()
        for i in tr_idx:
            opt.zero_grad()
            loss = loss_fn(model(_to_tensor(X_tr[i])), _to_tensor(Y_tr[i]))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv), Yv).item()
        if v < best_val - 1e-6:
            best_val, best_state, since = v, \
                {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= EARLY_STOP_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def loocv_conv(X_all, Y_all, weight_decay=WEIGHT_DECAY):
    n_features = X_all[0].shape[1]
    n_channels = Y_all[0].shape[1]
    Y_pred_all, Y_true_all = [], []
    for i in range(len(X_all)):
        X_tr = [X_all[j] for j in range(len(X_all)) if j != i]
        Y_tr = [Y_all[j] for j in range(len(Y_all)) if j != i]
        model = train_conv(X_tr, Y_tr, n_features, n_channels, weight_decay)
        model.eval()
        with torch.no_grad():
            pred = model(_to_tensor(X_all[i])).numpy()[0].T  # (T, n_ch)
        Y_pred_all.append(pred)
        Y_true_all.append(Y_all[i])
    return np.concatenate(Y_pred_all), np.concatenate(Y_true_all)


# ── Ridge LOOCV (verbatim logic from TRF_ridge_3.py) ─────────────────────────
def select_alpha_loocv(Phi_all, Y_all, alphas):
    Phi_full = np.concatenate(Phi_all)
    Y_full   = np.concatenate(Y_all)
    p        = Phi_full.shape[1]
    XTX      = Phi_full.T @ Phi_full
    XTY      = Phi_full.T @ Y_full
    best_alpha, best_r = alphas[0], -np.inf
    for alpha in alphas:
        alpha_I = alpha * np.eye(p)
        r_folds = []
        for Phi_i, Y_i in zip(Phi_all, Y_all):
            XTX_tr = XTX - Phi_i.T @ Phi_i
            XTY_tr = XTY - Phi_i.T @ Y_i
            W      = np.linalg.solve(XTX_tr + alpha_I, XTY_tr)
            r_fold = np.mean([pearsonr(Y_i[:, c], (Phi_i @ W)[:, c])[0]
                              for c in range(Y_i.shape[1])])
            r_folds.append(r_fold)
        avg_r = np.mean(r_folds)
        if avg_r > best_r:
            best_r, best_alpha = avg_r, alpha
    return best_alpha


def loocv_ridge(Phi_all, Y_all, alpha):
    Phi_full = np.concatenate(Phi_all)
    Y_full   = np.concatenate(Y_all)
    p        = Phi_full.shape[1]
    XTX      = Phi_full.T @ Phi_full
    XTY      = Phi_full.T @ Y_full
    alpha_I  = alpha * np.eye(p)
    Y_pred   = np.zeros_like(Y_full)
    offset   = 0
    for Phi_i, Y_i in zip(Phi_all, Y_all):
        n_i       = len(Phi_i)
        W         = np.linalg.solve(XTX - Phi_i.T @ Phi_i + alpha_I,
                                    XTY - Phi_i.T @ Y_i)
        Y_pred[offset: offset + n_i] = Phi_i @ W
        offset += n_i
    return Y_pred, Y_full


def mean_r(Y_true, Y_pred):
    n = Y_true.shape[1]
    return np.mean([pearsonr(Y_true[:, c], Y_pred[:, c])[0] for c in range(n)])


# ── Synthetic data generator ──────────────────────────────────────────────────
def make_synthetic_data(rng, n_trials, trial_len, n_features, n_channels,
                        snr_linear, true_kernels):
    """
    Returns X_trials, Y_trials (each a list of (T, n_features/n_channels) arrays),
    per-trial z-scored exactly as TRF_ridge_3 / TRF_conv_1 do.

    true_kernels: (n_channels, n_features, N_LAGS) — the ground-truth TRF.
    y[t] = sum_f sum_l W[ch,f,l] * x[t - lag_l]  +  noise, where lag_l is
    the ridge column convention (col 0 → lag_min, col N_LAGS-1 → lag_max).
    """
    X_trials, Y_trials = [], []
    for _ in range(n_trials):
        X = rng.standard_normal((trial_len, n_features))
        # Build lag design for each feature, stack
        Phi = build_design_matrix(X, TMIN, TMAX, SFREQ)  # (T, n_feat*N_LAGS)
        # Reshape true_kernels to (n_features*N_LAGS, n_channels) for matmul
        W_flat = true_kernels.reshape(n_channels, n_features * N_LAGS).T
        signal = Phi @ W_flat                             # (T, n_channels)
        sig_std  = signal.std(axis=0, keepdims=True) + 1e-12
        noise    = rng.standard_normal(signal.shape)
        # y = snr * (signal/sig_std) + sqrt(1-snr²) * noise  →  unit variance
        Y = snr_linear * (signal / sig_std) + np.sqrt(1 - snr_linear**2) * noise
        X_trials.append(zscore(X))
        Y_trials.append(zscore(Y))
    return X_trials, Y_trials


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — D1 extended: kernel recovery on a single long trial
# ══════════════════════════════════════════════════════════════════════════════

def run_d1_extended(rng):
    print("=" * 70)
    print("PART 1 — D1 extended: multi-feature kernel recovery")
    print("  6 features × 8 channels, 20 000-sample trial, alpha=1.0 (light reg)")
    print("=" * 70)

    lag_axis_ms = np.arange(LAG_MIN, LAG_MAX + 1) / SFREQ * 1000.0

    # True kernels: each feature gets a Gaussian bump at a different latency
    latencies_ms = np.array([100, 150, 200, 250, 100, 150])  # one per feature
    true_kernels = np.zeros((N_CHANNELS, N_FEATURES, N_LAGS))
    for f, lat in enumerate(latencies_ms):
        bump = np.exp(-0.5 * ((lag_axis_ms - lat) / 30.0) ** 2)
        for ch in range(N_CHANNELS):
            # slight per-channel gain variation
            true_kernels[ch, f] = bump * (1.0 + 0.1 * ch / N_CHANNELS)

    T = 20000
    X = rng.standard_normal((T, N_FEATURES))
    Phi = build_design_matrix(X, TMIN, TMAX, SFREQ)
    W_flat   = true_kernels.reshape(N_CHANNELS, N_FEATURES * N_LAGS).T
    signal   = Phi @ W_flat
    noise    = rng.standard_normal(signal.shape)
    snr      = 0.99
    Y = snr * (signal / (signal.std(axis=0) + 1e-12)) + np.sqrt(1 - snr**2) * noise

    X_z = zscore(X)
    Y_z = zscore(Y)

    # Ridge (single fit, alpha=1.0 — matched light reg, same as original D1)
    ALPHA = 1.0
    Phi_z  = build_design_matrix(X_z, TMIN, TMAX, SFREQ)
    p      = Phi_z.shape[1]
    W_ridge = np.linalg.solve(Phi_z.T @ Phi_z + ALPHA * np.eye(p), Phi_z.T @ Y_z)
    # W_ridge: (n_feat*N_LAGS, n_channels) — each col is one channel's TRF vector

    # Conv (single fit, no weight decay, train to convergence)
    model = make_linear_conv(N_FEATURES, N_CHANNELS, bias=False)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
    xt = _to_tensor(X_z)   # (1, F, T)
    yt = _to_tensor(Y_z)   # (1, C, T)
    for step in range(5000):
        opt.zero_grad()
        nn.functional.mse_loss(model(xt), yt).backward()
        opt.step()

    # Conv weights: model[1].weight shape = (n_channels, n_features, N_LAGS)
    conv_w = model[1].weight.detach().numpy()   # (n_ch, n_feat, N_LAGS)
    # Ridge weights reshaped: (n_feat, N_LAGS, n_ch)
    ridge_w = W_ridge.reshape(N_FEATURES, N_LAGS, N_CHANNELS)

    # Compare per-feature, per-channel: does each pair correlate?
    # Ridge convention: col j → lag (lag_min + j); conv tap k → lag (LAG_MAX - k).
    # Both recover the same function; we just need to correlate the kernel vectors
    # after mapping to the same lag axis.  Reverse conv tap axis to match ridge.
    conv_w_ridge_order = conv_w[:, :, ::-1]  # (n_ch, n_feat, N_LAGS) reversed

    corrs = np.zeros((N_CHANNELS, N_FEATURES))
    peak_errs_ms = np.zeros((N_CHANNELS, N_FEATURES))
    for ch in range(N_CHANNELS):
        for f in range(N_FEATURES):
            r_kern = np.corrcoef(ridge_w[f, :, ch],
                                 conv_w_ridge_order[ch, f])[0, 1]
            corrs[ch, f] = r_kern
            peak_ridge = lag_axis_ms[np.argmax(np.abs(ridge_w[f, :, ch]))]
            peak_conv  = lag_axis_ms[np.argmax(np.abs(conv_w_ridge_order[ch, f]))]
            peak_errs_ms[ch, f] = abs(peak_ridge - peak_conv)

    mean_corr      = corrs.mean()
    max_peak_err   = peak_errs_ms.max()
    samp_tol_ms    = 1000.0 / SFREQ + 1e-6   # one sample = 15.625 ms

    print(f"\n  Mean kernel correlation (ridge vs conv): {mean_corr:.4f}")
    print(f"  Max peak-lag error:                      {max_peak_err:.1f} ms "
          f"(tolerance = {samp_tol_ms:.1f} ms = 1 sample)")
    pass_d1 = (mean_corr > 0.99) and (max_peak_err <= samp_tol_ms)
    print(f"\n  D1 extended: {'PASS — H1 cleared in multi-feature case' if pass_d1 else 'FAIL — H1 STILL ACTIVE, kernel mismatch present'}")

    if not pass_d1:
        worst_ch, worst_f = np.unravel_index(corrs.argmin(), corrs.shape)
        print(f"    Worst pair: ch={worst_ch}, feature={worst_f}, "
              f"corr={corrs[worst_ch, worst_f]:.4f}, "
              f"peak_err={peak_errs_ms[worst_ch, worst_f]:.1f} ms")
    return pass_d1


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — D3 synthetic: LOOCV r vs alpha (regularisation mismatch test)
# ══════════════════════════════════════════════════════════════════════════════

def run_d3_synthetic(rng):
    print()
    print("=" * 70)
    print("PART 2 — D3 synthetic: LOOCV r vs regularisation")
    print(f"  {N_TRIALS} trials × {TRIAL_LEN} samples × {N_FEATURES} features"
          f" × {N_CHANNELS} channels,  SNR ≈ {SNR_LINEAR:.2f}")
    print("  Sweeping ridge alpha over logspace(1,7,7) + LOOCV-selected alpha.")
    print("  Conv: weight_decay=1e-3, early stopping (matching TRF_conv_1.py).")
    print("=" * 70)

    lag_axis_ms  = np.arange(LAG_MIN, LAG_MAX + 1) / SFREQ * 1000.0
    latencies_ms = np.array([100, 150, 200, 250, 100, 150])
    true_kernels = np.zeros((N_CHANNELS, N_FEATURES, N_LAGS))
    for f, lat in enumerate(latencies_ms):
        bump = np.exp(-0.5 * ((lag_axis_ms - lat) / 30.0) ** 2)
        for ch in range(N_CHANNELS):
            true_kernels[ch, f] = bump * (1.0 + 0.1 * ch / N_CHANNELS)

    X_all, Y_all = make_synthetic_data(
        rng, N_TRIALS, TRIAL_LEN, N_FEATURES, N_CHANNELS, SNR_LINEAR, true_kernels)

    Phi_all = [build_design_matrix(X, TMIN, TMAX, SFREQ) for X in X_all]

    # ── LOOCV-selected ridge alpha ────────────────────────────────────────────
    print("\n  Selecting ridge alpha via LOOCV (same 25-candidate search as ridge script)...")
    best_alpha = select_alpha_loocv(Phi_all, Y_all, RIDGE_ALPHAS)
    print(f"  LOOCV-selected alpha: {best_alpha:.2e}")

    Y_pred_ridge, Y_true = loocv_ridge(Phi_all, Y_all, best_alpha)
    r_ridge_loocv = mean_r(Y_true, Y_pred_ridge)
    print(f"  Ridge (LOOCV-selected alpha={best_alpha:.2e}):  mean r = {r_ridge_loocv:.4f}")

    # ── Conv LOOCV (matching TRF_conv_1.py protocol) ─────────────────────────
    print("\n  Training conv LOOCV (this takes ~1-2 min)...")
    Y_pred_conv, Y_true_conv = loocv_conv(X_all, Y_all, weight_decay=WEIGHT_DECAY)
    r_conv = mean_r(Y_true_conv, Y_pred_conv)
    print(f"  Conv  (weight_decay={WEIGHT_DECAY:.0e}):               mean r = {r_conv:.4f}")

    # ── Ridge alpha sweep (7 evenly spaced log points) ───────────────────────
    alpha_sweep = np.logspace(1, 7, 7)
    print(f"\n  Ridge alpha sweep  (conv r = {r_conv:.4f} for reference):")
    print(f"  {'alpha':>12s}  {'ridge r':>9s}  {'conv r':>9s}  {'gap (conv-ridge)':>18s}")
    print("  " + "-" * 56)
    for alpha in alpha_sweep:
        Y_pred_a, _ = loocv_ridge(Phi_all, Y_all, alpha)
        r_a = mean_r(Y_true, Y_pred_a)
        print(f"  {alpha:12.2e}  {r_a:9.4f}  {r_conv:9.4f}  {r_conv - r_a:+18.4f}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print()
    if r_conv > r_ridge_loocv * 1.5:
        print("  [!!] Conv r still > 1.5× ridge even on synthetic data with identical")
        print("       inputs.  Regularisation mismatch alone does NOT fully explain")
        print("       the gap — re-examine H3 (inner-val split / early-stop inflation)")
        print("       or run D2 (shuffle test on real data) next.")
    elif r_conv > r_ridge_loocv * 1.1:
        print("  [~]  Conv r is 10–50% above ridge on synthetic data.  Partial H2")
        print("       contribution confirmed; also check the real-data alpha (D3-real).")
    else:
        print("  [OK] Conv and ridge match on synthetic data.  If the gap persists on")
        print("       real data, the real ridge may be over-regularised (alpha too large).")
        print("       Print the selected alpha from TRF_ridge_3.py output for Sub2 and")
        print("       compare to the sweep above.")

    return r_ridge_loocv, r_conv, best_alpha


# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    print("\n" + "=" * 70)
    print(" TRF_conv_1 Diagnostics D1 (extended) + D3 (synthetic)")
    print(f" SFREQ={SFREQ} Hz  |  TMIN={TMIN}  TMAX={TMAX}  |  N_LAGS={N_LAGS}")
    print(f" LAG_MIN={LAG_MIN}  LAG_MAX={LAG_MAX}")
    print("=" * 70)

    d1_passed = run_d1_extended(rng)
    r_ridge, r_conv, best_alpha = run_d3_synthetic(rng)

    print()
    print("=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"  D1 extended (H1 — lag alignment, multi-feature): "
          f"{'PASS' if d1_passed else 'FAIL'}")
    print(f"  D3 synthetic (H2 — reg mismatch):")
    print(f"    LOOCV-selected alpha on synthetic data: {best_alpha:.2e}")
    print(f"    Ridge r = {r_ridge:.4f}  |  Conv r = {r_conv:.4f}  "
          f"|  Ratio = {r_conv / (r_ridge + 1e-9):.2f}×")
    if d1_passed and r_conv <= r_ridge * 1.1:
        print()
        print("  Both methods agree on synthetic data.  Most likely explanation for")
        print("  the Sub2 gap: ridge selected an overly large alpha on real data.")
        print("  Next step: run TRF_ridge_3.py on Sub2 and check the printed alpha.")
    elif not d1_passed:
        print()
        print("  H1 is active — fix the CausalPad / kernel orientation in")
        print("  TRF_conv_1.py before proceeding to D2/D3 on real data.")
    else:
        print()
        print("  Gap persists on synthetic data — run diagnostic_d2_shuffle.py")
        print("  (D2 shuffle/null test) to check for a data-leak in TRF_conv_1.py.")


if __name__ == "__main__":
    main()
