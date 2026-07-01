"""
GFNN Events Table — Time-series correlation chart
==================================================
Generates a multi-signal plot of:
  - Acoustic envelope
  - Onset regressor  (clipped positive derivative of envelope)
  - GFNN amplitude   (simplified single-oscillator integration)
  - EEG              (simulated neural response with lag + noise)

Usage
-----
  python gfnn_events_chart.py

Optional CLI arguments:
  --trial   INT   Trial index 0-29         (default: 0)
  --osc     STR   Oscillator key           (default: damped_dlc_weak)
  --lag     INT   Neural lag in samples    (default: 8, i.e. ~80 ms at 100 Hz)
  --seed    INT   RNG seed                 (default: 42)
  --out     STR   Output file path         (default: gfnn_chart.png)

Oscillator keys
---------------
  damped_dlc_weak  |  damped_dlc_intermediate  |  critical
  limit_cycle_weak |  limit_cycle_strong
  double_lc_weak   |  double_lc_intermediate   |  double_lc_strong
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import resample
from matplotlib.lines import Line2D

# ── Oscillator parameter presets ──────────────────────────────────────────────
OSC_PARAMS = {
    "damped_dlc_weak":         dict(alpha=-1.0, beta1=2.5,   beta2=-1.0, eps=1.0, F=0.10),
    "damped_dlc_intermediate": dict(alpha=-1.0, beta1=2.5,   beta2=-1.0, eps=1.0, F=0.20),
    "critical":                dict(alpha=0.0,  beta1=-100.0, beta2=0.0, eps=1.0, F=0.20),
    "limit_cycle_weak":        dict(alpha=1.0,  beta1=-100.0, beta2=0.0, eps=1.0, F=0.02),
    "limit_cycle_strong":      dict(alpha=1.0,  beta1=-100.0, beta2=0.0, eps=1.0, F=0.20),
    "double_lc_weak":          dict(alpha=-1.0, beta1=4.0,   beta2=-1.0, eps=1.0, F=0.10),
    "double_lc_intermediate":  dict(alpha=-1.0, beta1=4.0,   beta2=-1.0, eps=1.0, F=0.30),
    "double_lc_strong":        dict(alpha=-1.0, beta1=4.0,   beta2=-1.0, eps=1.0, F=1.50),
}

FS = 100          # sampling rate (Hz)
SONG_DURS = [4, 5, 6, 4.5, 5.5, 7, 4, 5, 6, 4.5]   # rough trial durations (s)

# ── Signal generators ─────────────────────────────────────────────────────────

def _lcg(seed: int):
    """Simple linear-congruential generator (float in [0,1))."""
    s = int(seed) & 0xFFFFFFFF
    while True:
        s = (s * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        yield s / 4_294_967_296


def gen_envelope(N: int, seed: int) -> np.ndarray:
    """Synthetic acoustic envelope: sum of raised-cosine bursts, smoothed."""
    rng = _lcg(seed)
    env = np.zeros(N, dtype=np.float32)
    n_segs = 4 + int(next(rng) * 4)
    for _ in range(n_segs):
        onset = int(next(rng) * N * 0.7)
        dur   = int(N * 0.05 + next(rng) * N * 0.18)
        amp   = 0.2 + next(rng) * 0.7
        for i in range(onset, min(onset + dur, N)):
            t = (i - onset) / max(dur, 1)
            env[i] += amp * np.sin(np.pi * t) * np.exp(-t * 2.5)
    mx = env.max()
    if mx > 0:
        env /= mx
    # Smooth with a box filter
    W = 8
    smooth = np.convolve(env, np.ones(2 * W + 1) / (2 * W + 1), mode="same")
    return smooth.astype(np.float32)


def gen_onsets(env: np.ndarray) -> np.ndarray:
    """Positive-only first-difference of envelope, normalised."""
    diff = np.clip(np.diff(env, prepend=env[0]), 0, None)
    mx = diff.max()
    return (diff / mx).astype(np.float32) if mx > 0 else diff


def integrate_gfnn(env: np.ndarray, osc_key: str, base_freq: float = 1.5) -> np.ndarray:
    """
    Lightweight single-frequency GFNN integration (RK4, Hopf-style NL).
    Returns normalised amplitude |z|.
    """
    p = OSC_PARAMS[osc_key]
    alpha, beta1, beta2, eps, F = p["alpha"], p["beta1"], p["beta2"], p["eps"], p["F"]
    N  = len(env)
    dt = 1.0 / FS

    def nl(z):
        z2 = abs(z) ** 2
        z4 = z2 ** 2
        d  = 1.0 - eps * z2
        d  = d if abs(d) > 1e-9 else (1e-9 if d >= 0 else -1e-9)
        return (alpha + 1j * 2 * np.pi * base_freq
                + beta1 * z2 + (eps * beta2 * z4) / d)

    def rhs(z, x):
        return base_freq * (z * nl(z) + x)

    def rk4(z, xn, xn1):
        xh = 0.5 * (xn + xn1)
        k1 = rhs(z,              xn)
        k2 = rhs(z + 0.5*dt*k1, xh)
        k3 = rhs(z + 0.5*dt*k2, xh)
        k4 = rhs(z +     dt*k3, xn1)
        return z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    amp = np.zeros(N, dtype=np.float32)
    z   = 1e-6 + 1j * 1e-6
    amp[0] = abs(z)
    for n in range(N - 1):
        z = rk4(z, float(F * env[n]), float(F * env[n + 1]))
        amp[n + 1] = abs(z)

    mx = amp.max()
    return (amp / mx).astype(np.float32) if mx > 0 else amp


def gen_eeg(env: np.ndarray, gfnn: np.ndarray, seed: int,
            lag: int = 8) -> np.ndarray:
    """Simulate filtered EEG: lagged mixture of env+gfnn + coloured noise."""
    rng  = _lcg(seed + 9999)
    N    = len(env)
    eeg  = np.zeros(N, dtype=np.float32)
    for i in range(lag, N):
        noise   = (next(rng) - 0.5) * 0.4
        eeg[i]  = 0.25 * env[i - lag] + 0.35 * gfnn[i - lag] + noise
    W      = 5
    smooth = np.convolve(eeg, np.ones(2 * W + 1) / (2 * W + 1), mode="same")
    mu, sd = smooth.mean(), smooth.std()
    return ((smooth - mu) / (sd if sd > 0 else 1) * 1.5).astype(np.float32)


def _norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else arr - lo


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "envelope": "#378ADD",
    "onsets":   "#1D9E75",
    "gfnn":     "#D85A30",
    "eeg":      "#7F77DD",
}

LABELS = {
    "envelope": "Envelope",
    "onsets":   "Onsets",
    "gfnn":     "GFNN amplitude",
    "eeg":      "EEG (normalised)",
}


def make_chart(trial: int = 0,
               osc_key: str = "damped_dlc_weak",
               lag: int = 8,
               seed_offset: int = 42,
               out: str = "gfnn_chart.png") -> None:

    # ── Generate signals ──────────────────────────────────────────────────────
    song_id = trial % 10
    N       = round(SONG_DURS[song_id] * FS)
    seed    = trial * 17 + song_id * 3 + seed_offset

    env    = gen_envelope(N, seed)
    onsets = gen_onsets(env)
    gfnn   = integrate_gfnn(env, osc_key)
    eeg    = gen_eeg(env, gfnn, seed, lag=lag)

    env_n  = _norm(env)
    ons_n  = _norm(onsets)
    gfnn_n = _norm(gfnn)
    eeg_n  = _norm(eeg)

    t = np.arange(N) / FS

    r_eg  = pearson(env_n,  gfnn_n)
    r_ge  = pearson(gfnn_n, eeg_n)
    r_ee  = pearson(env_n,  eeg_n)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 6.5), facecolor="#ffffff")
    gs  = gridspec.GridSpec(
        2, 1,
        height_ratios=[5, 1],
        hspace=0.35,
        left=0.07, right=0.97, top=0.88, bottom=0.10
    )

    ax  = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1])

    # ── Main time-series ──────────────────────────────────────────────────────
    ax.plot(t, env_n,  color=COLORS["envelope"], lw=1.8, label="Envelope",       zorder=4)
    ax.plot(t, ons_n,  color=COLORS["onsets"],   lw=1.2, label="Onsets",
            linestyle=(0, (4, 4)), zorder=3)
    ax.plot(t, gfnn_n, color=COLORS["gfnn"],     lw=1.8, label="GFNN amplitude", zorder=4)
    ax.plot(t, eeg_n,  color=COLORS["eeg"],      lw=1.5, label="EEG (normalised)", zorder=3)

    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.15, 1.2)
    ax.set_xlabel("Time (s)", fontsize=12, color="#444444")
    ax.set_ylabel("Normalised amplitude", fontsize=12, color="#444444")
    ax.tick_params(labelsize=10, color="#aaaaaa")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.grid(axis="both", color="#eeeeee", linewidth=0.7, zorder=0)

    osc_label = osc_key.replace("_", " ").title()
    fig.suptitle(
        f"Trial {trial + 1}  ·  {osc_label}",
        fontsize=14, fontweight="500", color="#222222", y=0.97
    )

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color=COLORS["envelope"], lw=2,   label="Envelope"),
        Line2D([0], [0], color=COLORS["onsets"],   lw=1.5, label="Onsets",
               linestyle=(0, (4, 4))),
        Line2D([0], [0], color=COLORS["gfnn"],     lw=2,   label="GFNN amplitude"),
        Line2D([0], [0], color=COLORS["eeg"],      lw=1.5, label="EEG (normalised)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
        edgecolor="#dddddd",
        fancybox=False,
    )

    # ── Correlation metric bar ────────────────────────────────────────────────
    axm.set_xlim(0, 1)
    axm.set_ylim(0, 1)
    axm.axis("off")

    metrics = [
        ("Env–GFNN  r", r_eg,  COLORS["gfnn"]),
        ("GFNN–EEG  r", r_ge,  COLORS["eeg"]),
        ("Env–EEG   r", r_ee,  COLORS["envelope"]),
    ]
    xpos = [0.05, 0.38, 0.71]
    for (lbl, val, col), x in zip(metrics, xpos):
        axm.add_patch(plt.Rectangle(
            (x, 0.05), 0.27, 0.90,
            transform=axm.transAxes,
            fc="#f7f7f7", ec="#dddddd", lw=0.7, zorder=1, clip_on=False
        ))
        axm.text(x + 0.135, 0.74, lbl,
                 ha="center", va="center", fontsize=9, color="#888888",
                 transform=axm.transAxes)
        axm.text(x + 0.135, 0.32, f"{val:+.3f}",
                 ha="center", va="center", fontsize=16, fontweight="500", color=col,
                 transform=axm.transAxes)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.show()


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Plot GFNN events-table signals for a single trial."
    )
    p.add_argument("--trial", type=int,   default=0,                    help="Trial index 0–29")
    p.add_argument("--osc",   type=str,   default="damped_dlc_weak",    help="Oscillator key")
    p.add_argument("--lag",   type=int,   default=8,                    help="EEG lag (samples)")
    p.add_argument("--seed",  type=int,   default=42,                   help="RNG seed offset")
    p.add_argument("--out",   type=str,   default="gfnn_chart.png",     help="Output file")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.osc not in OSC_PARAMS:
        raise ValueError(
            f"Unknown oscillator key '{args.osc}'.\n"
            f"Available: {', '.join(OSC_PARAMS)}"
        )
    make_chart(
        trial=args.trial,
        osc_key=args.osc,
        lag=args.lag,
        seed_offset=args.seed,
        out=args.out,
    )