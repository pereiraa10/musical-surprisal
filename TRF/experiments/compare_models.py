"""
compare_models.py — cross-model comparison plots from saved TRF pickles.

Ports the prototype already worked out interactively in run_models.ipynb
(permutation_test / plot / plot_across_models) into a script driven by
whatever pickles actually exist in a save_dir, rather than a hardcoded
subject count. Reads only results.py's pickle schema (see its module
docstring) — no dependency on any model-specific code, so this never needs
torch/mne/eelbrain.

Only 'meta' and 'r_per_channel' are kept from each pickle; Y_pred/Y_true/
weights/training_history are dropped immediately after loading. Each pickle
is ~150MB (see results.py's docstring), so holding all of them at once for a
full run (19 subjects x 2 conditions x 4 models) would be 20GB+ — this keeps
peak memory to ~1 file at a time.

Usage
-----
    python compare_models.py                                  # today's save_dir
    python compare_models.py --save-dir pickles/encoding_2026-07-06
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from statsmodels.stats.multitest import fdrcorrection

from config import load_config

CONDITIONS = ('acoustic', 'acoustic_and_surprisal')
CONDITION_LABELS = {
    'acoustic': 'Acoustic\n(envelope)',
    'acoustic_and_surprisal': 'Full\n(+ onsets + surprisal + entropy)',
}


def discover_results(save_dir, models=None):
    """Load every `{subject}__{model_tag}__{condition}.pkl` in save_dir into a
    tidy DataFrame: one row per (subject, model_tag, condition), with
    `r_per_channel` (ndarray) and `channel_names` (list) columns.

    `models`, if given, restricts to those model_tag values (e.g.
    ['sklearn_ridge', 'conv_nonlinear']); default is whatever's present.
    """
    rows = []
    for path in sorted(Path(save_dir).glob('Sub*__*__*.pkl')):
        with open(path, 'rb') as f:
            result = pickle.load(f)

        meta = result['meta']
        model_tag = meta['model_family']
        if meta.get('model_variant'):
            model_tag = f"{model_tag}_{meta['model_variant']}"
        if models is not None and model_tag not in models:
            continue

        rows.append({
            'subject': meta['subject'],
            'subject_type': meta.get('subject_type'),
            'condition': meta['condition'],
            'model_tag': model_tag,
            'channel_names': meta.get('channel_names'),
            'r_per_channel': np.asarray(result['r_per_channel']),
        })
        del result  # drop Y_pred/Y_true/weights/training_history before next file

    if not rows:
        raise ValueError(f"No result pickles found under {save_dir}")
    return pd.DataFrame(rows)


def permutation_test(a, b, n_permutations=10000, two_tailed=True, seed=42):
    """Sign-flip permutation test for paired data."""
    rng = np.random.default_rng(seed)
    diff = b - a
    observed = diff.mean()

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diff))
        perm_mean = (signs * diff).mean()
        if two_tailed:
            count += abs(perm_mean) >= abs(observed)
        else:
            count += perm_mean >= observed

    p_val = count / n_permutations
    return observed, p_val


def _sig_label(p_val):
    return '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'


def _acoustic_full_arrays(df, model_tag):
    """(r_acoustic_all, r_full_all): (n_subjects, n_channels) arrays for a
    model_tag, restricted to subjects that have both conditions."""
    sub_df = df[df['model_tag'] == model_tag]
    acoustic = sub_df[sub_df['condition'] == 'acoustic'].set_index('subject')
    full = sub_df[sub_df['condition'] == 'acoustic_and_surprisal'].set_index('subject')
    common_subjects = sorted(set(acoustic.index) & set(full.index))
    r_acoustic_all = np.array([acoustic.loc[s, 'r_per_channel'] for s in common_subjects])
    r_full_all = np.array([full.loc[s, 'r_per_channel'] for s in common_subjects])
    channel_names = acoustic.loc[common_subjects[0], 'channel_names'] if common_subjects else None
    return common_subjects, r_acoustic_all, r_full_all, channel_names


def plot_single_model(df, model_tag, save_dir):
    """Per-model acoustic-vs-full violin, per-subject delta-r bar, and
    per-channel FDR-corrected significance. Adapted from run_models.ipynb's
    `plot()`, pulling subjects/channels from `df` instead of an assumed range."""
    subjects, r_acoustic_all, r_full_all, channel_names = _acoustic_full_arrays(df, model_tag)
    if len(subjects) == 0:
        print(f"  [warn] {model_tag}: no subjects with both conditions — skipping")
        return

    n_subjects = len(subjects)
    n_channels = r_acoustic_all.shape[1]
    r_acoustic_per_sub = r_acoustic_all.mean(axis=1)
    r_full_per_sub = r_full_all.mean(axis=1)
    r_diff_per_sub = r_full_per_sub - r_acoustic_per_sub

    mean_diff, p_val = permutation_test(r_acoustic_per_sub, r_full_per_sub)
    sem_diff = r_diff_per_sub.std() / np.sqrt(n_subjects)
    sig_label = _sig_label(p_val)
    print(f"  {model_tag}: mean dr = {mean_diff:.4f} +/- {sem_diff:.4f} SEM, p = {p_val:.4f} {sig_label}")

    out_dir = Path(save_dir) / 'comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1 – group violin: acoustic vs full ──
    fig, ax = plt.subplots(figsize=(7, 6))
    data = [r_acoustic_per_sub, r_full_per_sub]
    labels = [CONDITION_LABELS['acoustic'], CONDITION_LABELS['acoustic_and_surprisal']]
    colors = ['steelblue', 'darkorange']

    parts = ax.violinplot(data, positions=[1, 2], showmedians=False, showextrema=False)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    for i in range(n_subjects):
        ax.plot([1, 2], [r_acoustic_per_sub[i], r_full_per_sub[i]],
                color='gray', alpha=0.4, linewidth=0.8)

    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.uniform(-0.04, 0.04, size=len(d))
        ax.scatter(i + jitter, d, color=color, s=40, alpha=0.8,
                   edgecolors='white', linewidths=0.5, zorder=4)

    for i, d in enumerate(data, start=1):
        mean_val = d.mean()
        ax.errorbar(i, mean_val, yerr=d.std() / np.sqrt(len(d)),
                    fmt='o', color='black', markersize=7, capsize=5, linewidth=2, zorder=5)
        ax.text(i + 0.08, mean_val, f'Mean: {mean_val:.4f}',
                color='black', fontsize=10, fontweight='bold', va='center', ha='left', zorder=6)

    y_bracket = max(r_acoustic_per_sub.max(), r_full_per_sub.max()) + 0.005
    ax.plot([1, 1, 2, 2], [y_bracket - 0.003, y_bracket, y_bracket, y_bracket - 0.003],
            color='black', linewidth=1.2)
    ax.text(1.5, y_bracket + 0.003, f'p = {p_val:.4f}  {sig_label}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Mean CV Correlation (r)', fontsize=12)
    ax.set_title(f'{model_tag}: Group-Level Encoding Correlation (N = {n_subjects} subjects)', fontsize=13)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlim(0.4, 2.6)
    plt.tight_layout()
    plt.savefig(out_dir / f'{model_tag}_group_violin.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 2 – per-subject delta-r bar ──
    fig, ax = plt.subplots(figsize=(12, 4))
    sub_labels = list(subjects)
    colors_bar = ['seagreen' if d > 0 else 'crimson' for d in r_diff_per_sub]
    ax.bar(range(n_subjects), r_diff_per_sub, color=colors_bar, alpha=0.8, edgecolor='white')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axhline(mean_diff, color='black', linewidth=1.5, linestyle='--',
               label=f'Group mean dr = {mean_diff:.4f}')
    ax.set_xticks(range(n_subjects))
    ax.set_xticklabels(sub_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('dr (full - acoustic)')
    ax.set_title(f'{model_tag}: Per-Subject Improvement from Adding Surprisal + Entropy')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'{model_tag}_delta_r_per_subject.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 3 – per-channel group delta-r + FDR-corrected significance ──
    r_diff_per_channel = (r_full_all - r_acoustic_all).mean(axis=0)
    p_per_channel = np.array([
        permutation_test(r_acoustic_all[:, ch], r_full_all[:, ch])[1]
        for ch in range(n_channels)
    ])
    _, p_fdr = fdrcorrection(p_per_channel)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    colors_ch = ['seagreen' if d > 0 else 'crimson' for d in r_diff_per_channel]
    axes[0].bar(range(n_channels), r_diff_per_channel, color=colors_ch, alpha=0.8, edgecolor='white')
    axes[0].axhline(0, color='gray', linewidth=0.8)
    axes[0].axhline(r_diff_per_channel.mean(), color='black', linestyle='--', linewidth=1.2,
                    label=f'Mean = {r_diff_per_channel.mean():.4f}')
    axes[0].set_ylabel('Mean dr across subjects')
    axes[0].set_title(f'{model_tag}: Per-Channel Group dr (full - acoustic)')
    axes[0].legend()

    neg_log_p_fdr = -np.log10(np.maximum(p_fdr, 1e-10))
    axes[1].bar(range(n_channels), neg_log_p_fdr, color='mediumpurple', alpha=0.8, edgecolor='white')
    axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=1.2, label='FDR p = 0.05')
    axes[1].axhline(-np.log10(0.01), color='orange', linestyle='--', linewidth=1.2, label='FDR p = 0.01')
    axes[1].set_ylabel('-log10(p_FDR)')
    axes[1].set_title('Per-Channel Significance of Improvement (FDR-corrected permutation test)')
    axes[1].set_xticks(range(n_channels))
    axes[1].set_xticklabels(channel_names, rotation=90, fontsize=7)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'{model_tag}_per_channel.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_model_comparison(df, save_dir, models=None):
    """Cross-model violin: acoustic vs full, side by side per model_tag.
    Adapted from run_models.ipynb's `plot_across_models()`."""
    model_tags = models if models is not None else sorted(df['model_tag'].unique())
    colors = ['steelblue', 'darkorange']
    fig, ax = plt.subplots(figsize=(max(12, 3 * len(model_tags)), 6))

    spacing = 3
    centers = []
    legend_handles = [
        Patch(facecolor=colors[0], edgecolor='none', label=CONDITION_LABELS['acoustic']),
        Patch(facecolor=colors[1], edgecolor='none', label=CONDITION_LABELS['acoustic_and_surprisal']),
    ]

    plotted_tags = []
    for i, model_tag in enumerate(model_tags):
        subjects, r_acoustic_all, r_full_all, _ = _acoustic_full_arrays(df, model_tag)
        if len(subjects) == 0:
            continue
        plotted_tags.append(model_tag)

        r_acoustic_per_sub = r_acoustic_all.mean(axis=1)
        r_full_per_sub = r_full_all.mean(axis=1)

        base = len(plotted_tags[:-1]) * spacing
        pos = [base + 1, base + 2]
        centers.append((pos[0] + pos[1]) / 2)

        parts = ax.violinplot([r_acoustic_per_sub, r_full_per_sub],
                               positions=pos, showmedians=False, showextrema=False)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        for j in range(len(r_acoustic_per_sub)):
            ax.plot([pos[0], pos[1]], [r_acoustic_per_sub[j], r_full_per_sub[j]],
                    color='gray', alpha=0.4, linewidth=0.7)

        jitter = np.random.uniform(-0.04, 0.04, size=len(r_acoustic_per_sub))
        ax.scatter(np.full(len(r_acoustic_per_sub), pos[0]) + jitter, r_acoustic_per_sub,
                   color=colors[0], edgecolors='white', linewidths=0.5, s=40, alpha=0.8, zorder=4)
        jitter = np.random.uniform(-0.04, 0.04, size=len(r_full_per_sub))
        ax.scatter(np.full(len(r_full_per_sub), pos[1]) + jitter, r_full_per_sub,
                   color=colors[1], edgecolors='white', linewidths=0.5, s=40, alpha=0.8, zorder=4)

        for k, d in enumerate([r_acoustic_per_sub, r_full_per_sub]):
            mean_val = d.mean()
            sem = d.std() / np.sqrt(len(d))
            ax.errorbar(pos[k], mean_val, yerr=sem, fmt='o', color='black',
                        markersize=6, capsize=4, linewidth=1.5, zorder=5)

        try:
            _, p_val = permutation_test(r_acoustic_per_sub, r_full_per_sub)
            sig_label = _sig_label(p_val)
            y_bracket = max(r_acoustic_per_sub.max(), r_full_per_sub.max()) + 0.005
            ax.plot([pos[0], pos[0], pos[1], pos[1]],
                    [y_bracket - 0.003, y_bracket, y_bracket, y_bracket - 0.003],
                    color='black', linewidth=1.0)
            ax.text((pos[0] + pos[1]) / 2, y_bracket + 0.003, f'p={p_val:.3f} {sig_label}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        except Exception:
            pass

    if not plotted_tags:
        print("  [warn] plot_model_comparison: no model_tag had usable data — skipping")
        plt.close(fig)
        return

    ax.set_xticks(centers)
    ax.set_xticklabels(plotted_tags, fontsize=11)
    ax.set_xlim(-0.5, spacing * len(plotted_tags) - 0.5)
    ax.set_ylabel('Mean CV Correlation (r)', fontsize=12)
    ax.set_title('Comparison across models', fontsize=13)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend(handles=legend_handles, loc='upper left')
    plt.tight_layout()

    out_dir = Path(save_dir) / 'comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_summary_table(df):
    """Mean r (averaged over channels and subjects) per model_tag x condition."""
    summary = (
        df.assign(mean_r=df['r_per_channel'].apply(np.mean))
          .groupby(['model_tag', 'condition'])['mean_r']
          .agg(['mean', 'std', 'count'])
          .round(4)
    )
    print("\nSummary (mean r across channels, then across subjects):")
    print(summary.to_string())


def _build_parser():
    p = argparse.ArgumentParser(description="Compare saved TRF results across models.")
    p.add_argument("--config", default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--models", default=None,
                   help="Comma-separated model_tag subset (e.g. sklearn_ridge,conv_nonlinear). "
                        "Default: whatever's present in save_dir.")
    return p


def main(save_dir=None, models=None):
    if save_dir is None or models is None:
        args, _ = _build_parser().parse_known_args(sys.argv[1:])
        if save_dir is None:
            config = load_config(cli_args=sys.argv[1:])
            save_dir = Path(args.save_dir) if args.save_dir else config.save_dir
        if models is None:
            models = args.models.split(',') if args.models else None

    save_dir = Path(save_dir)
    print(f"Comparing results in {save_dir}")
    df = discover_results(save_dir, models=models)

    print_summary_table(df)

    for model_tag in sorted(df['model_tag'].unique()):
        plot_single_model(df, model_tag, save_dir)
    plot_model_comparison(df, save_dir, models=models)

    print(f"\nComparison plots saved to {save_dir / 'comparison'}")


if __name__ == '__main__':
    main()
