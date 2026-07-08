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
full run (19 subjects x 2 feature_sets x 4 models) would be 20GB+ — this keeps
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


def _resolve_feature_sets(config):
    """Return (baseline_name, comparison_name, baseline_label, comparison_label)
    for the fixed pairwise (baseline vs. comparison) violin/comparison plots,
    driven by `config.feature_sets` / `config.feature_set_labels` instead of
    hardcoded literal strings. The first two feature_sets in config order are
    the baseline and comparison groups; a feature_set without an entry in
    `config.feature_set_labels` falls back to a title-cased default label."""
    names = list(config.feature_sets.keys())
    if len(names) < 2:
        raise ValueError(
            f"compare_models requires >= 2 feature_sets in config, got {names}")
    baseline, comparison = names[0], names[1]
    labels = config.feature_set_labels or {}

    def _label(name):
        return labels.get(name, name.replace('_', ' ').title())

    return baseline, comparison, _label(baseline), _label(comparison)


def discover_results(save_dir, models=None):
    """Load every `{subject}__{model_tag}__{feature_set}.pkl` in save_dir into a
    tidy DataFrame: one row per (subject, model_tag, feature_set), with
    `r_per_channel` (ndarray) and `channel_names` (list) columns.

    `models`, if given, restricts to those model_tag values (e.g.
    ['sklearn_ridge', 'conv_nonlinear']); default is whatever's present.
    """
    # `{subject}__{model_tag}__{feature_set}.pkl` (see results.result_filename)
    # -- glob on the two `__` separators + `.pkl` extension only, not a
    # `Sub*`-prefixed subject name, so this works for any dataset's subject
    # naming convention (liberi's 'Sub2', OpenMIIR's 'P01', Daly's 'sub-01', ...).
    rows = []
    for path in sorted(Path(save_dir).glob('*__*__*.pkl')):
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
            'feature_set': meta['feature_set'],
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


def _paired_feature_set_arrays(df, model_tag, baseline, comparison):
    """(r_baseline_all, r_comparison_all): (n_subjects, n_channels) arrays for a
    model_tag, restricted to subjects that have both `baseline` and
    `comparison` feature_sets."""
    sub_df = df[df['model_tag'] == model_tag]
    baseline_df = sub_df[sub_df['feature_set'] == baseline].set_index('subject')
    comparison_df = sub_df[sub_df['feature_set'] == comparison].set_index('subject')
    common_subjects = sorted(set(baseline_df.index) & set(comparison_df.index))
    r_baseline_all = np.array([baseline_df.loc[s, 'r_per_channel'] for s in common_subjects])
    r_comparison_all = np.array([comparison_df.loc[s, 'r_per_channel'] for s in common_subjects])
    channel_names = baseline_df.loc[common_subjects[0], 'channel_names'] if common_subjects else None
    return common_subjects, r_baseline_all, r_comparison_all, channel_names


def plot_single_model(df, model_tag, save_dir, baseline, comparison,
                       baseline_label, comparison_label):
    """Per-model baseline-vs-comparison violin, per-subject delta-r bar, and
    per-channel FDR-corrected significance. Adapted from run_models.ipynb's
    `plot()`, pulling subjects/channels from `df` instead of an assumed range."""
    subjects, r_baseline_all, r_comparison_all, channel_names = _paired_feature_set_arrays(
        df, model_tag, baseline, comparison)
    if len(subjects) == 0:
        print(f"  [warn] {model_tag}: no subjects with both feature_sets — skipping")
        return

    n_subjects = len(subjects)
    n_channels = r_baseline_all.shape[1]
    r_baseline_per_sub = r_baseline_all.mean(axis=1)
    r_comparison_per_sub = r_comparison_all.mean(axis=1)
    r_diff_per_sub = r_comparison_per_sub - r_baseline_per_sub

    mean_diff, p_val = permutation_test(r_baseline_per_sub, r_comparison_per_sub)
    sem_diff = r_diff_per_sub.std() / np.sqrt(n_subjects)
    sig_label = _sig_label(p_val)
    print(f"  {model_tag}: mean dr = {mean_diff:.4f} +/- {sem_diff:.4f} SEM, p = {p_val:.4f} {sig_label}")

    out_dir = Path(save_dir) / 'comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1 – group violin: baseline vs comparison ──
    fig, ax = plt.subplots(figsize=(7, 6))
    data = [r_baseline_per_sub, r_comparison_per_sub]
    labels = [baseline_label, comparison_label]
    colors = ['steelblue', 'darkorange']

    parts = ax.violinplot(data, positions=[1, 2], showmedians=False, showextrema=False)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    for i in range(n_subjects):
        ax.plot([1, 2], [r_baseline_per_sub[i], r_comparison_per_sub[i]],
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

    y_bracket = max(r_baseline_per_sub.max(), r_comparison_per_sub.max()) + 0.005
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
    ax.set_ylabel(f'dr ({comparison} - {baseline})')
    ax.set_title(f'{model_tag}: Per-Subject Improvement from {baseline} to {comparison}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'{model_tag}_delta_r_per_subject.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 3 – per-channel group delta-r + FDR-corrected significance ──
    r_diff_per_channel = (r_comparison_all - r_baseline_all).mean(axis=0)
    p_per_channel = np.array([
        permutation_test(r_baseline_all[:, ch], r_comparison_all[:, ch])[1]
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
    axes[0].set_title(f'{model_tag}: Per-Channel Group dr ({comparison} - {baseline})')
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


def plot_model_comparison(df, save_dir, baseline, comparison,
                           baseline_label, comparison_label, models=None):
    """Cross-model violin: baseline vs comparison, side by side per model_tag.
    Adapted from run_models.ipynb's `plot_across_models()`."""
    model_tags = models if models is not None else sorted(df['model_tag'].unique())
    colors = ['steelblue', 'darkorange']
    fig, ax = plt.subplots(figsize=(max(12, 3 * len(model_tags)), 6))

    spacing = 3
    centers = []
    legend_handles = [
        Patch(facecolor=colors[0], edgecolor='none', label=baseline_label),
        Patch(facecolor=colors[1], edgecolor='none', label=comparison_label),
    ]

    plotted_tags = []
    for i, model_tag in enumerate(model_tags):
        subjects, r_baseline_all, r_comparison_all, _ = _paired_feature_set_arrays(
            df, model_tag, baseline, comparison)
        if len(subjects) == 0:
            continue
        plotted_tags.append(model_tag)

        r_baseline_per_sub = r_baseline_all.mean(axis=1)
        r_comparison_per_sub = r_comparison_all.mean(axis=1)

        base = len(plotted_tags[:-1]) * spacing
        pos = [base + 1, base + 2]
        centers.append((pos[0] + pos[1]) / 2)

        parts = ax.violinplot([r_baseline_per_sub, r_comparison_per_sub],
                               positions=pos, showmedians=False, showextrema=False)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        for j in range(len(r_baseline_per_sub)):
            ax.plot([pos[0], pos[1]], [r_baseline_per_sub[j], r_comparison_per_sub[j]],
                    color='gray', alpha=0.4, linewidth=0.7)

        jitter = np.random.uniform(-0.04, 0.04, size=len(r_baseline_per_sub))
        ax.scatter(np.full(len(r_baseline_per_sub), pos[0]) + jitter, r_baseline_per_sub,
                   color=colors[0], edgecolors='white', linewidths=0.5, s=40, alpha=0.8, zorder=4)
        jitter = np.random.uniform(-0.04, 0.04, size=len(r_comparison_per_sub))
        ax.scatter(np.full(len(r_comparison_per_sub), pos[1]) + jitter, r_comparison_per_sub,
                   color=colors[1], edgecolors='white', linewidths=0.5, s=40, alpha=0.8, zorder=4)

        for k, d in enumerate([r_baseline_per_sub, r_comparison_per_sub]):
            mean_val = d.mean()
            sem = d.std() / np.sqrt(len(d))
            ax.errorbar(pos[k], mean_val, yerr=sem, fmt='o', color='black',
                        markersize=6, capsize=4, linewidth=1.5, zorder=5)

        try:
            _, p_val = permutation_test(r_baseline_per_sub, r_comparison_per_sub)
            sig_label = _sig_label(p_val)
            y_bracket = max(r_baseline_per_sub.max(), r_comparison_per_sub.max()) + 0.005
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
    """Mean r (averaged over channels and subjects) per model_tag x feature_set."""
    summary = (
        df.assign(mean_r=df['r_per_channel'].apply(np.mean))
          .groupby(['model_tag', 'feature_set'])['mean_r']
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


def main(save_dir, models=None, config=None):
    """`models`, if given, restricts to those model_tag values (e.g.
    ['sklearn_ridge', 'conv_nonlinear']); None means no filtering (whatever's
    present in save_dir). Deliberately does NOT fall back to parsing
    sys.argv itself -- run_all_models.py calls this in-process with its own
    argv still on the stack, and its `--models` flag uses a different
    vocabulary (short script names like 'sklearn') than this module's
    model_tag names, so re-deriving `models` from sys.argv here would
    silently misinterpret it (see the CLI wrapper below for the
    argv-parsing this function itself intentionally avoids).

    `config` is required -- it's the source of which 2 feature_sets to
    compare (baseline vs. comparison) and their display labels; see
    _resolve_feature_sets."""
    if config is None:
        raise ValueError(
            "compare_models.main() requires a `config` (from config.load_config()) "
            "to resolve which feature_sets to compare.")
    baseline, comparison, baseline_label, comparison_label = _resolve_feature_sets(config)

    save_dir = Path(save_dir)
    print(f"Comparing results in {save_dir}")
    df = discover_results(save_dir, models=models)

    print_summary_table(df)

    for model_tag in sorted(df['model_tag'].unique()):
        plot_single_model(df, model_tag, save_dir, baseline, comparison,
                           baseline_label, comparison_label)
    plot_model_comparison(df, save_dir, baseline, comparison,
                           baseline_label, comparison_label, models=models)

    print(f"\nComparison plots saved to {save_dir / 'comparison'}")


if __name__ == '__main__':
    _args, _ = _build_parser().parse_known_args(sys.argv[1:])
    _config = load_config(cli_args=sys.argv[1:])
    _save_dir = Path(_args.save_dir) if _args.save_dir else _config.save_dir
    _models = _args.models.split(',') if _args.models else None
    main(_save_dir, models=_models, config=_config)
