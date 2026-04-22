"""
compare_stochastic_methods.py  (v4)

One figure per method (TTA / GPS / MCDropout).
Layout: 1 row x 3 cols (one per shift type), sharey across all cells.
Each cell: 3 boxplots (DELTA-AUROC-F | Delta-Acc | Delta-AUGRC) with individual
scatter dots, and lines connecting the same (dataset, backbone, setup)
entry across the 3 metric boxes within the cell.

Usage:
    python compare_stochastic_methods.py [--subdirs ...] [--output-dir <path>] [--show]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[4]
OLD_DIR   = REPO_ROOT / 'Benchmarks' / 'medMNIST' / 'results_backup_20260414' / 'jsons_results'
NEW_DIR   = REPO_ROOT / 'Benchmarks' / 'medMNIST' / 'results' / 'jsons_results'
OUT_DIR   = REPO_ROOT / 'Benchmarks' / 'medMNIST' / 'results' / 'figures' / 'stochastic_comparison'

SUBDIRS       = ['in_distribution', 'corruption_shifts', 'population_shifts']
METHODS       = ['TTA', 'GPS', 'MCDropout']

SHIFT_LABELS  = {
    'in_distribution':   'In-Distribution',
    'corruption_shifts': 'Corruption shifts',
    'population_shifts': 'Population shifts',
}
METRIC_KEYS    = ['auroc_f_mean', 'accuracy_mean', 'augrc_mean']
METRIC_LABELS  = ['AUROC-F', 'Acc', 'AUGRC']
METRIC_COLORS  = ['#4C72B0', '#DD8452', '#55A868']

DATASET_ORDER = [
    'bloodmnist', 'breastmnist', 'dermamnist-e-id', 'octmnist',
    'organamnist', 'pathmnist', 'pneumoniamnist', 'tissuemnist',
    'amos2022', 'dermamnist-e-external',
]
_cmap = plt.cm.tab10(np.linspace(0, 1, 10))
DS_COLOR = {ds: _cmap[i] for i, ds in enumerate(DATASET_ORDER)}

JITTER_SCALE = 0.10
DOT_SIZE     = 22
LINE_ALPHA   = 0.30
LINE_LW      = 0.8


# ─── Data loading ─────────────────────────────────────────────────────────────

def extract_metrics(md: dict) -> dict | None:
    if not md:
        return None
    auroc = md.get('auroc_f_mean')
    augrc = md.get('augrc_mean')
    pf    = md.get('per_fold_metrics', [])
    if auroc is None or augrc is None or not pf:
        return None
    return {
        'auroc_f_mean':  float(auroc),
        'accuracy_mean': float(np.mean([f['accuracy'] for f in pf])),
        'augrc_mean':    float(augrc),
    }


def collect_diffs(method_key: str, subdirs: list[str]) -> list[dict]:
    entries = []
    for subdir in subdirs:
        new_sub, old_sub = NEW_DIR / subdir, OLD_DIR / subdir
        if not new_sub.exists() or not old_sub.exists():
            continue
        for new_jf in sorted(new_sub.glob('*.json')):
            old_jf = old_sub / new_jf.name
            if not old_jf.exists():
                continue
            with open(new_jf) as f: nd = json.load(f)
            with open(old_jf) as f: od = json.load(f)
            nm = extract_metrics(nd.get('methods', {}).get(method_key))
            om = extract_metrics(od.get('methods', {}).get(method_key))
            if nm is None or om is None:
                continue
            entries.append({
                'dataset':  nd.get('flag', ''),
                'backbone': nd.get('model_backbone', ''),
                'setup':    nd.get('setup', '') or '',
                'shift':    subdir,
                **{f'{k}_diff': nm[k] - om[k] for k in METRIC_KEYS},
            })
    return entries


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_method(method_key: str, entries: list[dict], out_dir: Path, show: bool) -> None:
    if not entries:
        print(f"  No data for {method_key}, skipping.")
        return

    shifts   = [s for s in SUBDIRS if any(e['shift'] == s for e in entries)]
    n_shifts = len(shifts)

    # Assign a fixed per-entry jitter so dots stay aligned across metrics
    all_keys = sorted({(e['dataset'], e['backbone'], e['setup']) for e in entries})
    rng = np.random.default_rng(0)
    jitter_map = {k: rng.uniform(-JITTER_SCALE, JITTER_SCALE) for k in all_keys}

    fig, axes = plt.subplots(
        1, n_shifts,
        figsize=(3.8 * n_shifts, 4.5),
        sharey=True, squeeze=False,
    )
    fig.subplots_adjust(wspace=0.06)

    for si, sh in enumerate(shifts):
        ax   = axes[0][si]
        cell = [e for e in entries if e['shift'] == sh]
        if not cell:
            ax.set_visible(False)
            continue

        # ── one boxplot per metric at x = 0, 1, 2 ──────────────────────────
        for mi, (mk, mc) in enumerate(zip(METRIC_KEYS, METRIC_COLORS)):
            vals = [e[f'{mk}_diff'] for e in cell]
            ax.boxplot(
                vals, positions=[mi], widths=0.45,
                showfliers=False, patch_artist=True,
                medianprops=dict(color='black', lw=1.8),
                boxprops=dict(facecolor=mc, alpha=0.25, lw=0.8),
                whiskerprops=dict(lw=0.8, linestyle='--', color=mc),
                capprops=dict(lw=0.8, color=mc),
            )

        # ── scatter + connection lines per entry ────────────────────────────
        for key in all_keys:
            key_entries = [e for e in cell
                           if (e['dataset'], e['backbone'], e['setup']) == key]
            if not key_entries:
                continue
            e0    = key_entries[0]
            jx    = jitter_map[key]
            color = DS_COLOR.get(key[0], 'grey')
            xs, ys = [], []
            for mi, mk in enumerate(METRIC_KEYS):
                v = e0[f'{mk}_diff']
                xs.append(mi + jx)
                ys.append(v)
                ax.scatter(mi + jx, v, s=DOT_SIZE, color=color,
                           alpha=0.80, zorder=5, linewidths=0)
            # lines connecting the 3 metric dots for this entry
            ax.plot(xs, ys, '-', color=color, alpha=LINE_ALPHA,
                    lw=LINE_LW, zorder=4)

        # ── zero line ───────────────────────────────────────────────────────
        ax.axhline(0, color='k', lw=0.8, ls='--', zorder=2)

        # ── axes styling ────────────────────────────────────────────────────
        ax.set_xlim(-0.6, len(METRIC_KEYS) - 0.4)
        ax.set_xticks(range(len(METRIC_KEYS)))
        ax.set_xticklabels(METRIC_LABELS, fontsize=10)
        ax.tick_params(axis='x', length=0, pad=4)
        ax.grid(axis='y', ls=':', alpha=0.4, zorder=0)
        ax.tick_params(axis='y', labelsize=7)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune='both'))
        ax.set_title(SHIFT_LABELS.get(sh, sh), fontsize=10, pad=5)
        if si == 0:
            ax.set_ylabel('new - old', fontsize=9)

    # ── dataset legend ──────────────────────────────────────────────────────
    present_ds = sorted({e['dataset'] for e in entries},
                        key=lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else 99)
    handles = [mlines.Line2D([], [], marker='o', ls='', markersize=6,
                              color=DS_COLOR.get(ds, 'grey'), label=ds)
               for ds in present_ds]
    fig.legend(handles=handles, loc='lower center', ncol=min(5, len(handles)),
               fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f'{method_key}  -  new - old', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'compare_{method_key.lower()}.png'
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subdirs', nargs='+', default=SUBDIRS, choices=SUBDIRS)
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    for mk in METHODS:
        print(f"\n{'='*50}\n{mk}\n{'='*50}")
        entries = collect_diffs(mk, args.subdirs)
        print(f"  {len(entries)} paired entries.")
        plot_method(mk, entries, args.output_dir, args.show)

    print("\nDone.")


if __name__ == '__main__':
    main()
