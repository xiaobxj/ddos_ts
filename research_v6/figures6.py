"""Exportable scientific figures from finished, verified round-6 results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
ARMS = ['baseline', 'small', 'dropout', 'decay', 'combined']
NAMES = {'baseline': 'Baseline', 'small': 'Smaller width', 'dropout': 'Dropout 0.3',
         'decay': 'Weight decay 0.1', 'combined': 'All three changes'}
COLORS = {'baseline': '#42566e', 'small': '#1b8e81', 'dropout': '#337abb',
          'decay': '#bd7527', 'combined': '#a5548e'}
HISTORIES = ['full', 'drop_early', 'drop_middle', 'drop_recent']
HISTORY_NAMES = ['Full history', 'Omit earliest 20%', 'Omit middle 20%', 'Omit latest 20%']


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    assert read('verification.json')['status'] == 'PASS'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18})
    metrics = pd.read_csv(OUT / 'main_metrics.csv')
    curves = pd.read_csv(OUT / 'main_training_curves.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13.7, 5.7), layout='constrained')
    for arm in ARMS:
        g = metrics[metrics.arm == arm].sort_values('epoch')
        axes[0].plot(g.epoch, g.rmse * 10000, marker='o', label=NAMES[arm], color=COLORS[arm])
        train = curves[curves.arm == arm].groupby('epoch').final_model_train_return_mse.mean().dropna()
        axes[1].plot(train.index, train, marker='o', label=NAMES[arm], color=COLORS[arm])
    mean = next(x['rmse'] for x in read('selection.json')['baseline_metrics'] if x['method'] == 'training_mean')
    old = read('previous_reference.json')['round4_reference']['rmse']
    axes[0].axhline(mean*10000, color='#777777', linestyle='--', label='Mature training mean')
    axes[0].axhline(old*10000, color='#777777', linestyle=':', label='Round-4 frozen reference*')
    axes[0].set(title='Chronological validation: three-seed ensemble',
                xlabel='Training epochs', ylabel='Execution-return RMSE (bp)')
    axes[1].set(title='Full retained training set: mean of nine fits',
                xlabel='Training epochs', ylabel='Final-model standardized return MSE')
    for ax in axes:
        ax.set_xticks([2, 5, 10, 20])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=9, frameon=False)
    fig.suptitle('Model size and regularization | fixed raw OHLCV inputs and joint target\n'
                 '2018-2020 historical validation; *round-4 training set has five additional early rows', fontsize=13)
    fig.savefig(OUT / 'regularization_comparison.png', dpi=180)
    fig.savefig(OUT / 'regularization_comparison.svg')
    plt.close(fig)

    block = pd.read_csv(OUT / 'block_metrics.csv')
    sensitivity = pd.read_csv(OUT / 'block_prediction_sensitivity.csv')
    values = block.pivot(index='arm', columns='history', values='mse_skill_vs_training_mean').loc[ARMS, HISTORIES].to_numpy() * 100
    ranks = block.pivot(index='arm', columns='history', values='rank_within_history').loc[ARMS, HISTORIES].to_numpy()
    limit = max(float(np.abs(values).max()), 1.)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1]})
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    shown = axes[0].imshow(values, aspect='auto', cmap='RdYlGn', norm=norm)
    axes[0].grid(False)
    axes[0].set_xticks(range(4), HISTORY_NAMES, rotation=18, ha='right')
    axes[0].set_yticks(range(5), [NAMES[a] for a in ARMS])
    for row in range(5):
        for col in range(4):
            color = 'white' if abs(values[row, col]) > .7 * limit else '#172331'
            axes[0].text(col, row, f'{values[row,col]:+.2f}%\nrank {ranks[row,col]}', ha='center', va='center', color=color, fontsize=10)
    axes[0].set_title('MSE improvement over the retained training mean\nPositive is better; ranks compare five arms within a history')
    fig.colorbar(shown, ax=axes[0], shrink=.78, label='MSE improvement (%)', pad=.02)
    for arm in ARMS:
        g = sensitivity[sensitivity.arm == arm].set_index('history').loc[HISTORIES[1:]]
        axes[1].plot(range(3), g.prediction_rms_shift*10000, marker='o', color=COLORS[arm], label=NAMES[arm])
    axes[1].set_xticks(range(3), ['Omit earliest', 'Omit middle', 'Omit latest'])
    axes[1].set(title='Forecast change relative to full-history fit',
                ylabel='RMS prediction shift (bp)', xlabel='Approximately 20% of labeled anchors omitted')
    axes[1].legend(fontsize=9, frameon=False)
    fig.suptitle('Training-history sensitivity | fixed 10 epochs and seed 20260910\n'
                 'Equal optimizer updates; retained rows repeated; historical diagnostic only', fontsize=13)
    fig.savefig(OUT / 'block_stability.png', dpi=180)
    fig.savefig(OUT / 'block_stability.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
