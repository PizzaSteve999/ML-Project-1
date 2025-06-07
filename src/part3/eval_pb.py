import matplotlib.pyplot as plt
import numpy as np

models = ['Random Forest', 'Lasso', 'Ridge', 'SVR RBF', 'SVR Linear', 'SVR Polynomial']
r2_scores = [0.9978, 0.9975, 0.9976, 0.9953, 0.8812, 0.9953]
rmse = [257.00, 263.41, 258.01, 341.08, 1707.39, 341.08]
mape = [2.20, 2.26, 2.31, 3.08, 18.95, 3.08]

def plot_metric(values, title, xlabel, filename, color='skyblue'):
    fig, ax = plt.subplots(figsize=(15, 6))
    y = np.arange(len(models))
    bars = ax.barh(y, values, height=0.5, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    for rect in bars:
        width = rect.get_width()
        ax.annotate(f'{width:.2f}', xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=9)
    plt.tight_layout(pad=1.5)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved graph: {filename}")

plot_metric(r2_scores, 'Fixed Growth: R² Score Comparison', 'R² Score', 'fixed_r2_score_comparison.png', color='lightblue')
plot_metric(rmse, 'Fixed Growth: RMSE Comparison', 'RMSE', 'fixed_rmse_comparison.png', color='lightcoral')
plot_metric(mape, 'Fixed Growth: MAPE Comparison', 'MAPE (%)', 'fixed_mape_comparison.png', color='lightgreen')