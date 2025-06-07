import matplotlib.pyplot as plt
import numpy as np

models = ['Random Forest', 'Lasso', 'Ridge', 'SVR RBF', 'SVR Linear', 'SVR Polynomial']
r2_scores = [0.9989, 0.9975, 0.9974, 0.9831, 0.9996, 0.9045]
rmse = [164.03, 259.56, 259.89, 644.41, 95.93, 1530.27]
mape = [0.87, 2.26, 2.31, 6.88, 0.33, 16.70]

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

plot_metric(r2_scores, 'Performance-Based Growth: R² Score Comparison', 'R² Score', 'performance_r2_score_comparison.png', color='lightblue')
plot_metric(rmse, 'Performance-Based Growth: RMSE Comparison', 'RMSE', 'performance_rmse_comparison.png', color='lightcoral')
plot_metric(mape, 'Performance-Based Growth: MAPE Comparison', 'MAPE (%)', 'performance_mape_comparison.png', color='lightgreen')