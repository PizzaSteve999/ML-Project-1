import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def histogram(losses: pd.Series, model_name: str, bins: int = 30):
    plt.figure()
    plt.hist(losses, bins=bins, alpha=0.75)
    plt.xlabel('Expected Salary Loss')
    plt.ylabel('Number of Employees')
    plt.title(f'{model_name}: Distribution of Expected Salary Loss')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    p = Path(__file__).parent
    df_fixed = pd.read_csv(p / 'expected_loss_fixed.csv')
    df_perf = pd.read_csv(p / 'expected_loss_performance.csv')
    fixed_losses = df_fixed['ExpectedLoss']
    perf_losses = df_perf['ExpectedLoss']
    histogram(fixed_losses, 'Fixed-Based Model')
    histogram(perf_losses, 'Performance-Based Model')

if __name__ == '__main__':
    main()
