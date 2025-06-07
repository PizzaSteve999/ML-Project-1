import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score, accuracy_score

base_dir = Path(__file__).parent
part1_file = base_dir.parent / 'part1' / 'part1lg_output.txt'
df = pd.read_csv(part1_file, sep='\t', index_col='EmployeeIndex')
df['P_stay'] = 1.0 - df['Attrition_Probability']

labels_csv = base_dir.parent.parent / 'dataset' / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
labels_df = pd.read_csv(labels_csv, index_col='EmployeeNumber')

y_true = (labels_df['Attrition'] == 'No').iloc[df.index].astype(int).values
y_scores = df['P_stay'].values

thresholds = np.linspace(0, 1, 101)
metrics = []
for t in thresholds:
    y_pred = (y_scores >= t).astype(int)
    metrics.append({
        'threshold': t,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    })

metrics_df = pd.DataFrame(metrics)
print("\nTop 5 thresholds by F1 score:")
print(metrics_df.sort_values('f1', ascending=False).head(5))

best_row = metrics_df.loc[metrics_df['f1'].idxmax()]
best_threshold = best_row['threshold']
best_f1 = best_row['f1']
best_prec = best_row['precision']
best_rec = best_row['recall']
print(f"\nBest F1 threshold: {best_threshold:.2f} (F1={best_f1:.3f}, Precision={best_prec:.3f}, Recall={best_rec:.3f})")

df['likely_to_stay'] = df['P_stay'] >= best_threshold
counts = df['likely_to_stay'].value_counts()

plt.figure()
plt.pie(counts.values, labels=['Stay', 'Leave'], autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
plt.title(f'Pie Chart of Stay vs. Leave (Threshold = {best_threshold:.2f})')
plt.axis('equal')
plt.show()
