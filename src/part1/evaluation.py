import matplotlib.pyplot as plt
import numpy as np

models = ['Logistic Regression', 'Decision Tree', 'SVM', 'Random Forest', 'RF + SMOTE', 'XGBoost']
accuracy = [0.88, 0.77, 0.82, 0.84, 0.82, 0.85]
precision = [0.72, 0.27, 0.45, 0.44, 0.41, 0.54]
recall = [0.41, 0.26, 0.53, 0.09, 0.26, 0.45]
f1_score = [0.52, 0.26, 0.49, 0.14, 0.32, 0.49]
roc_auc = [0.82, 0.56, 0.79, 0.75, 0.75, 0.75]

data = list(zip(models, accuracy, precision, recall, f1_score, roc_auc))
sorted_data = sorted(data, key=lambda x: x[4], reverse=True)

models_sorted, accuracy, precision, recall, f1_score, roc_auc = zip(*sorted_data)
metrics = [accuracy, precision, recall, f1_score, roc_auc]
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
colors = ['skyblue', 'lightgreen', 'salmon', 'lightsteelblue', 'orange']

fig, ax = plt.subplots(figsize=(12, 8))
bar_height = 0.12
y = np.arange(len(models_sorted))

for i, (metric, color) in enumerate(zip(metrics, colors)):
    bars = ax.barh(y + i * bar_height, metric, bar_height, label=metric_labels[i], color=color)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center', fontsize=9)

ax.set_yticks(y + 2 * bar_height)
ax.set_yticklabels(models_sorted)
ax.set_xlabel('Score')
ax.set_xlim(0, 1.05)
ax.set_title('Model Comparison (Sorted by F1 Score)', fontsize=16)
ax.invert_yaxis()
ax.grid(True, axis='x', linestyle='--', alpha=0.6)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=10)

plt.tight_layout()
plt.show()
