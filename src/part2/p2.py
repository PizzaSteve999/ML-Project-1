import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

DATA_PATH = os.path.join("..", "..", "dataset", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = pd.read_csv(DATA_PATH)
print("Initial dataset shape:", df.shape)

df['Increment_FixedGrowth'] = 1.08
df['FutureSalary_FixedGrowth'] = df['MonthlyIncome'] * df['Increment_FixedGrowth']
df['Increment_PerformanceBased'] = df['PerformanceRating'].apply(lambda x: 1.10 if x == 4 else 1.05)
df['FutureSalary_PerformanceBased'] = df['MonthlyIncome'] * df['Increment_PerformanceBased']

np.random.seed(42)
noise = np.random.normal(loc=1.0, scale=0.03, size=len(df))
df['FutureSalary_FixedGrowth'] = df['FutureSalary_FixedGrowth'] * noise
df['FutureSalary_FixedGrowth'] = df['FutureSalary_FixedGrowth'].round(2)
df['FutureSalary_PerformanceBased'] = df['FutureSalary_PerformanceBased'] * noise
df['FutureSalary_PerformanceBased'] = df['FutureSalary_PerformanceBased'].round(2)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "augmented_salary_data.csv")
df.to_csv(OUTPUT_PATH, index=False)
print(f"[✓] Future salary simulation complete.")
print(f"[✓] Augmented dataset saved to: {OUTPUT_PATH}")

X = df[['PerformanceRating']]
y = df['FutureSalary_PerformanceBased']
model = LinearRegression()
model.fit(X, y)
df['Predicted_FutureSalary_PerformanceBased'] = model.predict(X)
print(f"Linear regression model fitted: PerformanceRating vs FutureSalary_PerformanceBased")
print(f"Model coefficients: {model.coef_}, Intercept: {model.intercept_}")

plt.figure(figsize=(10, 6))
sns.kdeplot(df['MonthlyIncome'], label="Current Salary", shade=True)
sns.kdeplot(df['FutureSalary_FixedGrowth'], label="Future Salary (Fixed Growth)", shade=True)
sns.kdeplot(df['FutureSalary_PerformanceBased'], label="Future Salary (Performance-Based)", shade=True)
plt.title("Salary Distribution: Current vs Future (Fixed Growth vs Performance-Based)")
plt.xlabel("Salary (Monthly Income)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='PerformanceRating', y='FutureSalary_PerformanceBased', data=df)
plt.title("Future Salary (Performance-Based) by Performance Rating")
plt.xlabel("Performance Rating")
plt.ylabel("Simulated Future Salary (Performance-Based)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='PerformanceRating', y='FutureSalary_FixedGrowth', data=df)
plt.title("Future Salary (Fixed Growth) by Performance Rating")
plt.xlabel("Performance Rating")
plt.ylabel("Simulated Future Salary (Fixed Growth)")
plt.tight_layout()
plt.show()

correlation_columns = ['MonthlyIncome', 'FutureSalary_FixedGrowth', 'FutureSalary_PerformanceBased', 'PerformanceRating']
correlation_matrix = df[correlation_columns].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix: Salary vs Performance Rating")
plt.tight_layout()
plt.show()
