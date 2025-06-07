import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score


df = pd.read_csv("augmented_salary_data.csv")
drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours", "Attrition"]
df_cleaned = df.drop(columns=drop_cols)

df_encoded = pd.get_dummies(df_cleaned, drop_first=True)

X = df_encoded.drop(columns=["Increment_PerformanceBased", "FutureSalary_PerformanceBased"])
y_inc = df_encoded["Increment_PerformanceBased"]
y_sal = df_encoded["FutureSalary_PerformanceBased"]


X_train_sal, X_test_sal, y_train_sal, y_test_sal = train_test_split(X, y_sal, test_size=0.2, random_state=42)
scaler_sal = StandardScaler()
X_train_sal_scaled = scaler_sal.fit_transform(X_train_sal)
X_test_sal_scaled = scaler_sal.transform(X_test_sal)

svr_sal = SVR(kernel='rbf', C=1000, epsilon=0.2, gamma=0.01)
svr_sal.fit(X_train_sal_scaled, y_train_sal)
y_pred_sal = svr_sal.predict(X_test_sal_scaled)

r2_sal = r2_score(y_test_sal, y_pred_sal)
rmse_sal = np.sqrt(mean_squared_error(y_test_sal, y_pred_sal))
mape_sal = mean_absolute_percentage_error(y_test_sal, y_pred_sal)

df_test_sal = pd.DataFrame({
    "Actual_FutureSalary_PerformanceBased": y_test_sal.values,
    "Predicted_FutureSalary_PerformanceBased": y_pred_sal
})
df_test_sal.to_csv("svr_predicted_futuresalary_performance_based.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_sal, y_pred_sal, alpha=0.6, edgecolors='k')
plt.plot([y_test_sal.min(), y_test_sal.max()], [y_test_sal.min(), y_test_sal.max()], 'r--')
plt.xlabel("Actual Future Salary (PerformanceBased)")
plt.ylabel("Predicted")
plt.title(f"SVR Model A: PerformanceBased\nR²: {r2_sal:.4f} | RMSE: {rmse_sal:.2f} | MAPE: {mape_sal:.2%}")
plt.grid(True)
plt.tight_layout()
plt.savefig("svr_performance_based_actual_vs_predicted.png")
plt.close()

X_train_inc, X_test_inc, y_train_inc, y_test_inc = train_test_split(X, y_inc, test_size=0.2, random_state=42)
scaler_inc = StandardScaler()
X_train_inc_scaled = scaler_inc.fit_transform(X_train_inc)
X_test_inc_scaled = scaler_inc.transform(X_test_inc)

svr_inc = SVR(kernel='rbf', C=1000, epsilon=0.2, gamma=0.01)
svr_inc.fit(X_train_inc_scaled, y_train_inc)
y_pred_inc = svr_inc.predict(X_test_inc_scaled)

monthly_income_test = df_encoded.loc[y_test_inc.index, 'MonthlyIncome']
y_pred_salary_from_inc = monthly_income_test * y_pred_inc
y_actual_salary_from_inc = df_encoded.loc[y_test_inc.index, 'FutureSalary_PerformanceBased']

r2_inc_based = r2_score(y_actual_salary_from_inc, y_pred_salary_from_inc)
rmse_inc_based = np.sqrt(mean_squared_error(y_actual_salary_from_inc, y_pred_salary_from_inc))
mape_inc_based = mean_absolute_percentage_error(y_actual_salary_from_inc, y_pred_salary_from_inc)

df_test_inc = pd.DataFrame({
    "Actual_FutureSalary": y_actual_salary_from_inc.values,
    "Predicted_FutureSalary_from_Increment": y_pred_salary_from_inc
})
df_test_inc.to_csv("svr_predicted_futuresalary_from_increment.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_actual_salary_from_inc, y_pred_salary_from_inc, alpha=0.6, edgecolors='k')
plt.plot([y_actual_salary_from_inc.min(), y_actual_salary_from_inc.max()],
         [y_actual_salary_from_inc.min(), y_actual_salary_from_inc.max()], 'r--')
plt.xlabel("Actual Future Salary (PerformanceBased)")
plt.ylabel("Predicted (via Increment * MonthlyIncome)")
plt.title(f"SVR Model B: Increment-Based Salary Prediction\nR²: {r2_inc_based:.4f} | RMSE: {rmse_inc_based:.2f} | MAPE: {mape_inc_based:.2%}")
plt.grid(True)
plt.tight_layout()
plt.savefig("svr_increment_based_salary_actual_vs_predicted.png")
plt.close()


print("💰 SVR Model A: FutureSalary_PerformanceBased")
print(f"R² Score : {r2_sal:.4f}")
print(f"RMSE     : {rmse_sal:.2f}")
print(f"MAPE     : {mape_sal:.2%}")
print("Saved: svr_predicted_futuresalary_performance_based.csv & plot\n")

print("🔁 SVR Model B: Salary = Increment * MonthlyIncome")
print(f"R² Score : {r2_inc_based:.4f}")
print(f"RMSE     : {rmse_inc_based:.2f}")
print(f"MAPE     : {mape_inc_based:.2%}")
print("Saved: svr_predicted_futuresalary_from_increment.csv & plot")