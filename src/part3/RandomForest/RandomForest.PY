import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("augmented_salary_data.csv")
drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours", "Attrition"]
df_cleaned = df.drop(columns=drop_cols)
df_encoded = pd.get_dummies(df_cleaned, drop_first=True)

X = df_encoded.drop(columns=["Increment_PerformanceBased", "FutureSalary_PerformanceBased"])
y_inc = df_encoded["Increment_PerformanceBased"]
y_sal = df_encoded["FutureSalary_PerformanceBased"]
monthly_income = df_encoded["MonthlyIncome"]

X_train_inc, X_test_inc, y_train_inc, y_test_inc = train_test_split(X, y_inc, test_size=0.2, random_state=42)
X_train_sal, X_test_sal, y_train_sal, y_test_sal = train_test_split(X, y_sal, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_inc_scaled = scaler.fit_transform(X_train_inc)
X_test_inc_scaled = scaler.transform(X_test_inc)
X_train_sal_scaled = scaler.fit_transform(X_train_sal)
X_test_sal_scaled = scaler.transform(X_test_sal)

rf_sal = RandomForestRegressor(n_estimators=100, random_state=42)
rf_sal.fit(X_train_sal_scaled, y_train_sal)
pred_sal = rf_sal.predict(X_test_sal_scaled)

r2_sal = r2_score(y_test_sal, pred_sal)
rmse_sal = np.sqrt(mean_squared_error(y_test_sal, pred_sal))
mape_sal = mean_absolute_percentage_error(y_test_sal, pred_sal)

df_modelA = pd.DataFrame({
    "Actual_FutureSalary": y_test_sal.values,
    "Predicted_FutureSalary": pred_sal
})
df_modelA.to_csv("rf_predictions_performance_based.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_sal, pred_sal, alpha=0.6, edgecolors='k')
plt.plot([y_test_sal.min(), y_test_sal.max()], [y_test_sal.min(), y_test_sal.max()], 'r--')
plt.xlabel("Actual Future Salary")
plt.ylabel("Predicted Future Salary")
plt.title(f"Model A (Direct): R² = {r2_sal:.4f}, RMSE = {rmse_sal:.2f}, MAPE = {mape_sal:.2%}")
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_modelA_futuresalary_actual_vs_predicted.png")
plt.close()

rf_inc = RandomForestRegressor(n_estimators=100, random_state=42)
rf_inc.fit(X_train_inc_scaled, y_train_inc)
pred_inc = rf_inc.predict(X_test_inc_scaled)

monthly_income_test = df_encoded.loc[y_test_inc.index, 'MonthlyIncome']
pred_salary_from_inc = pred_inc * monthly_income_test
actual_salary_from_inc = df_encoded.loc[y_test_inc.index, 'FutureSalary_PerformanceBased']

r2_inc = r2_score(actual_salary_from_inc, pred_salary_from_inc)
rmse_inc = np.sqrt(mean_squared_error(actual_salary_from_inc, pred_salary_from_inc))
mape_inc = mean_absolute_percentage_error(actual_salary_from_inc, pred_salary_from_inc)

df_modelB = pd.DataFrame({
    "Actual_FutureSalary": actual_salary_from_inc.values,
    "Predicted_FutureSalary_FromIncrement": pred_salary_from_inc
})
df_modelB.to_csv("rf_predictions_increment_based.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(actual_salary_from_inc, pred_salary_from_inc, alpha=0.6, edgecolors='k')
plt.plot([actual_salary_from_inc.min(), actual_salary_from_inc.max()],
         [actual_salary_from_inc.min(), actual_salary_from_inc.max()], 'r--')
plt.xlabel("Actual Future Salary")
plt.ylabel("Predicted (Increment * MonthlyIncome)")
plt.title(f"Model B (Increment-Based): R² = {r2_inc:.4f}, RMSE = {rmse_inc:.2f}, MAPE = {mape_inc:.2%}")
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_modelB_increment_based_actual_vs_predicted.png")
plt.close()

print("✅ Model A: Directly Predicting FutureSalary_PerformanceBased")
print(f"R² Score  : {r2_sal:.4f}")
print(f"RMSE      : {rmse_sal:.2f}")
print(f"MAPE      : {mape_sal:.2%}")
print("Saved: rf_predictions_performance_based.csv & plot\n")

print("✅ Model B: Predict Increment and Multiply with MonthlyIncome")
print(f"R² Score  : {r2_inc:.4f}")
print(f"RMSE      : {rmse_inc:.2f}")
print(f"MAPE      : {mape_inc:.2%}")
print("Saved: rf_predictions_increment_based.csv & plot")

X_all_sal_scaled = scaler.transform(X)
all_pred_sal = rf_sal.predict(X_all_sal_scaled)

pd.DataFrame({
    "Predicted_FutureSalary_PerformanceBased": all_pred_sal
}).to_csv("Predicted_FutureSalary_PerformanceBased", index=False)

print("✅ Saved all Model A predictions ")

X_all_inc_scaled = scaler.transform(X)
all_pred_inc = rf_inc.predict(X_all_inc_scaled)

all_pred_salary_inc = all_pred_inc * monthly_income

pd.DataFrame({
    "Predicted_FutureSalary_FromIncrement": all_pred_salary_inc
}).to_csv("Predicted_FutureSalary_FromIncrement", index=False)

print("✅ Saved all Model B predictions ")