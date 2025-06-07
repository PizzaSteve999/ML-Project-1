import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

df = pd.read_csv("augmented_salary_data.csv")
df_encoded = pd.get_dummies(df, drop_first=True)

features = ['JobInvolvement', 'Education', 'JobSatisfaction',
            'MaritalStatus_Married', 'TotalWorkingYears', 'JobLevel',
            'EnvironmentSatisfaction', 'JobRole_Research Director',
            'WorkLifeBalance', 'PercentSalaryHike', 'JobRole_Manager',
            'PerformanceRating', 'MonthlyIncome']

X = df_encoded[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_performance = df_encoded['FutureSalary_PerformanceBased']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_performance, test_size=0.2, random_state=23)

lasso_cv_pb = LassoCV(cv=5, random_state=42)
lasso_cv_pb.fit(X_train, y_train)

print("Best alpha value for PerformanceBased salary:", lasso_cv_pb.alpha_)

y_pred_performance = lasso_cv_pb.predict(X_test)
r2_performance = r2_score(y_test, y_pred_performance)
rmse_performance = np.sqrt(mean_squared_error(y_test, y_pred_performance))
mape_performance = mean_absolute_percentage_error(y_test, y_pred_performance)

print("\nModel Evaluation Metrics for PerformanceBased Salary:")
print(f"R² Score       : {r2_performance:.4f}")
print(f"RMSE           : {rmse_performance:.2f}")
print(f"MAPE            : {mape_performance:.2%}")

y_fixedgrowth = df_encoded['FutureSalary_FixedGrowth']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_fixedgrowth, test_size=0.2, random_state=23)

lasso_cv_fg = LassoCV(cv=5, random_state=42)
lasso_cv_fg.fit(X_train, y_train)

print("Best alpha value for FixedGrowth salary:", lasso_cv_fg.alpha_)

y_pred_fixedgrowth = lasso_cv_fg.predict(X_test)
r2_fixedgrowth = r2_score(y_test, y_pred_fixedgrowth)
rmse_fixedgrowth = np.sqrt(mean_squared_error(y_test, y_pred_fixedgrowth))
mape_fixedgrowth = mean_absolute_percentage_error(y_test, y_pred_fixedgrowth)

print("\nModel Evaluation Metrics for FixedGrowth Salary:")
print(f"R² Score       : {r2_fixedgrowth:.4f}")
print(f"RMSE           : {rmse_fixedgrowth:.2f}")
print(f"MAPE            : {mape_fixedgrowth:.2%}")

print("\nFeature Coefficients for PerformanceBased Salary:")
coef_performance = pd.Series(lasso_cv_pb.coef_, index=features)
print(coef_performance.sort_values())

print("\nFeature Coefficients for FixedGrowth Salary:")
coef_fixedgrowth = pd.Series(lasso_cv_fg.coef_, index=features)
print(coef_fixedgrowth.sort_values())

y_pred_performance_all = lasso_cv_pb.predict(X_scaled)
y_pred_fixedgrowth_all = lasso_cv_fg.predict(X_scaled)

df['Predicted_FutureSalary_PerformanceBased'] = y_pred_performance_all
df['Predicted_FutureSalary_FixedGrowth'] = y_pred_fixedgrowth_all

df.to_csv("predicted_salaries_part3_lasso.csv", index=False)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_performance, color='skyblue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Lasso: Actual vs Predicted (PerformanceBased) R² = {r2_performance:.4f}")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_fixedgrowth, color='orange', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Lasso: Actual vs Predicted (FixedGrowth) R² = {r2_fixedgrowth:.4f}")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
coef_performance.sort_values().plot(kind='barh', color='skyblue', edgecolor='black')
plt.title("Lasso Coefficients - PerformanceBased Salary")
plt.xlabel("Coefficient Value")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
coef_fixedgrowth.sort_values().plot(kind='barh', color='orange', edgecolor='black')
plt.title("Lasso Coefficients - FixedGrowth Salary")
plt.xlabel("Coefficient Value")
plt.grid(True)
plt.tight_layout()
plt.show()