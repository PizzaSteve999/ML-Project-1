import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier

df = pd.read_csv("../../dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

sns.countplot(x='Attrition', data=df)
plt.title("Attrition Class Distribution")
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.show()

df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'scale_pos_weight': [1, 2]
}

xgb_random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_grid,
    scoring='roc_auc',
    cv=5,
    n_iter=10,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

xgb_random_search.fit(X_train_smote, y_train_smote)

print("Best Parameters for XGBoost:\n", xgb_random_search.best_params_)

xgb_best_model = xgb_random_search.best_estimator_

y_pred_xgb = xgb_best_model.predict(X_test)
y_proba_xgb = xgb_best_model.predict_proba(X_test)[:, 1]

print("XGBoost - Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("XGBoost - ROC AUC Score:", roc_auc_score(y_test, y_proba_xgb))

conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

RocCurveDisplay.from_estimator(xgb_best_model, X_test, y_test)
plt.title('ROC Curve - XGBoost')
plt.show()

importances_xgb = pd.Series(xgb_best_model.feature_importances_, index=X.columns)
top_features_xgb = importances_xgb.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_features_xgb.values, y=top_features_xgb.index)
plt.title("Top 10 Important Features - XGBoost")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

final_model = xgb_best_model
X_all = df.drop("Attrition", axis=1)
proba_all = final_model.predict_proba(X_all)[:, 1]

output_df = pd.DataFrame({
    "EmployeeIndex": X_all.index,
    "Attrition_Probability": proba_all
})

output_df.to_csv("part1_output.txt", index=False, sep='\t')
print("Predicted attrition probabilities saved to part1_output.txt")
