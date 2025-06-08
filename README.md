#  Multi-Step Regression + Classification for Employee Attrition & Salary Estimation

---

## Overview
Most HR analytics projects focus only on employee attrition prediction. This project adds a second very practical dimension: estimating the future salary of employees likely to stay (using regression models).

---

## Part 1: Employee Attrition Prediction (Classification)

---

##  Objective
The goal of Part 1 is to build and evaluate multiple classification models to predict whether an employee will leave the company (Attrition = Yes/No) using IBM's HR Analytics dataset.

---

##  Dataset
- **File Path:** `dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Features:** Includes attributes like `Age`, `BusinessTravel`, `Department`, `Education`, `EnvironmentSatisfaction`, `JobRole`, `MonthlyIncome`, etc.
- **Target Column:** `Attrition` (Binary: Yes/No)

---

##  Preprocessing Steps
- Handling categorical variables using Label Encoding
- Scaling numerical features using `StandardScaler`
- Train-test split (80% train, 20% test)
- Handled class imbalance using SMOTE in some models

---

##  Classification Models Used

### 1. Logistic Regression
- Simple, interpretable baseline model for binary classification.

### 2. Decision Tree
- Tree-based model for capturing feature interactions.

### 3. Random Forest
- An ensemble of decision trees to reduce overfitting and improve accuracy.

### 4. Random Forest + SMOTE
- Used SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance before applying RF.

### 5. XGBoost
- Gradient boosting algorithm that provides high performance on structured data.

### 6. Support Vector Machine (SVM)
- Maximizes class margin in high-dimensional space; good for binary classification.

---

### Model Performance (Part 1)
- Both **Logistic Regression** and **XGBoost** showed the best performance in predicting employee attrition. However, **Logistic Regression** outperformed **XGBoost** by a slight margin, making it the more reliable model for this classification task.

---

##  Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **ROC-AUC Curve**

---

## Python Libraries Used (For Part 1)

- **Data Manipulation**: 
    - `pandas`
    - `numpy`

- **Visualization**:
    - `matplotlib`
    - `seaborn`

- **Machine Learning**:
    - `scikit-learn`
        - `LogisticRegression`
        - `DecisionTreeClassifier`
        - `RandomForestClassifier`
        - `SVC`
        - `classification_report`
        - `confusion_matrix`
        - `train_test_split`
        - `roc_auc_score`
        - `accuracy_score`
        - `f1_score`
        - `precision_score`
        - `recall_score`

- **Handling Imbalance**:
    - `imblearn` (SMOTE)

- **Boosting**:
    - `xgboost`

 ---

## Part 2: Simulating Future Salaries (Data Augmentation)

---

## Objective
The goal of Part 2 is to simulate future salary predictions for employees based on two growth models:
1. **Fixed Growth Increment**: A uniform salary increase (8%) applied to all employees.
2. **Performance-Based Growth Increment**: Salary increase based on an employee’s performance rating.

Additionally, a **Linear Regression** model is used to analyze the relationship between an employee’s **Performance Rating** and their **simulated future salary**.

---

## Dataset
- **File Path:** `dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Features:** Includes attributes like `MonthlyIncome`, `PerformanceRating`, etc.
- **Target Column:** Simulated future salaries based on fixed growth and performance-based increments.

---

## Simulation Steps
1. **Fixed Growth Increment**: 
   - An 8% increase is applied uniformly to all employees’ monthly income, with added noise (±3%) for realistic simulation.
   
2. **Performance-Based Growth Increment**:
   - Employees with a performance rating of 4 receive a 10% increase, and those with lower ratings receive a 5% increase. Noise is also added to this model.

3. **Linear Regression**:
   - A **Linear Regression** model is built to explore the relationship between **Performance Rating** and **simulated future salary**.

---

## Simulation Results
- Both **Fixed Growth** and **Performance-Based Growth** models produce future salary predictions, which are saved in a new dataset for further analysis.
- The **linear regression model** reveals how performance ratings influence salary increments in the performance-based model.

---

## Model Performance (Part 2)
- The simulation provides future salary predictions based on both growth models. **Performance-Based Growth** is more dynamic as it accounts for individual performance ratings, while the **Fixed Growth Increment** offers a simpler, uniform approach.
- The **Linear Regression** model successfully identifies the correlation between **Performance Rating** and the **future salary** prediction under the performance-based model.

---

## Part 3: Salary Prediction using Regression Models

---

### Objective

Develop multiple regression models to predict employee **future salaries** using various features from the HR dataset. The focus is on comparing prediction accuracy between **fixed growth** and **performance-based growth** approaches.

---

### Dataset

- **Base Dataset:** `dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Simulated Salary Columns:** Augmented in Part 2 for fixed and performance-based salaries.
- **Target Column:** `PredictedFutureSalary`

---

### Models Used

- **Random Forest Regressor**
- **Lasso Regression**
- **Ridge Regression**
- **Support Vector Regressor (SVR)**:
  - RBF Kernel
  - Linear Kernel
  - Polynomial Kernel

---

### Evaluation Metrics

- **R² Score**  
- **RMSE** (Root Mean Square Error)  
- **MAPE** (Mean Absolute Percentage Error)

---

### Model Performance

- For **performance-based salary growth**, **SVR with linear kernel** performed the best.  
- For **fixed salary growth**, **Random Forest Regressor** delivered the most accurate results.

---

### Output Files

- Model plots (bar charts) saved in `images/part3/`
- Predictions saved for both growth approaches
- Code location: `src/part3/`

---

### Python Libraries Used

- **pandas**
- **numpy**
- **scikit-learn**
- **matplotlib**
- **seaborn**

---

## Part 4: Stay Probability Thresholding & Visualization

---

### Objective

Select and apply the optimal stay/leave decision threshold using probabilities from Part 1, then visualize the results.

---

### Dataset

* `src/part1/part1lg_output.txt`: `EmployeeIndex`, `Attrition_Probability`
* `dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv`: `Attrition` (Yes/No)

---

### Preprocessing Steps

1. Load probabilities and compute `P_stay = 1 - Attrition_Probability`.
2. Map `Attrition` → `y_true` (1=stayed, 0=left) for matching indices.

---

### Threshold Selection Method

* Sweep thresholds 0.00–1.00 (0.01 step).
* Compute accuracy, precision, recall, F1 at each step.
* Choose threshold maximizing F1-score.

---

### Performance

Best F1 threshold: `th = 0.57` (F1=0.939, Precision=0.916, Recall=0.964)

---

### Evaluation Metrics

* Accuracy, Precision, Recall, F1-Score

---

### Python Libraries Used

* pandas, numpy
* scikit-learn (precision\_score, recall\_score, f1\_score, accuracy\_score)
* matplotlib

## Part 5: Expected Salary Loss Calculation

---

### Objective

Compute per-employee expected salary loss based on stay probabilities (Part 4) and two salary prediction methods (fixed increment and performance-based).

---

### Dataset & Inputs

* **Stay Probabilities:** `src/part4/likely_to_stay_salaries.csv` (columns: `EmployeeIndex`, `P_stay`)
* **Fixed-Increment Predictions:** `src/part3/RandomForest/Predicted_FutureSalary_FromIncrement.csv`
* **Performance-Based Predictions:** `src/part3/SVR/Linear/performancebased.csv`

---

### Preprocessing Steps

1. Load stay probabilities into `df_prob`.
2. Read both salary prediction DataFrames (`df_fixed`, `df_perf`).
3. Detect salary columns via keyword matching (`increment`, `perf`).

---

### Expected Loss Computation

1. Merge `df_prob` with each salary DataFrame on `EmployeeIndex`.
2. Compute `P_leave = 1 - P_stay` and `ExpectedLoss = P_leave * PredictedSalary`.
3. Sum `ExpectedLoss` to obtain total expected loss per method.

---

### Output

* **Console:**

  * Detected salary column names.
  * Total expected loss for fixed and performance approaches (in ₹).
* **CSV Files:**

  * `expected_loss_fixed.csv`
  * `expected_loss_performance.csv`

---

## Python Libraries Used

* pandas
* pathlib
* sys
