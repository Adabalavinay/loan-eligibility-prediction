---

# Loan Eligibility Prediction using Machine Learning

##  Project Overview

This project predicts whether a loan application should be approved or not based on applicant details such as income, loan amount, education, marital status, and other financial indicators.
The system leverages machine learning techniques to automate the loan approval process, reducing manual work for banks and financial institutions.

---

##  Features

* End-to-end *ML pipeline*: data cleaning, preprocessing, feature engineering, model training, and evaluation.
* *Outlier removal* and *missing value handling* for clean data.
* *Feature scaling* using StandardScaler for better model performance.
* *Class imbalance handling* with RandomOverSampler (SMOTE-like oversampling).
* Trained with *Support Vector Machine (SVM, RBF kernel)* for binary classification.
* Evaluated with *ROC-AUC, Confusion Matrix, Precision, Recall, and F1-score*.

---

##  Tech Stack

* *Programming Language*: Python
* *Libraries*: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn

---

##  Dataset

The dataset (loan_data.csv) contains features such as:

* ApplicantIncome, CoapplicantIncome
* LoanAmount, Loan\_Amount\_Term
* Gender, Married, Dependents, Education, Self\_Employed
* Credit\_History
* Loan\_Status (Target Variable: Y/N)

---

##  Implementation Steps

1. *Data Loading & Exploration* – Load CSV, check distributions, visualize features.
2. *Data Preprocessing* – Handle missing values, encode categorical variables.
3. *Outlier Removal* – Remove extreme values in income & loan amount.
4. *Feature Scaling* – Apply StandardScaler.
5. *Handling Imbalance* – Use RandomOverSampler to balance classes.
6. *Model Training* – Train an SVM classifier with RBF kernel.
7. *Evaluation* – Compute ROC-AUC, Confusion Matrix, and Classification Report.

---

##  Results

* *Training ROC-AUC Score*: \~1.0
* *Validation ROC-AUC Score*: \~0.80
* Classification report showed *balanced precision, recall, and F1-score* across loan approval classes.

---

##  Future Improvements

* Test additional models (Logistic Regression, Random Forest, XGBoost).
* Hyperparameter tuning for SVM to improve validation performance.
* Deploy the model as a web application (Flask/Django + frontend).

---

##  License

This project is for educational purposes and open for further development.

---

⚡ Would you like me to *format this README with some badges (e.g., Python, Scikit-learn, ML)* and a *sample usage code snippet* so it looks more professional on GitHub?
