# 🏦 Credit Risk Classification Pipeline

## 📌 Project Overview
This project builds an end-to-end Machine Learning pipeline to predict credit risk (loan defaulters). It was completed as a strict **"No-AI / No-Copilot Challenge"** to solidify foundational skills in data preprocessing, pipeline architecture, and hyperparameter tuning using `scikit-learn`.

## 🛠️ Tech Stack & Architecture
* **Language:** Python
* **Libraries:** pandas, seaborn, matplotlib, scikit-learn
* **Pipeline Structure:**
  1. `SimpleImputer` (Median strategy for missing values)
  2. `StandardScaler` (Feature scaling)
  3. `RandomForestClassifier` (Ensemble modeling)

## 🚀 Key Highlights
* **Categorical Encoding:** Successfully transformed text features into numerical data using `pd.get_dummies` (avoiding the dummy variable trap).
* **Hyperparameter Tuning:** Utilized `GridSearchCV` to test 72 candidates (216 fits) across parameters like `n_estimators`, `max_depth`, and `min_samples_split`.
* **Imbalanced Data Handling:** Applied class weights and utilized `stratify=y` during train-test splitting to ensure minority classes were represented accurately.

## 📊 Business Impact & Model Performance
Predicting loan defaults requires extreme precision to avoid turning away good customers while identifying actual risks. The tuned Random Forest model achieved:
* **Overall Accuracy:** 93.25%
* **Precision (Defaulters):** **96%** (When the model flags a customer as a risk, it is correct 96% of the time).
* **Recall (Defaulters):** 72%

## 📂 Project Structure
* `data/` - Contains the raw dataset.
* `notebooks/` - Jupyter notebook containing EDA, Pipeline creation, and Evaluation.