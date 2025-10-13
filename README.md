## Diabetes Predictor (Female)

A complete, end-to-end machine learning pipeline to predict diabetes in female patients using ensemble learning. This repository includes data preprocessing (imputation, outlier handling, scaling), model training with a soft-voting ensemble, evaluation, and saved artifacts for reproducible inference.

### Why this project?
Early detection helps with timely intervention. This project demonstrates a clean, reproducible ML workflow built with scikit-learn that you can run locally and extend.

---

## Features at a Glance
- **Data preprocessing**: Median imputation for biologically impossible zeros, 3-sigma outlier capping, train/test split with stratification, standardization.
- **Ensemble model**: Soft VotingClassifier combining RandomForest, KNN, and SVM (RBF) for robust performance.
- **Reproducibility**: All key transformers and the trained ensemble are saved as `.pkl` artifacts; intermediate datasets are exported as `.csv`.
- **Clear artifacts**: `imputer.pkl`, `scaler.pkl`, `feature_columns.pkl`, `voting_classififer.pkl` (intentionally kept file name as generated), and split/scaled CSVs.

---

## Repository Structure

```
/diabetes_predictor
├─ dataset.xlsx                               # Raw dataset (PIMA-like schema expected)
├─ preprocessing.py                           # Preprocessing pipeline script
├─ model.py                                   # Model training and evaluation script
├─ preprocessing.ipynb                        # Notebook version of preprocessing
├─ Model.ipynb                                # Notebook version of modeling
├─ preprocessed_dataset_without_scaling.csv   # After imputation (before scaling)
├─ outlier_refined.csv                        # After 3σ outlier capping
├─ X_train_scaled.csv                         # Scaled training features
├─ X_test_scaled.csv                          # Scaled test features
├─ y_train.csv                                # Training labels
├─ y_test.csv                                 # Test labels
├─ imputer.pkl                                # Fitted SimpleImputer (median)
├─ scaler.pkl                                 # Fitted StandardScaler
├─ feature_columns.pkl                        # Feature name list used in training
├─ voting_classififer.pkl                     # Trained soft-voting ensemble (note filename)
└─ README.md
```

---

## Data & Features
The project expects a dataset similar to the PIMA Indians Diabetes dataset, with columns typically including:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target: 0 = no diabetes, 1 = diabetes)

If your schema differs, update `preprocessing.py` accordingly and regenerate artifacts.

---

## Preprocessing Pipeline (`preprocessing.py`)
1. **Load data** from `dataset.xlsx`.
2. **Treat zeros as missing** in selected biometric columns by replacing `0` with `NaN`, then apply median imputation via `SimpleImputer(strategy="median")`.
   - Columns handled in code: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`.
3. **Save imputer** to `imputer.pkl` for reuse during inference.
4. **Export** an intermediate dataset `preprocessed_dataset_without_scaling.csv`.
5. **Outlier handling**: 3-sigma capping per feature (exclude the target) and export to `outlier_refined.csv`.
6. **Split** into train/test with `stratify=y`, `test_size=0.2`, `random_state=42`.
7. **Scale features** using `StandardScaler` (fit on train, transform test).
   - Save `X_train_scaled.csv`, `X_test_scaled.csv`, `y_train.csv`, `y_test.csv`.
   - Save `scaler.pkl` and `feature_columns.pkl`.

Run the preprocessing end-to-end:

```bash
python preprocessing.py
```

Artifacts generated will be used by the model training stage and for inference.

---

## Modeling (`model.py`)
The project trains a soft-voting ensemble on the scaled features.

- Base learners:
  - RandomForestClassifier (n_estimators=100, random_state=42)
  - KNeighborsClassifier (n_neighbors=5)
  - SVC (RBF kernel, probability=True, random_state=42)
- Combiner: `VotingClassifier(..., voting='soft')`
- Input: `X_train_scaled.csv`, `X_test_scaled.csv`, `y_train.csv`, `y_test.csv`
- Output: Classification metrics + `voting_classififer.pkl`

Train and evaluate:

```bash
python model.py
```

Expected console output includes test accuracy, a classification report, and a confusion matrix.

---

## Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -U pip
pip install pandas numpy scikit-learn openpyxl
```

### 2) Preprocess
```bash
python preprocessing.py
```

### 3) Train
```bash
python model.py
```

### 4) Use the Model for Inference
The snippet below shows how to load the saved artifacts and predict for new samples. Ensure your input uses the exact `feature_columns.pkl` order.

```python
import pickle
import pandas as pd

# Load artifacts
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("imputer.pkl", "rb") as f:
    imputer = pickle.load(f)
with open("voting_classififer.pkl", "rb") as f:  # note filename
    model = pickle.load(f)

# Example: a single new patient (replace with real values)
raw_sample = {
    "Pregnancies": 2,
    "Glucose": 140,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 100,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35,
}

# Create DataFrame in the exact training feature order
X_new = pd.DataFrame([raw_sample])[feature_columns]

# Match training-time preprocessing
X_new["Glucose"] = X_new["Glucose"].replace(0, pd.NA)
X_new["BloodPressure"] = X_new["BloodPressure"].replace(0, pd.NA)
X_new["SkinThickness"] = X_new["SkinThickness"].replace(0, pd.NA)
X_new["Insulin"] = X_new["Insulin"].replace(0, pd.NA)
X_new["BMI"] = X_new["BMI"].replace(0, pd.NA)
X_new[feature_columns] = imputer.transform(X_new[feature_columns])

X_new_scaled = scaler.transform(X_new)

pred = model.predict(X_new_scaled)[0]
proba = getattr(model, "predict_proba")(X_new_scaled)[0][1]

print({"prediction": int(pred), "probability_diabetes": float(proba)})
```

---

## Reproducibility Notes
- `random_state=42` is used where relevant to stabilize splits and model behavior.
- Keep the artifacts (`imputer.pkl`, `scaler.pkl`, `feature_columns.pkl`) tightly coupled with the model; mixing versions may degrade performance.
- The trained model filename is preserved as `voting_classififer.pkl` to match the generated file in `model.py`.

---

## Evaluation
During training, the following are printed:
- **Test Accuracy**
- **Classification Report**: precision, recall, F1-score per class
- **Confusion Matrix** (via `confusion_matrix` in code; you can extend printing/plotting)

You can compute additional metrics or add cross-validation as needed.

---

## Extending the Project
- Add more algorithms (e.g., Logistic Regression, XGBoost) to the voting ensemble
- Hyperparameter tuning with GridSearchCV/Optuna
- Calibrated probabilities for improved decision thresholds
- Model explainability (e.g., SHAP)
- Export to a simple API (FastAPI/Flask) for real-time predictions

---

## Limitations
- Trained on a specific dataset distribution; may not generalize to all populations.
- Zeros-as-missing assumption is domain-informed; confirm for your data before applying.
- Feature list order must be preserved for correct scaling and prediction.

---

## License
This project is for educational and research purposes. Add a license if you plan to distribute.

---

## Acknowledgments
- Built with `pandas`, `numpy`, and `scikit-learn`.
- Inspired by common pipelines for the PIMA Indians Diabetes dataset.
