# 🏡 House Price Prediction with Linear Regression

This project is a complete implementation of a House Price Prediction system using **Linear Regression**, with full data preprocessing, outlier handling, scaling, encoding, model training, evaluation, saving, and deployment for user input-based prediction.


---

## 🧠 Objective

Predict the **price of a house** based on features like:

- Numerical: `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
- Categorical: `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`

---

## 📦 Requirements

Install the required Python libraries:

```bash
pip install -r requirements.txt
---
## ⚙️ Features & Workflow

### ✅ Data Preprocessing
- Handled missing values (none in dataset)
- Removed duplicates
- Log-transformed skewed numerical features (like `area`)
- Outliers handled using **IQR-based capping**

### ✅ Encoding & Scaling
- Numerical features scaled using **StandardScaler**
- Categorical features encoded using **OneHotEncoder** (`handle_unknown='ignore'`)

### ✅ Model
- Model: **Linear Regression** from `scikit-learn`
- Target variable (`price`) log-transformed for better model performance

### ✅ Evaluation Metrics
- **MAE**, **MSE**, **R² Score**
- Actual vs Predicted **visualization**

### ✅ Model Saving
Saved using `joblib`:
- `scaler.pkl`
- `encoder.pkl`
- `linear_regression_model.pkl`

---

## 🔮 Prediction Script: `prediction.py`

### ▶️ Run the script

```bash
python prediction.py