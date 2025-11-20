
# 📘 **Credit Card Fraud Detection — Machine Learning Project**

This project implements a complete **end-to-end Credit Card Fraud Detection system** using the well-known Kaggle dataset.
It includes Exploratory Data Analysis (EDA), preprocessing, class imbalance handling, model training, evaluation, and a modular project structure suitable for scaling.

---

## 🚀 **Project Overview**

The goal is to build a machine learning pipeline that can classify whether a credit card transaction is **fraudulent** or **legitimate**, using supervised learning algorithms.

This project uses multiple baseline models:

* **Logistic Regression**
* **Random Forest Classifier**
* **XGBoost Classifier**

Fraud datasets are highly imbalanced, so techniques like **SMOTE oversampling**, **scaling**, and **stratified splits** are used.

---

## 📂 **Project Structure**

```
creditcard-fraud-detection/
│
├── data/
│   ├── raw/                # raw dataset (ignored by git)
│   └── processed/          # processed data files
│
├── models/                 # saved ML models (.joblib)
│
├── notebooks/
│   ├── 1_EDA.ipynb         # exploratory data analysis
│   ├── 2_Preprocessing.ipynb
│   └── 3_Modeling.ipynb
│
├── src/
│   ├── data_processing.py  # loading, scaling, SMOTE, splits
│   ├── train.py            # trains baseline models
│   └── evaluate.py         # evaluates saved models
│
├── utils/
│   └── plot_utils.py       # custom plotting helpers
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📊 **Dataset**

We use the popular **Credit Card Fraud Detection Dataset** from Kaggle.

* **Total Samples:** 284,807
* **Fraud Cases:** 492
* **Imbalance:** ~0.17% fraud
* **Features:**

  * Time, Amount
  * Features `V1`–`V28` (PCA-transformed for confidentiality)
  * Label: `Class` (1 = Fraud, 0 = Legit)

### ⚠️ Dataset not included in this repo

Due to Kaggle licensing and GitHub's 100 MB limit, the CSV is not included.

**Download it manually from Kaggle**, then place it here:

```
data/raw/creditcard.csv
```

---

## 🔧 **Installation & Setup**

### 1️⃣ Create environment (optional)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare dataset

Download from Kaggle → Move to:

```
data/raw/creditcard.csv
```

---

## 🧪 **How to Run the Project**

### 📘 **Run Notebooks**

Open Jupyter and run:

* `1_EDA.ipynb`
* `2_Preprocessing.ipynb`
* `3_Modeling.ipynb`

### 🛠 **Train Baseline Models (Script Version)**

```bash
python src/train.py
```

### 📈 **Evaluate Models**

```bash
python src/evaluate.py
```

Models will be saved automatically to:

```
models/
```

---

## 📉 **Techniques Used**

### 🔹 **Data Preprocessing**

* StandardScaler
* SMOTE oversampling
* Train/Test split with stratification

### 🔹 **Models**

* Logistic Regression
* Random Forest
* XGBoost

### 🔹 **Evaluation Metrics**

* Precision
* Recall
* F1-Score
* ROC-AUC

*Recall is especially important due to class imbalance (frauds are rare but critical).*

---

## 🔮 **Future Improvements**

* Add SHAP interpretability
* Add LightGBM / CatBoost
* Build FastAPI inference API
* Deploy on Render / HuggingFace Spaces
* Hyperparameter optimization (Optuna)
* Add MLflow tracking

---

## 🤝 **Contributing**

Feel free to open issues or submit pull requests to improve the project!

---

## 📜 **License**

This project is open-source under the **MIT License**.
