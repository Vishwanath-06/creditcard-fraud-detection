
---
# 📘 **Credit Card Fraud Detection — End-to-End Machine Learning Pipeline**

This project implements a complete **fraud detection pipeline** using the Kaggle Credit Card Fraud dataset.
It includes:

✔ Modular pipeline design
✔ Preprocessing + SMOTE
✔ Multiple ML models (LR, Random Forest, XGBoost)
✔ Training & evaluation CLI (`main.py`)
---

## 🚀 **Project Workflow**

The entire ML pipeline is built as a **Python package (`src/`)** and is operated through a single entry point:

```
python main.py --train
python main.py --evaluate
```

### Pipeline Flow:

1. **Load dataset**
2. **Preprocess**

   * Scaling
   * Train/test split
   * SMOTE oversampling
3. **Train multiple models**
4. **Save trained models**
5. **Evaluate models on holdout test set**
6. **Print full classification report + AUC**

---

## 📂 **Project Structure**

```
creditcard-fraud-detection/
│
├── main.py                   # pipeline entrypoint
│
├── src/                      # ML pipeline package
│   ├── __init__.py
│   ├── config.py             # data paths, model paths
│   ├── data_loader.py        # load raw data
│   ├── preprocessor.py       # scaling, split, SMOTE
│   ├── models.py             # model definitions
│   └── pipeline.py           # full ML workflow
│
├── data/
│   ├── raw/                  # place creditcard.csv here
│   └── processed/
│
├── models/                   # saved trained models
│
├── notebooks/                # EDA, preprocessing, modeling
│   ├── 1_EDA.ipynb
│   ├── 2_Preprocessing.ipynb
│   └── 3_Modeling.ipynb
│
├── utils/                    # helper functions for notebooks
│   └── plot_utils.py
│
├── README.md
└── requirements.txt
```

---

## 📊 **Dataset**

We use the popular **Kaggle Credit Card Fraud Detection dataset**:

* **Total transactions:** 284,807
* **Fraud cases:** 492 (0.17%)
* **Features:**

  * `Time`, `Amount`
  * PCA-transformed features `V1`–`V28`
  * `Class` → 1 = Fraud, 0 = Legit

Place the dataset manually at:

```
data/raw/creditcard.csv
```

---

## 🔧 **Installation & Setup**

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Place dataset

```
data/raw/creditcard.csv
```

### 3️⃣ Run training

```
python main.py --train
```

### 4️⃣ Run evaluation

```
python main.py --evaluate
```

---

# 📈 **Model Performance**

After training, here are the real evaluation results from this pipeline:

### **Logistic Regression**

* Precision (fraud): **0.06**
* Recall (fraud): **0.92**
* AUC: **0.97**
  👉 High recall, very low precision (lots of false alarms)

---

### **Random Forest**

* Precision (fraud): **0.86**
* Recall (fraud): **0.83**
* F1: **0.84**
* AUC: **0.977**
  👉 **Best overall model** — balanced precision & recall

---

### **XGBoost**

* Precision (fraud): **0.30**
* Recall (fraud): **0.86**
* AUC: **0.977**
  👉 Strong recall, but too many false positives compared to RF

---

## 🏆 **Best Model: Random Forest**

Excellent precision–recall balance and high AUC.

---

# 🔮 **Future Improvements**

✓ Add SHAP feature interpretability
✓ Hyperparameter tuning (Optuna)
✓ Use class weights instead of SMOTE
✓ Build a real-time fraud detection API (FastAPI)
✓ Add confusion matrix & ROC curve visualizations

---

# 🤝 **Contributions**

Feel free to open issues or submit PRs!

---