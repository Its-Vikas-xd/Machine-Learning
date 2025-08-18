<!-- Banner -->
<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Repo-blueviolet?style=for-the-badge" alt="ML Repo"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/License-MIT-informational?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🧠 Machine Learning – End-to-End Notes, Code & Roadmap</h1>

<p align="center">
  A curated collection of notebooks, notes, and mini-projects covering the ML journey from data cleaning to model evaluation.<br/>
  Built by <a href="https://github.com/Its-Vikas-xd">Vikas</a> with ❤️
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-folder-structure">Structure</a> •
  <a href="#-setup">Setup</a> •
  <a href="#-roadmap">Roadmap</a> •
  <a href="#-learning-path">Learning Path</a> •
  <a href="#-contributing">Contributing</a> •
  <a href="#-license">License</a>
</p>

---

## ✨ Features

- 📓 Clean, commented **Jupyter notebooks** for every topic
- 🧹 **Data Cleaning**: missing values, outliers, duplicates, dtype fixes
- 🔤 **Encoding**: Label, One-Hot, Ordinal
- 📏 **Scaling**: Normalization & Standardization
- 📈 **Regression**: Simple, Multiple, Polynomial, Cost functions, R²/Adjusted R²
- 🧩 **Regularization**: L1 (Lasso), L2 (Ridge), Elastic Net
- 🧮 **Classification**: Logistic (binary & multiclass), OvR vs Multinomial, Polynomial input LR
- 🧪 **Model evaluation**: Confusion matrix, Precision/Recall/F1, ROC-AUC
- 🧱 **Pipelines & FunctionTransformer** examples
- 📊 Visuals with Matplotlib & Seaborn

---
Machine-Learning/
├─ Data Preprocessing/
│ ├─ Handling Missing Data/
│ ├─ Outliers & Duplicates/
│ ├─ Encoding (Label, OneHot, Ordinal)/
│ └─ Scaling (Standardization, Normalization)/
├─ Supervised Machine Learning/
│ ├─ Linear Regression/
│ │ ├─ Simple Linear/
│ │ ├─ Multiple Linear/
│ │ └─ Polynomial Regression/
│ ├─ Regularization/
│ │ ├─ L1_Lasso/
│ │ ├─ L2_Ridge/
│ │ └─ ElasticNet/
│ └─ Classification/
│ ├─ Logistic Regression (Binary)/
│ ├─ Polynomial Input Logistic Regression/
│ └─ Multiclass (OvR vs Multinomial)/
├─ Utils/
│ ├─ plotting.py
│ └─ preprocessing.py
├─ requirements.txt
└─ README.md

> 🔗 Example notebook links (update paths as needed):
>
> - Multiple Linear Regression: `Supervised Machine Learning/Linear Regression/Multiple Liner Alogrithm/Multiple_Liner.ipynb`  
> - Polynomial Input Logistic Regression: `Supervised Machine Learning/Classification/Binary Classification/Polynomial input Logistic Regression/`  
> - Logistic Regression (Binary): `Supervised Machine Learning/Classification/Binary Classification/Logistic Regression/logistic_Regression.ipynb`

---

## ⚙️ Setup

```bash
# 1) Clone
git clone https://github.com/Its-Vikas-xd/Machine-Learning.git
cd Machine-Learning

# 2) (Optional) Create a virtual env
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Install deps
pip install -r requirements.txt

# 4) Run notebooks
jupyter notebook

## 📂 Folder Structure

