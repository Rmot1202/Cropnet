# CropNet — Climate‑Aware Crop Yield Prediction (Student Research Repo)

> **Owner:** Raven Mott (VSU)
> **Status:** work‑in‑progress
> **Keywords:** crop yield, climate change, multimodal data, deep learning, USDA, WRF‑HRRR, Sentinel‑2, Keras

---

## 🚀 Overview

This repository contains my experiments and artifacts for studying **climate‑aware crop yield prediction** using the open‑source **CropNet** dataset family and related methods. It includes data preprocessing utilities, exploratory notebooks, a trained Keras model artifact, and validation metrics produced during my MS‑CC / research work.

The goals of this repo are to:

* Explore relationships between **weather/climate signals** and **county‑level yields**
* Prototype **classical ML** and **deep learning** models on derived features
* Document a **reproducible pipeline** from raw data → engineered features → model → evaluation

> If you’re looking for the *official* dataset and model code, see **Resources** below.

---

## 🗂️ Repository Structure

```
.
├── filter.py                          # Utility to filter/select file lists (e.g., by year or crop)
├── growing_stats_monthly_derived.csv  # Feature table (monthly weather/climate-derived stats)
├── yield_model_validation_metrics.csv # Saved metrics (e.g., MAE/RMSE/R² per crop or fold)
├── multi_yield_regression_model.keras # Trained Keras model artifact
├── Untitled-1.ipynb                   # Notebook: EDA / training / evaluation (WIP)
├── WEEK 1 Deliverable .docx           # Project milestone docs
├── Week 2 Task Sheet.docx             # Project milestone docs
├── Week 3 Task Sheet.docx             # Project milestone docs
├── Evaluating-Climate-Change-Effects-on-Agricultural-Yield-Using-Deep-Learning.pdf  # Project write‑up
└── MSCC (1).pdf                       # Related report/slides
```

---

## 📦 Data Sources (high‑level)

This project builds on the **CropNet** data ecosystem (Sentinel‑2 imagery, WRF‑HRRR computed meteorology, and USDA county‑level yields). In this repo I primarily use **engineered, tabular summaries** (e.g., monthly aggregates) suitable for quick prototyping.

* **growing_stats_monthly_derived.csv** — feature matrix derived from meteorology and season dynamics (e.g., degree days, precipitation, temperature stats, etc.).
* **Target(s)** — county‑level yields for major crops (e.g., corn, cotton, soybean, winter wheat) aligned by year and county FIPS.

> Note: Raw satellite imagery and gridded meteorology are not stored here due to size; see **Resources** for official data access.

---

## 🧰 Environment & Setup

**Python ≥ 3.10** is recommended.

```bash
# (Optional) create a fresh environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Core scientific stack
pip install numpy pandas scikit-learn matplotlib jupyter

# Deep learning (choose one):
pip install tensorflow  # or: pip install torch torchvision
```

If your GPU/driver stack differs, pick versions of TensorFlow/PyTorch compatible with your system.

---

## ▶️ Quickstart (Tabular Pipeline)

The notebook **Untitled-1.ipynb** contains an end‑to‑end example. A minimal Python sketch is below:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# 1) Load engineered features
Xy = pd.read_csv('growing_stats_monthly_derived.csv')

# 2) Select features/target (adjust to your schema)
# Example: numeric weather features → features; 'yield' → target
numeric_cols = [c for c in Xy.columns if c not in ['yield', 'county_fips', 'year', 'crop']]
X = Xy[numeric_cols]
y = Xy['yield']

# 3) Train/validation split by year (or by random split)
train = Xy[Xy['year'] < 2022]
valid = Xy[Xy['year'] == 2022]

X_train, y_train = train[numeric_cols], train['yield']
X_valid, y_valid = valid[numeric_cols], valid['yield']

# 4) Model pipeline (example: RF baseline)
model = Pipeline([
    ("scale", ColumnTransformer([('num', StandardScaler(), numeric_cols)], remainder='drop')),
    ("rf", RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42))
])

model.fit(X_train, y_train)

# 5) Evaluate
pred = model.predict(X_valid)
print('MAE:', mean_absolute_error(y_valid, pred))
print('R^2:', r2_score(y_valid, pred))
```

> Tip: For multi‑crop modeling, treat **crop** as a categorical feature or train **one model per crop**.

---

## 🤖 Deep Learning Sketch (Keras)

If you prefer a neural approach for tabular features (separate from image inputs), here’s a compact baseline:

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

Xy = pd.read_csv('growing_stats_monthly_derived.csv')
num_cols = [c for c in Xy.columns if c not in ['yield', 'county_fips', 'year', 'crop']]
X = Xy[num_cols].astype('float32').values
y = Xy['yield'].astype('float32').values

# simple split; replace with group/temporal split for rigor
mask = Xy['year'] < 2022
X_train, y_train = X[mask], y[mask]
X_valid, y_valid = X[~mask], y[~mask]

inp = keras.Input(shape=(len(num_cols),))
x = layers.Normalization()(inp)
x.adapt = None  # placeholder to indicate that adaptation should be done separately
x = layers.Dense(256, activation='relu')(inp)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1)(x)

model = keras.Model(inp, out)
model.compile(optimizer='adam', loss='mae', metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, batch_size=256, verbose=2)

model.save('multi_yield_regression_model.keras')
```

> For **imagery‑aware** models (e.g., MMST‑ViT), use the official code and loaders (see **Resources**).

---

## 📊 Results & Tracking

* **yield_model_validation_metrics.csv** stores evaluation outputs from prior runs (e.g., MAE/RMSE per crop/year). You can append to this file as you test new models.
* Consider logging with **TensorBoard**, **Weights & Biases**, or a simple CSV appender to keep a clear audit trail.

---

## 🔍 Filtering Utility

`filter.py` provides helpers to select and list dataset files (e.g., restrict to `year == 2022` or to a single crop). Example usage:

```bash
python filter.py --year 2022 --crop soybean --write filtered_2022_files.txt
```

(Inspect the script for supported flags; adapt paths to your local dataset layout.)

---

## 🧪 Reproducibility Tips

* Use **temporal splits** (e.g., train on 2017–2021, validate on 2022) to avoid leakage.
* Stratify by **county/crop** when appropriate.
* Standardize units (°C vs °F, mm vs inches) before feature engineering.
* When using imagery, align county polygons to the same spatial grid as meteorology.

---

## 🗺️ Roadmap

* [ ] Clean notebook into `/notebooks/` with clear, executable cells
* [ ] Add `/src/` Python package with CLI for train/eval
* [ ] Expand feature engineering (GDD, drought indices, lagged features)
* [ ] Add cross‑validation by year and county group
* [ ] Add baselines (Linear, XGBoost/LightGBM) + DL (tabular MLP)
* [ ] (Stretch) Integrate imagery + meteorology with a multi‑modal backbone

---

## 📚 Resources (official repos & papers)

* CropNet Dataset (KDD 2024) — dataset, loaders, and tutorials
* MMST‑ViT (ICCV 2023) — multi‑modal spatial‑temporal ViT for yield prediction
* USDA QuickStats — county‑level yields and production

> These resources provide the canonical data definitions and end‑to‑end examples for imagery + meteorology workflows.

---

## 🤝 Acknowledgments

Thanks to collaborators, mentors, and programs that supported this work (VSU, MS‑CC, advisors/mentors). The **CropNet** dataset creators and the **MMST‑ViT** authors deserve special credit; please cite their work when appropriate.

---

## 📄 Citation (example)

If this repository, figures, or code snippets are helpful in your research, please cite:

```
Mott, R. (2025). CropNet — Climate‑Aware Crop Yield Prediction (Student Research Repo). GitHub repository. https://github.com/Rmot1202/Cropnet
```

For the underlying dataset/model, please cite the original **CropNet (KDD 2024)** and **MMST‑ViT (ICCV 2023)** papers.

---

## 📜 License

No license has been specified yet. By default, this means **all rights reserved**. If you intend others to reuse this work, please add a LICENSE file (e.g., MIT for code + note dataset license).
