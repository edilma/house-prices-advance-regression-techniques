# House Prices: Advanced Regression Techniques

A Kaggle competition project that predicts house sale prices using advanced regression techniques.

🏆 **Competition**: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Overview

This project applies machine learning to predict the final price of residential homes in Ames, Iowa based on 79 explanatory variables describing almost every aspect of the properties.

## Project Structure

```
house-prices-advance-regression-techniques/
├── data/
│   ├── train.csv              # Training dataset (download from Kaggle)
│   ├── test.csv               # Test dataset (download from Kaggle)
│   ├── data_description.txt   # Feature descriptions (download from Kaggle)
│   └── sample_submission.csv  # Submission format (download from Kaggle)
├── notebooks/
│   └── house_prices_analysis.ipynb  # Main analysis notebook
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/edilma/house-prices-advance-regression-techniques.git
cd house-prices-advance-regression-techniques
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the data

Download the competition data from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place all files inside the `data/` directory.

Alternatively, use the Kaggle CLI:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/
cd data && unzip house-prices-advanced-regression-techniques.zip && cd ..
```

### 4. Run the notebook

```bash
jupyter notebook notebooks/house_prices_analysis.ipynb
```

## Notebook Contents

The notebook `house_prices_analysis.ipynb` covers the full machine learning pipeline:

1. **Data Loading & Overview** – Load datasets and inspect shapes, dtypes, and sample rows.
2. **Exploratory Data Analysis (EDA)** – Distribution of the target variable, missing values heatmap, correlation analysis, and key feature visualisations.
3. **Data Preprocessing & Feature Engineering** – Imputation strategies for numerical and categorical features, encoding, skewness correction, and new derived features.
4. **Model Training** – Ridge Regression, Lasso Regression, ElasticNet, Gradient Boosting, XGBoost, and LightGBM trained with cross-validation.
5. **Model Evaluation** – RMSE comparison across models, and a stacking / blending ensemble.
6. **Submission** – Generate the `submission.csv` file ready for Kaggle upload.

## Dependencies

See `requirements.txt` for the full list. Key libraries:

- `pandas`, `numpy` – data manipulation
- `matplotlib`, `seaborn` – visualisation
- `scikit-learn` – preprocessing and models
- `xgboost`, `lightgbm` – gradient boosting
- `scipy` – statistical transformations

## Results

| Model | CV RMSE (log) |
|-------|--------------|
| Ridge | ~0.1150 |
| Lasso | ~0.1120 |
| ElasticNet | ~0.1115 |
| Gradient Boosting | ~0.1200 |
| XGBoost | ~0.1180 |
| LightGBM | ~0.1160 |
| **Ensemble (blend)** | **~0.1100** |

> Results are approximate and will vary depending on random seeds and hyperparameter tuning.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
