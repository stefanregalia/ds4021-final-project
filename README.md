# DS 4021 Final Project: Climate Temperature Prediction
By Teagan Britten, Alka Link, and Stefan Regalia

This project applies machine learning models to predict average temperature using climate-related features. We compare penalized linear regression, support vector machines, random forests, and neural networks.

## Repository Contents

This repository contains all notebooks, data, and outputs for our final project analysis.

## Software and Platform

### Software Used
- **Python**: 3.12.5
- **Jupyter Notebooks**

### Packages
- `pandas` - data manipulation
- `numpy` - numerical operations
- `matplotlib` - plotting
- `seaborn` - statistical visualizations
- `scikit-learn` - machine learning models and utilities
  - `StandardScaler`, `Pipeline`, `GridSearchCV`, `KFold`
  - `Ridge`, `Lasso`, `ElasticNet` (penalized regression)
  - `SVR` (support vector regression)
  - `RandomForestRegressor` (ensemble)
  - `PCA` (dimensionality reduction)
  - `train_test_split`, `cross_val_score`
  - `mean_squared_error`, `r2_score`, `mean_absolute_error`
- `torch` (PyTorch) - neural network implementation
  - `torch.nn`, `torch.optim`

### Operating System
- Developed on macOS & Windows

## Documentation Map

```
ds4021-final-project/
│
├── README.md                        # Project overview 
│
├── final_project.pdf                # Final report
│
├── data/
│   ├── climate_change_dataset.csv   # Original dataset
│   ├── train_set_X.csv              # Training features
│   ├── train_set_y.csv              # Training target
│   ├── test_set_X.csv               # Test features (held out)
│   └── test_set_y.csv               # Test target (held out)
│
├── notebooks/
│   ├── summary-info.ipynb           # Exploratory data analysis
│   ├── penalized-linreg.ipynb       # Ridge, Lasso, ElasticNet regression
│   ├── svm.ipynb                    # Support Vector Machine
│   ├── random-forest.ipynb          # Random Forest ensemble
│   ├── neural-network.ipynb         # PyTorch neural network
│   └── test-evaluation.ipynb        # Final model evaluation on test set
│
└── output/
    ├── summary_statistics.csv               # Descriptive statistics
    ├── column_names.csv                     # Feature names
    ├── climate_variable_distributions.png   # Distribution plots
    ├── climate_variable_pairplot.png        # Pairplot visualization
    ├── correlation_heatmap.png              # Feature correlations
    ├── final_test_predictions.png           # Final model test set results
    │
    └── penalizedlinreg-outputs/             # Penalized regression results
        ├── best_model_coefficients.png
        └── cv_results_summary.csv
```

