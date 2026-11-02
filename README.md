# 🏎️ AutoMax — Enterprise Vehicle Fuel Economy (MPG) Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoMax is an interactive machine learning dashboard designed to predict automotive fuel efficiency (Miles Per Gallon) based on key structural and mechanical features. The web portal features side-by-side inference across multiple regularized regression algorithms (**Linear**, **Ridge**, **Lasso**) and dynamically adapts its UI/visualization theme according to performance predictions.

---

## 🎯 Key Capabilities

* **Multi-Algorithm Model Comparison:** Evaluates input features simultaneously across Ordinary Least Squares (OLS) Linear Regression, Ridge ($L_2$), and Lasso ($L_1$) GridSearch models.
* **Standardized Pipeline Execution:** Guarantees exact feature scaling using a pre-fitted `StandardScaler` pipeline trained on automotive benchmark datasets.
* **Dynamic Automotive Categorization:** Real-time visual layout adaptation matching the vehicle profile (**Eco Compact**, **Executive Sedan**, or **V8 Performance Muscle**) based on output MPG thresholds.
* **Modern Store Dashboard UI:** Glassmorphism overlay container cards over custom atmospheric vehicle showroom styling built natively with CSS injection.

---

## 📐 System Architecture

```text
  [ User Parameters ] ──> ( Streamlit Sidebar Input )
                                     │
                                     ▼
                          [ pandas DataFrame ]
                                     │
                                     ▼
                          ( StandardScaler Pipeline )
                                     │
                                     ▼
                        [ Scaled Feature Matrix ]
                                     │
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
    ( Linear Regression )    ( Ridge Regressor )     ( Lasso Regressor )
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     ▼
                       [ Model Inference Metrics ]
                                     │
                                     ▼
               ( Dynamic Category Match & Visual Rendering )
