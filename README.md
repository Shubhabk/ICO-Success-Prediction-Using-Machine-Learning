# ICO Success Prediction Using Machine Learning

This project applies multiple machine learning classification models to predict the success of Initial Coin Offerings (ICOs). An ICO is considered **successful** if it meets or exceeds its fundraising target.

The goal is to compare different models and identify which approach best predicts ICO success, while also highlighting the most influential factors behind successful token sales.

---

## Project Overview

Initial Coin Offerings (ICOs) are a popular but risky fundraising mechanism in the cryptocurrency ecosystem. Investors often lack reliable tools to evaluate whether a project is likely to succeed.

In this project, historical ICO data is used to:

* Clean and preprocess financial and project-related features
* Handle extensive missing values
* Engineer meaningful features such as ICO duration
* Train and compare multiple classification models
* Evaluate model performance using accuracy, ROC–AUC, sensitivity, and specificity

---

## Dataset

* **Observations:** 6,146 ICOs
* **Features:** 25 (numerical, categorical, and binary)
* **Target variable:** `success`

  * `1` = Successful ICO
  * `0` = Unsuccessful ICO

### Key Features

* Token price (standardised to USD)
* Tokens sold and tokens available for sale
* Team size
* ICO rating
* Whitelist and KYC requirements
* ERC20 token standard
* ICO and pre-ICO duration
* Country and accepted currencies

---

## Data Preprocessing

### Missing Values

* **Numerical features:** Median imputation
* **Categorical features:** Filled with `"Unknown"`
* **Binary features:** Encoded as 0/1, missing values treated as absence

### Feature Engineering

* Standardised token prices across multiple currencies
* Created:

  * `ico_duration`
  * `pre_ico_duration`
* Applied **z-score normalization** to numerical features

---

## Models Implemented

The following classification models were trained and evaluated:

* **Naive Bayes**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machine (SVM)**
* **Random Forest**

Hyperparameter tuning was performed using **GridSearchCV** where applicable.

---

## Model Evaluation

Models were evaluated using:

* Classification reports
* Confusion matrices
* ROC curves and AUC
* Sensitivity (Recall for successful ICOs)
* Specificity (Correctly identifying failures)
* Cross-validation accuracy

### Best Performing Model

**Random Forest** achieved the best overall balance:

* Accuracy: **0.72**
* ROC–AUC: **0.77**
* Sensitivity: **0.29**
* Specificity: **0.94**

It performed particularly well at filtering out unsuccessful ICOs, making it useful for investor risk screening.

---

## Key Insights

* ICO success is highly **imbalanced**, with more failures than successes
* Features such as:

  * ICO rating
  * Team size
  * Token distribution
    have noticeable influence on success probability
* Naive Bayes heavily over-predicts success
* Ensemble models (Random Forest) outperform simpler classifiers

---

## Limitations & Future Work

* Strong class imbalance affects sensitivity
* High missingness in some features (e.g. `sold_tokens`)
* Median imputation may introduce bias

Possible improvements:

* SMOTE or other resampling techniques
* Advanced imputation methods
* Inclusion of social media or sentiment analysis features
* Trying gradient boosting models (XGBoost, LightGBM)

---

## Tech Stack

* **Python**
* pandas, numpy
* scikit-learn
* matplotlib, seaborn


