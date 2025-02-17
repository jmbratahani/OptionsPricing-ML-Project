# OptionsPricing-ML-Project
A machine learning project that predicts European call option prices (regression task) and classifies Black-Scholes (BS) performance (over vs. underestimation) for the S&P 500. Various supervised learning methods (Linear Regression, Lasso, Ridge, K-Nearest Neighbors, Decision Trees, Random Forests, Gradient Boosting, SVM) are explored, with Random Forest emerging as the top performer in both regression and classification tasks.


## Project Overview

Options pricing is critical for risk management and investment decisions. Traditional methods like Black-Scholes (BS) assume market conditions remain relatively stable, but real-world markets are highly dynamic. This repository applies modern machine learning techniques to predict:

- **Regression Task:** Predict the current option value of European call options (Continuous target: `Value`).  
- **Classification Task:** Classify whether the BS model *over* or *under* estimates the option price (Binary target: `BS` → {Over, Under}).

Our best-performing models leverage Random Forest, offering strong predictive power while capturing complex, non-linear interactions among parameters.


## Data

- **Training File:** `option_train.csv`
  - **Columns:** 
    - `Value` (float)  
    - `S` (float) - Current underlying asset value  
    - `K` (int) - Strike price  
    - `tau` (float) - Time to maturity  
    - `r` (float) - Risk-free interest rate  
    - `BS` (string) - Indicator of whether Black-Scholes is “Under” or “Over”  

**Preprocessing**  
- Check for missing values and duplicates.  
- Scale feature variables (`S`, `K`, `tau`, `r`) when beneficial (e.g., for KNN, SVM, etc.).  


## Approaches

1. **Linear Models**  
   - **Linear Regression:** Baseline approach for option-value prediction (regression).  
   - **Logistic Regression:** Baseline approach for BS classification.

2. **Regularization Techniques**  
   - **Lasso (L1)** and **Ridge (L2)** regression to handle multicollinearity and simplify models.

3. **Tree-Based Methods**  
   - **Decision Trees:** Easily interpretable, used for both regression and classification.  
   - **Random Forests (RF):** Ensembles of decision trees, typically more robust.  
   - **Gradient Boosting:** Iterative boosting approach that can outperform simple ensembles in certain cases.

4. **K-Nearest Neighbors (KNN)**  
   - Simple distance-based method, sensitive to feature scaling.

5. **Support Vector Machines (SVM)**  
   - Explored for classification with various kernels.

**Key Observations**  
- **Random Forest** yielded the highest R-squared for regression (~0.996) and best classification accuracy (~93.66%).  
- Feature scaling improved some models slightly (e.g., KNN, SVM) but not all.  

**Model Training & Evaluation**  
   - Use the provided Jupyter notebooks (`CV_Regression_Models.ipynb`, `Linear_Regression.ipynb`, etc.) to:
     - Perform K-fold cross-validation.  
     - Compare multiple algorithms (Linear Regression, Random Forest, etc.).  
     - Tune hyperparameters (e.g., via `GridSearchCV`).


## Results

1. **Regression (Predicting `Value`)**  
   - **Best Model:** RandomForestRegressor (base configuration or slight tuning).  
   - **Mean R-squared (5-fold CV):** ~0.996  
   - **Mean MSE (5-fold CV):** ~54  

2. **Classification (Predicting `BS`)**  
   - **Best Model:** RandomForestClassifier (`n_estimators=200`).  
   - **Accuracy (10-fold CV):** ~93.66%  

3. **Additional Insights**  
   - Including or excluding interest rate `r` had minimal effect on linear regression.  
   - Feature scaling was especially helpful for KNN and SVM.  
   - Random Forest was robust across varying parameters, outperforming simpler methods.

## Contributors

**DSO 530: Applied Modern Statistical Learning Methods**  at **USC Marshall School of Business**
- Jessica Bratahani  
- Pin Hsuan Chang  
- Suhan Ho  
- Sheena Huang  
- Yunchi Lee  


**Disclaimer**  
The models and code provided are for educational purposes only. Investment decisions should integrate robust risk management practices, expert insights, and quantitative financial analysis.
