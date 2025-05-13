# 🏨 Hotel Reservation| End-to-End Data Science Project

Predict whether the customer will **get his booking** or **cancel it** , using a complete data-science pipeline from raw data through final model evaluation

## Project Overview

This notebook walks through a full Data Science workflow on a **hotel bookings reservations for 2017 & 2018 year dataset** :

1. **Data Ingestion & Cleaning**  
   - Load reservation data, handle missing values, drop irrelevant columns.  
   - Recode categorical fields and convert the target (`booking_status`) to binary.

2. **Exploratory Data Analysis (EDA)**  
   - Summary statistics, Distributions, countplots,Correlation heatmap.  
   -  Deep dives into all categorical and numerical columns to gain insights and relationships between different features.
3. **Visualization**    
   - **Interactive Plotly** charts embedded for dynamic exploration of patterns.
   - Static plots via Matplotlib & Seaborn.

4. **Preprocessing & Feature Engineering**
   - Encoding categorical variables.
   - Outlier detection and removal.   
   - Balancing target class `Booking_status` with **SMOTE** technique.  
   - Testing multiple **feature scalers**: StandardScaler, MinMaxScaler, RobustScaler to decide which is the most suitable one for data. 
   - Selecting the **Top 10 highest-impact features** based on feature importance.

6. **Modeling**  
   - Train and compare three classifiers:  
     - Logistic Regression. 
     - Random Forest.
     - Support Vector Machine (SVM).

7. **Hyperparameter Tuning**  
   - Use **GridSearchCV** to find optimal parameters for SVM as it was indicated as the best suitable model for data with highest accuracy after comparing
     its result with the other 2 models : Random Forest, Logistic Regression. 

8. **Final Evaluation & Conclusions**  
   - Compare all pipelines’ accuracies.  
   - **Best result:** SVM + StandardScaler + Top 10 Features : **83% accuracy**

---

## 📈 Plotly Integration

- **Why Plotly?**  
  Plotly enables highly interactive, zoomable charts in Colab, making pattern discovery more intuitive.

- **GitHub Rendering Note:**  
  GitHub’s static notebook preview cannot execute Plotly’s JavaScript. So there's a problem showing plotly in github but to show these plots you can open the notebook in colab to check the wonderful visualization of data. 
 **Open the notebook in Google Colab** using the badge below  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XjVBiI4VFGYlybjtGFmCBlkkZ0cvhjG7#scrollTo=JFKiEg8Ppl23)

---

## 🏆 Key Results

| Pipeline                              | Model         | Accuracy |
|---------------------------------------|---------------|----------|
| **StandardScaler + SVM**                  | **SVM**           | **82.2%**    |
| **Top 10 Features + StandardScaler**  | **SVM**       | **83%**  |
| **SMOTE + any model**                     | **Various**       | **80%**     |
| **GridSearchCV tuning**                   | **SVM**    | **80%** |

> **Conclusion:**  [SVM + StandardScaler + Top 10 Features] ➔  is the best approach that produces **83% Accuracy**.



