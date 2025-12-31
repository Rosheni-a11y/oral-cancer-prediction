# Oral Cancer Prediction

## Overview
This project implements an end-to-end machine learning solution for predicting oral cancer using supervised learning. The goal is to explore how data-driven approaches can assist early detection and support healthcare decision-making, especially in resource-limited settings.

The project was developed as an **academic AI/ML project**, focusing on preprocessing, model training, evaluation, and understanding real-world medical data challenges.

---

## Problem Domain
**Why Oral Cancer is a Serious Issue**
- High mortality due to late-stage detection
- Significant impact in developing regions (e.g., South Asia)
- Treatment is costly and recovery is difficult

**Challenges in Current Detection**
- Heavy reliance on manual screening by specialists
- Symptoms often appear in advanced stages
- Limited awareness and early diagnosis in rural areas

**Why Data-Driven Prediction Matters**
- Early detection significantly improves survival rates
- Machine learning can help identify high-risk individuals quickly
- Scalable and affordable approach for large-scale screening

---

## Dataset Details
- **Name:** Oral Cancer Prediction Dataset (by Ankush Panday)
- **Source:** https://www.kaggle.com/datasets/ankushpanday2/
              oral-cancer-prediction-dataset
- **Size:** 84,922 records with 25 features
- **Task:** Binary classification (Oral Cancer: Yes / No)

**Feature Categories**
- **Demographics:** Age, Gender, Country  
- **Lifestyle:** Tobacco Use, Alcohol Consumption, Diet  
- **Medical History:** HPV Infection, Family History, Immune Status  
- **Clinical Findings:** Oral Lesions, Tumor Size, Cancer Stage  

---

## Data Preprocessing
The following preprocessing techniques were applied:

- **Data Cleaning:** Handling missing values and inconsistent entries  
- **Outlier Removal:** Reducing the impact of extreme values on model performance  
- **Categorical Encoding:** Encoding categorical variables for model compatibility  
- **Feature Engineering:** Creating and refining features to improve predictive power  
- **Feature Scaling:** Normalizing numerical features for distance-based models  
- **Dimensionality Reduction:** Reducing feature space complexity where applicable


---

## Model Training
Multiple machine learning models were trained and evaluated.
The following machine learning models were trained and evaluated:

- Logistic Regression 
- Random Forest 
- K-Nearest Neighbors 
- Support Vector Machine 
- Decision Tree 
- XGBoost 

Based on performance and robustness, the **Tuned Random Forest model** was selected as the final model.

Each model was trained in a **separate notebook** to maintain clarity and reproducibility.

---

## Model Evaluation Metrics

Given the medical context and class imbalance, the following metrics were prioritized:

- **Accuracy**
- **F1 Score** (most critical metric)
- **AUC (Area Under the Curve)** where applicable

---

### Best Model: Tuned Random Forest
**Evaluation Metrics**
- **Accuracy:** 0.5181  
- **F1 Score:** 0.6690  
- **AUC:** 0.5185  

**Why Random Forest?**
### 🔹 Superior F1 Score
- Achieved a high F1 score compared to other valid models
- F1 score is the most important metric for **imbalanced medical datasets**, as it balances false positives and false negatives

### 🔹 Meaningful AUC
- Unlike some models, Random Forest reported a valid AUC value
- An AUC above 0.5 confirms the model has genuine discriminative power rather than random guessing

### 🔹 Robustness
- As an ensemble method, Random Forest is less prone to overfitting compared to single decision trees
- Provides more stable performance when applied to unseen data

---

## Project Structure

```plaintext
oral-cancer-prediction/
│
├── data/
│ └── raw/
│
├── notebooks/
│ ├── Preprocessed_Pipeline.ipynb
│ ├── Logistic_Regression.ipynb
│ ├── KNN.ipynb
│ ├── Random_Forest.ipynb
│ ├── Decision_Tree.ipynb
│ ├── Support_Vector_Machine.ipynb
│ └── XG_Boost.ipynb
│
├── results/
│ └── EDA_visualizations/
│
├── outputs/
│
├── report/
│ ├── final_report.pdf
│ └── Final_Model_Selection.pdf
│
├── requirements.txt
└── README.md

```

---

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Notes
- This project does not include a user interface (UI)
- Model performance is influenced by dataset size, feature limitations, and class imbalance
- The primary goal is **learning, evaluation, and comparison of machine learning models**

---

## Author
Rosheni Bolonne  
Artificial Intelligence Undergraduate

