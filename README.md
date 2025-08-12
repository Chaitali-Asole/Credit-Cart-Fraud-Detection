# Credit-Cart-Fraud-Detection
Sure! Here's your *README* in the same structured and styled format as your example, but adapted for your *Credit Card Fraud Detection* project:

---

# 🕵‍♀ Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using historical data. By analyzing transaction patterns, merchant details, user behavior, and more, machine learning models are trained to classify whether a transaction is *fraudulent (1)* or *legitimate (0)*.

---

## 🚀 Features

### 🧹 Data Preprocessing

* Dropped irrelevant columns (e.g., names, IDs, timestamps)
* Encoded categorical variables using *One-Hot Encoding*
* Scaled numerical features with *StandardScaler*

---

### 🌲 Model Training

* Trained three classification models:

  * *Logistic Regression*
  * *Decision Tree*
  * *Random Forest Classifier*

---

### 📊 Model Evaluation

* Evaluated performance using:

  * *Accuracy Score*
  * *Classification Report* (Precision, Recall, F1-Score)
  * *Confusion Matrix Visualization*

---

### 📈 Visualization

* Used *Seaborn* and *Matplotlib* to:

  * Plot class distribution
  * Visualize confusion matrices for all models

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn

---

## 📂 Dataset

The dataset includes credit card transaction data with features like amount, merchant, job, and location.
It is divided into:

* fraudTrain.csv (for training)
* fraudTest.csv (for testing)

📌 *Target Variable:* is_fraud
📌 *Dataset Source:* [Kaggle Fraud Detection Dataset](https://www.kaggle.com/datasets/kartikkannadaguli/credit-card-fraud-detection)

---

## 📊 Model Workflow

1. Load and explore dataset
2. Drop irrelevant or personal columns
3. Encode categorical variables & scale numerical features
4. Train-test split using provided data
5. Train three models:

   * Logistic Regression
   * Decision Tree
   * Random Forest
6. Evaluate and compare model performance
7. Visualize confusion matrices
8. Choose the best-performing model for fraud prediction
## 📌 Output Example

```yaml
Model: Random Forest Classifier

Accuracy: 0.97

Classification Report:
              precision    recall  f1-score   support

       Legit       0.99      0.98      0.98    555134
       Fraud       0.84      0.89      0.87     9441

    accuracy                           0.97    564575
   macro avg       0.91      0.94      0.93    564575
weighted avg       0.97      0.97      0.97    564575

 📈 Results

* Achieved *\~97% accuracy* with *Random Forest Classifier*
* Outperformed Logistic Regression and Decision Tree models
* Successfully detected fraudulent patterns using historical transaction data

---

Let me know if you want to include a requirements.txt or a badge for license, Python version, etc.
