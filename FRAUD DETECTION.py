#fraud Detection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

train_df = pd.read_csv(r"C:\Users\user\Downloads\archive (3)\fraudTrain.csv")
test_df = pd.read_csv(r"C:\Users\user\Downloads\archive (3)\fraudTest.csv")

print("Train Data Sample:\n", train_df.head())
print("\nClass Distribution:\n", train_df['is_fraud'].value_counts())

sns.countplot(x='is_fraud', data=train_df)
plt.title("Fraud (1) vs Legitimate (0)")
plt.show()

cols_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'dob', 'trans_num', 'zip']
train_df.drop(columns=cols_to_drop, inplace=True)
test_df.drop(columns=cols_to_drop, inplace=True)

categorical_features = ['category', 'merchant', 'gender', 'city', 'state', 'job']
numeric_features = [col for col in train_df.columns if col not in categorical_features + ['is_fraud']]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train = train_df.drop('is_fraud', axis=1)
y_train = train_df['is_fraud']

X_test = test_df.drop('is_fraud', axis=1)
y_test = test_df['is_fraud']

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n=== {model_name} Evaluation ===")
    print("Confusion Matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Legit", "Fraud"])
    disp.plot()
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_processed, y_train)
evaluate_model(log_model, X_test_processed, y_test, "Logistic Regression")

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_processed, y_train)
evaluate_model(tree_model, X_test_processed, y_test, "Decision Tree")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed, y_train)
evaluate_model(rf_model, X_test_processed, y_test, "Random Forest")

