#?----------------------------------
#? --Author : Casey Hansen
#? Project: Iris
#?----------------------------------

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd

iris = load_iris()

#? Features (input data)
X = iris.data

#? Target (output data)
y = iris.target

#? Feature names
feature_names = iris.feature_names

#? Target names
target_names = iris.target_names

print("Features: ", feature_names)
print("Target Names: ", target_names)
print("First 5 Samples: ", X[:5])

#! Data Preprocessing Step 1. Check for missing, incomplete, or duplicate data

data = pd.DataFrame(iris.data, columns=feature_names)

#? Check for missing values
print("\nMissing values per feature:", data.isnull().sum())

#? Check for duplicates
print("Number of duplicate rows:", data.duplicated().sum())

#? Drop the duplicates
data.drop_duplicates(inplace=True)

#? Check for missing values
print("\nMissing values per feature:", data.isnull().sum())

#? Check for duplicates
print("Number of duplicate rows:", data.duplicated().sum())

#! Data Preprocessing Step 2. Standardize Features Values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

#? Fit the scaler on the training data and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("First 5 rows of scaled training data\n", X_train_scaled[:5])

#! Step 3. Train and Evaluate the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

#? Save the trained model and scaler to a ile
with open("model_and_scaler.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

print("\n Model and Scaler have been saved to 'model_and_scaler.pkl'")

#? Make predictions on the test data
y_pred = model.predict(X_test_scaled)

#? Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

#? Generate a classification report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

#? Conf Matrix are used to evaluate performance of classification models.
#? They breakdown predictions compare to actual outcomes.
cm = confusion_matrix(y_test, y_pred)

#? Visualize the C-matrix in a heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Iris Project")
plt.show()