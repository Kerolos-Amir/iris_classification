# Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pandas as pd

# =========================
# 1) Load Dataset
# =========================

iris = load_iris()

# Features
X = iris.data

# Labels (Target)
y = iris.target

# عرض أول 5 صفوف
df = pd.DataFrame(X, columns=iris.feature_names)
print("First 5 Rows:")
print(df.head())

# =========================
# 2) Feature Scaling
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# 3) Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# =========================
# 4) Apply KNN Algorithm
# =========================

knn = KNeighborsClassifier(n_neighbors=3)

# Training
knn.fit(X_train, y_train)

# Prediction
y_pred = knn.predict(X_test)

# =========================
# 5) Evaluation
# =========================

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nF1 Score:")
print(f1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# 6) Test New Flower
# =========================

# Example:
# Sepal Length = 5.1
# Sepal Width  = 3.5
# Petal Length = 1.4
# Petal Width  = 0.2

new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Scaling
new_flower_scaled = scaler.transform(new_flower)

# Prediction
prediction = knn.predict(new_flower_scaled)

print("\nPredicted Class:")
print(iris.target_names[prediction][0])