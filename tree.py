#!.venv/bin/python3

# decision_tree_iris.py
# A simple, end-to-end Decision Tree example on the Iris dataset.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# 1) Load data
iris = load_iris()
X = iris.data                          # shape: (150, 4)
y = iris.target                        # shape: (150,)
feature_names = iris.feature_names     # list of 4 feature names
class_names = iris.target_names        # ['setosa', 'versicolor', 'virginica']

# 2) Train/test split (stratify to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) Define and train a Decision Tree
# - max_depth keeps the tree small (more interpretable; helps avoid overfitting)
# - criterion can be 'gini' (default) or 'entropy'
clf = DecisionTreeClassifier(
    criterion="gini", # values 'gini' or 'entropy'
    max_depth=3,
    random_state=42
)
clf.fit(X_train, y_train)

# 4) Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 5) Confusion matrix (visual)
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=class_names, cmap="Blues", values_format="d"
)
plt.title("Decision Tree – Confusion Matrix (Iris)")
plt.tight_layout()
plt.show()

# 6) Feature importances
importances = clf.feature_importances_
print("Feature Importances:")
for name, imp in sorted(zip(feature_names, importances), key=lambda t: -t[1]):
    print(f"  {name:20s} {imp:.3f}")

# 7) Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree (max_depth=3) – Iris")
plt.tight_layout()
plt.show()
