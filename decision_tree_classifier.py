
# decision_tree_classifier.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris.feature_names, iris.target_names


def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_decision_tree(X_train, y_train, criterion='entropy', max_depth=3):
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def visualize_tree(clf, feature_names, class_names):
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
    plt.title("Decision Tree Visualization")
    plt.show()


def main():
    X, y, feature_names, class_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_decision_tree(X_train, y_train)
    evaluate_model(clf, X_test, y_test)
    visualize_tree(clf, feature_names, class_names)


if __name__ == "__main__":
    main()
