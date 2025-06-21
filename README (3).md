
# 🌳 Decision Tree Classifier using Scikit-Learn

This project demonstrates how to build, visualize, and evaluate a **Decision Tree Classifier** using the **Iris dataset** in Python with Scikit-learn.

---

## 📌 Features

- Modular Python functions for loading data, training, evaluation, and visualization.
- Visualization of the decision tree structure.
- Performance evaluation using accuracy, classification report, and confusion matrix.
- Optional hyperparameter tuning (max_depth).
- Easy to extend to any CSV dataset.

---

## 📁 Project Structure

```
decision-tree-classifier/
│
├── decision_tree_classifier.ipynb   # Main notebook with full implementation
├── README.md                        # Project description and instructions
└── requirements.txt                 # (Optional) List of dependencies
```

---

## 🔧 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/rajjadhav7348/decision-tree-classifier.git
   cd decision-tree-classifier
   ```

2. Open the notebook:
   - Run locally using Jupyter Notebook, or  
   - Open in [Google Colab](https://colab.research.google.com/) and upload the notebook.

---

## 📦 Dependencies

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 🧠 Dataset

Using the built-in **Iris dataset** from `sklearn.datasets`.

You can replace it with your own CSV dataset:
```python
data = pd.read_csv('your_dataset.csv')
X = data.drop('target_column', axis=1)
y = data['target_column']
```

---

## 📊 Visualization Example

The tree structure is plotted using `plot_tree()` from Scikit-learn:

![Decision Tree Example](https://scikit-learn.org/stable/_images/sphx_glr_plot_tree_classification_001.png)

---

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

## 🧪 Optional: Hyperparameter Tuning

The notebook includes a loop to test different values of `max_depth` and measure accuracy.

---

## 📬 Author

- Name: Rajnandan Jadhav
- GitHub: [github.com/rajjadhav7348](https://github.com/rajjadhav7348)

---

## 📄 License

This project is licensed under the MIT License.
