# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# # ---

# # Lab: Decision Trees for Iris Classification
#
# **Goal:** In this lab, you will build a Decision Tree classifier and visualize it to understand how it makes decisions. Decision Trees are powerful because they are highly interpretable ("white box" models).
#
# **Key Concepts:**
# - **Multi-class Classification:** Predicting one of more than two possible outcomes.
# - **Decision Tree:** A tree-like model that makes decisions based on feature values.
# - **Model Interpretability:** The ability to understand and explain how a model works.
#
# ---
#
# ## 1. Setup
#
# We need `scikit-learn` for the model and dataset, and `matplotlib` for plotting the tree.

# + tags=['imports']
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# - 

# ## 2. Load and Explore the Iris Data
#
# The Iris dataset is a classic in machine learning. The goal is to classify a flower into one of three species (Setosa, Versicolour, Virginica) based on the length and width of its sepals and petals.

# + tags=['data_loading', 'data_exploration']
# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

print("---" + " Features (X) ---")
print(f"Feature names: {iris.feature_names}")
print(f"Sample data:\n{X[:5]}")

print("\n" + "---" + " Target (y) ---")
print(f"Target names: {iris.target_names}")
print(f"Sample labels: {y[:5]}")
# - 

# ## 3. Prepare the Data
#
# We'll split the data into a training set and a testing set.

# + tags=['data_preparation']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
# - 

# ## 4. Build and Train the Decision Tree
#
# We will create an instance of the `DecisionTreeClassifier` and fit it to our training data.

# + tags=['model_training']
# Initialize the model
# `max_depth` is a hyperparameter to prevent the tree from growing too complex and overfitting.
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the model
print("Training the Decision Tree...")
clf.fit(X_train, y_train)
print("Training complete.")
# - 

# ## 5. Visualize the Decision Tree
#
# This is the most powerful feature of this model. We can directly see the rules it learned from the data.

# + tags=['visualization']
plt.figure(figsize=(20,10))
plot_tree(clf,
          filled=True,
          rounded=True,
          class_names=iris.target_names,
          feature_names=iris.feature_names)
plt.title("Decision Tree for Iris Classification")
plt.show()
# - 

# **How to Read the Tree:**
# - Each node represents a decision based on a feature (e.g., "petal width (cm) <= 0.8").
# - If the condition is true, you follow the left branch; if false, you follow the right branch.
# - **Gini:** The Gini impurity is a measure of how "mixed" the classes are in a node. A Gini of 0.0 means the node is perfectly pure (all samples belong to one class).
# - **Samples:** The number of training samples in that node.
# - **Value:** The distribution of samples across the classes.
# - **Class:** The majority class in that node.
#
# ---
#
# ## 6. Evaluate the Model
#
# Now let's see how well our interpretable model performs on the test data.

# + tags=['model_evaluation']
# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")
# - 

# **Analysis:** The Decision Tree is not only highly accurate on this dataset, but we can also explain to a non-expert *exactly* why it made a particular prediction by tracing the path down the tree. This is a major advantage in regulated industries or when explaining model decisions is important.
#
# ---
#
# ## Conclusion
#
# In this lab, you have built and visualized a Decision Tree classifier. You learned:
# 1.  How to train a Decision Tree for a multi-class problem.
# 2.  The key advantage of Decision Trees: their interpretability.
# 3.  How to visualize the learned rules to understand the model's logic.
