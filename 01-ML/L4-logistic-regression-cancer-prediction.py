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

# # Lab: Logistic Regression for Binary Classification
#
# **Goal:** In this lab, you will build a logistic regression model to predict whether a tumor is malignant or benign based on medical measurements.
#
# **Key Concepts:**
# - **Binary Classification:** Predicting one of two possible outcomes.
# - **Logistic Regression:** A fundamental algorithm for classification tasks.
# - **Model Evaluation:** Using accuracy and a confusion matrix to assess performance.
#
# ---
#
# ## 1. Setup
#
# We'll use `scikit-learn` for the model and dataset, `pandas` for data handling, and `seaborn` for plotting.

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_breast_cancer
# - 

# ## 2. Load and Explore the Data
#
# We will use the Breast Cancer Wisconsin dataset, which is conveniently included in scikit-learn.

# + 
# Load the dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

# Display the first few rows of the features
print("---", "Features (X) ---")
print(X.head())

# Display the target variable info
print("\n--- Target (y) ---")
print(f"Target names: {cancer.target_names}")
print(f"Target distribution:\n{y.value_counts()}")
# - 

# ## 3. Prepare the Data
#
# The data is already clean. Our only step is to split it into training and testing sets. This ensures we can evaluate our model on data it has never seen before.

# + 
# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
# - 

# ## 4. Build and Train the Logistic Regression Model
#
# We will now create an instance of the `LogisticRegression` model and fit it to our training data.

# + 
# Initialize the model
# We increase max_iter to ensure the model converges.
model = LogisticRegression(max_iter=10000)

# Train the model
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.")
# - 

# ## 5. Evaluate the Model
#
# Now that the model is trained, let's see how well it performs on the unseen test data.
#
# ### Make Predictions

# + 
# Make predictions on the test set
y_pred = model.predict(X_test)
# - 

# ### Check Accuracy
# Accuracy is the simplest metric: the proportion of correct predictions.

# + 
# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
# - 

# ### Confusion Matrix
# A confusion matrix gives us a more detailed look at the model's performance, showing us what it got right and what it got wrong.
#
# - **True Positives (TP):** Correctly predicted positive (e.g., correctly identified as benign).
# - **True Negatives (TN):** Correctly predicted negative (e.g., correctly identified as malignant).
# - **False Positives (FP):** Incorrectly predicted positive (e.g., malignant tumor classified as benign - **Type I Error**).
# - **False Negatives (FN):** Incorrectly predicted negative (e.g., benign tumor classified as malignant - **Type II Error**).

# + 
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
# - 

# **Analysis:** The confusion matrix shows us exactly how many malignant and benign tumors were classified correctly and incorrectly. In a medical context, we are often most concerned with **False Positives** (classifying a malignant tumor as benign), as this is the most dangerous type of error.
#
# ---
#
# ## Conclusion
#
# In this lab, you have successfully built and evaluated a logistic regression model for a real-world classification task. You learned how to:
# 1.  Load a standard dataset from scikit-learn.
# 2.  Split data into training and testing sets.
# 3.  Train a logistic regression classifier.
# 4.  Evaluate the model's performance using both accuracy and a confusion matrix.
