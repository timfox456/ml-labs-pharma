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
# ---

# # Lab: Embed Lab Reports for Semantic Search and Anomaly Detection
#
# **Goal:** In this lab, you will learn how to convert entire documents into numerical representations called "embeddings." You will then use these embeddings to perform two powerful tasks: semantic search and anomaly detection. 
#
# **Key Concepts:**
# - **Embeddings:** Understanding that AI models can represent the "meaning" of text as a vector of numbers.
# - **Semantic Search:** Finding documents based on conceptual similarity, not just keyword matching.
# - **Anomaly Detection:** Identifying unusual or outlier data points in a dataset, which can be a sign of quality issues.
#
# ---
#
# ## 1. Setup
#
# We will need `openai` for creating embeddings, `numpy` for numerical operations, `scikit-learn` for PCA and distance calculations, and `matplotlib` for plotting.

# + 
# %pip install openai numpy scikit-learn matplotlib python-dotenv

import os
import openai
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# - 

# ## 2. Loading and Preparing the Data
#
# We have four lab reports. Three are "normal," and one (`lab_report_04.txt`) is an "anomaly." While it technically passes all specifications, its results are consistently at the edge of the acceptable limits. Our goal is to detect this subtle deviation using AI.
#
# First, let's load all four reports into memory.

# + 
report_files = [
    "data/lab_report_01.txt",
    "data/lab_report_02.txt",
    "data/lab_report_03.txt",
    "data/lab_report_04.txt"  # Our anomaly
]

reports = []
for file_path in report_files:
    try:
        with open(file_path, "r") as f:
            reports.append({"filename": os.path.basename(file_path), "text": f.read()})
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")

print(f"Loaded {len(reports)} reports.")
# - 

# ## 3. Generating Embeddings
#
# An **embedding** is a vector (a list of numbers) that represents the semantic meaning of a piece of text. We will use OpenAI's `text-embedding-3-small` model to generate an embedding for each lab report. Documents with similar meanings will have vectors that are "closer" to each other in multi-dimensional space.

# + 
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return openai.embeddings.create(input = [text], model=model).data[0].embedding

# Generate embeddings for each report
print("Generating embeddings for each lab report...")
for report in reports:
    report['embedding'] = get_embedding(report['text'])
    print(f"  - Generated embedding for {report['filename']}")

print("Embeddings generated successfully.")
# - 

# ## 4. Use Case 1: Semantic Search
#
# Now that we have embeddings, we can perform semantic search. Instead of matching keywords, we will search by comparing the *meaning*.
#
# Let's ask a question, embed it, and then find the lab report whose embedding is most similar to our question's embedding.
#
# **Our Question:** "Which batch had the highest dissolution rate?"

# + 
# 1. Embed the search query
search_query = "Which batch had the highest dissolution rate?"
query_embedding = get_embedding(search_query)

print(f"Search Query: '{search_query}'")

# 2. Calculate the similarity between the query and each report
# We use cosine similarity, which measures the cosine of the angle between two vectors.
# A value of 1 means they are identical, 0 means they are unrelated.
similarities = []
for report in reports:
    similarity = cosine_similarity([query_embedding], [report['embedding']])[0][0]
    similarities.append(similarity)
    print(f"  - Similarity with {report['filename']}: {similarity:.4f}")

# 3. Find the most similar report
most_similar_index = np.argmax(similarities)
most_similar_report = reports[most_similar_index]

print("\n--- Search Result ---")
print(f"The most relevant report is: {most_similar_report['filename']}")
print("\nReport Content:")
print(most_similar_report['text'])
# - 

# ### Exercise 1: Your Own Search
#
# **Your Task:** Try a different search query. For example:
# - "Find the report from analyst Ben Carter."
# - "Which batch had the most salicylic acid impurity?"
#
# Run the code below with your new query and see if it finds the correct report.

# + 
# YOUR CODE HERE
my_search_query = "Find the report from analyst Ben Carter."
my_query_embedding = get_embedding(my_search_query)

my_similarities = [cosine_similarity([my_query_embedding], [r['embedding']])[0][0] for r in reports]
best_match_index = np.argmax(my_similarities)

print(f"My Search Query: '{my_search_query}'")
print(f"Best match: {reports[best_match_index]['filename']}")
# - 

# ---
#
# ## 5. Use Case 2: Anomaly Detection
#
# Semantic search is powerful, but the real goal is to proactively identify potential quality issues. We can do this by finding reports that are "semantically different" from the norm.
#
# Our process will be:
# 1.  Define what "normal" looks like by averaging the embeddings of our known-good reports.
# 2.  Calculate the distance of each report from this "normal" center.
# 3.  The report with the greatest distance is our anomaly.
#
# We will use the first three reports as our "golden batches" to establish the norm.

# + 
# 1. Create a "mean embedding" from our normal batches
normal_embeddings = [reports[i]['embedding'] for i in range(3)] # Reports 0, 1, 2
mean_embedding = np.mean(normal_embeddings, axis=0)

print("Calculated the mean embedding for 'normal' batches.")

# 2. Calculate the cosine distance of each report from the mean
# Cosine Distance = 1 - Cosine Similarity
distances = []
for report in reports:
    similarity = cosine_similarity([report['embedding']], [mean_embedding])[0][0]
    distance = 1 - similarity
    report['distance_from_norm'] = distance
    distances.append(distance)
    print(f"  - Distance for {report['filename']}: {distance:.4f}")

# 3. Find the report with the highest distance
most_anomalous_index = np.argmax(distances)
most_anomalous_report = reports[most_anomalous_index]

print("\n--- Anomaly Detection Result ---")
print(f"The most anomalous report is: {most_anomalous_report['filename']}")
print(f"This report is the most semantically different from the 'golden batch' profile.")
# - 

# ## 6. Visualizing the Anomaly
#
# The distances clearly show that `lab_report_04.txt` is an outlier. But to make this even clearer for human analysts, we can visualize the embeddings.
#
# The embeddings have over 1500 dimensions, which we can't plot. We'll use a technique called **Principal Component Analysis (PCA)** to reduce the dimensionality down to 2D so we can create a scatter plot.

# + 
# Prepare the data for PCA
all_embeddings = [r['embedding'] for r in reports]
filenames = [r['filename'] for r in reports]

# Fit PCA and transform the embeddings to 2D
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(all_embeddings)

# Plot the results
plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', s=100)

# Add labels
for i, filename in enumerate(filenames):
    # Highlight the anomaly
    color = 'red' if "04" in filename else 'black'
    plt.annotate(filename, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                 textcoords="offset points", xytext=(0,10), ha='center', color=color)

plt.title("Visualizing Lab Report Embeddings with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()
# - 

# ### Interpreting the Chart
#
# As you can see, the three "normal" reports cluster closely together. The anomalous report, `lab_report_04.txt`, is clearly separated from the main cluster. This visual evidence provides a powerful and intuitive way for a QC analyst to spot potential issues, even when all tests are technically "passing."
#
# ---
#
# ## Conclusion
#
# In this lab, you have moved beyond simple text extraction to understanding the *meaning* behind documents. You have learned how to:
# 1.  Generate embeddings to represent documents as numerical vectors.
# 2.  Use these embeddings to find documents related to a natural language query (Semantic Search).
# 3.  Calculate a "normal" profile for a set of documents and find outliers (Anomaly Detection).
# 4.  Visualize high-dimensional data to make anomalies easy to spot.
#
# These techniques are fundamental for building proactive, intelligent quality control systems in the pharmaceutical industry.
