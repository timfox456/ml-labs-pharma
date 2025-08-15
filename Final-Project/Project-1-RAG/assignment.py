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

# # Final Project 1: Clinical Paper Q&A with RAG
# 
# **Objective:** Your goal is to build a question-answering system using the Retrieval-Augmented Generation (RAG) pattern. This system will help a researcher quickly find and synthesize information from a collection of clinical paper abstracts on diabetes. 
# 
# **Estimated Time:** 1 hour
# 
# **Core Task:** You will load several document abstracts, embed them into a vector store, and build a LangChain retrieval chain to answer complex questions. 
# 
# ---
# 
# ## 1. Setup
# 
# First, ensure you have a `.env` file in the project root with your `OPENAI_API_KEY`. Then, install the required packages from `requirements.txt`.
# 
# ```bash
# # In your terminal
# pip install -r requirements.txt
# ```
# 
# Now, let's import the necessary libraries.

# + import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader

load_dotenv()
# - 

# ## 2. Load and Prepare the Data
# 
# The dataset for this project is `data/diabetes_clinical_papers.csv`. It contains the title and abstract for several clinical papers.
# 
# **Your Task (Starter Code):**
# - Use the `CSVLoader` from LangChain to load the documents.
# - You need to specify which column contains the text you want to embed. For our purposes, the `abstract` is the most important source of information.

# + # Starter Code
loader = CSVLoader(
    file_path="./data/diabetes_clinical_papers.csv",
    source_column="title", # Use the paper title as the source metadata
    csv_args={"delimiter": ","}
)

# The 'page_content' of each document will be the text from the 'abstract' column.
documents = loader.load()


print(f"Loaded {len(documents)} documents.")
print("\n--- Sample Document ---")
print(documents[0].page_content)
print(f"\nSource: {documents[0].metadata['source']}")
# - 

# ## 3. Build the RAG Pipeline
# 
# Now it's time to build the core of our system. This involves two main steps:
# 1.  **Embedding and Storage:** Convert the documents into numerical vectors (embeddings) and store them in a searchable vector database (we'll use FAISS for its simplicity).
# 2.  **Retrieval and Generation:** Create a chain that can take a user's question, retrieve the most relevant documents from the vector store, and then use an LLM to generate a final answer based on that retrieved context.
# 
# **Your Task:**
# - Create an instance of `OpenAIEmbeddings`.
# - Use `FAISS.from_documents` to create a vector store from your loaded documents and the embeddings model.
# - Create a `RetrievalQA` chain using the `from_chain_type` method.
#   - Use an `OpenAI` LLM.
#   - The chain type should be `"stuff"`.
#   - The retriever will be created from your FAISS vector store (`vectorstore.as_retriever()`).

# + # --- YOUR CODE HERE ---

# 1. Create the embeddings model
embeddings = None # Replace None with the correct code

# 2. Create the FAISS vector store
vectorstore = None # Replace None with the correct code

# 3. Create the RetrievalQA chain
rag_chain = None # Replace None with the correct code

# --- END YOUR CODE ---
# - 

# ## 4. Test Your Q&A System
# 
# If your RAG pipeline is built correctly, you should be able to ask it complex questions that require synthesizing information from one or more abstracts.
# 
# **Your Task:**
# - Ask at least two questions to your `rag_chain`.
# - **Question 1 (Simple):** Ask a question that can be answered from a single abstract (e.g., "What is the main finding of the EMPA-REG OUTCOME trial?").
# - **Question 2 (Complex):** Ask a question that requires information from multiple abstracts (e.g., "Which drugs have shown benefits for both glycemic control and weight loss?").
# - Print the results.

# + # --- YOUR CODE HERE ---

# Question 1
simple_question = ""
# Add your simple question here
simple_answer = None # Use your rag_chain to get the answer

print(f"Q: {simple_question}")
print(f"A: {simple_answer}\n")


# Question 2
complex_question = ""
# Add your complex question here
complex_answer = None # Use your rag_chain to get the answer

print(f"Q: {complex_question}")
print(f"A: {complex_answer}\n")

# --- END YOUR CODE ---
# - 

# ## 5. (Optional) Bonus Challenge
# 
# The default `RetrievalQA` chain is great, but it doesn't cite its sources. Can you modify the chain to return the source documents it used to generate the answer?
# 
# **Hint:**
# - Look at the documentation for `RetrievalQA`. Is there a parameter you can set to make it return the source documents?
# - You might need to change how you call the chain and how you process the output to see the sources.
