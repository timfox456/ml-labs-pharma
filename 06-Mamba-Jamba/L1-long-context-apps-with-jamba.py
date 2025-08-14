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

# # Lab: Build Long-Context AI Apps with Jamba
#
# **Goal:** In this lab, you will experience the power of a long-context AI model. You will use a model like Jamba to process a lengthy, domain-specific document—something that would be difficult for a standard model with a small context window.
#
# **Key Concepts:**
# - **Long-Context Processing:** Understanding the challenges and benefits of working with large amounts of text at once.
# - **Hybrid Architecture (Jamba):** Appreciating how combining Transformer and Mamba blocks enables efficient long-context processing.
# - **Summarization and Q&A over entire documents.**
#
# **Note:** We will be using the OpenAI API for this lab, which may not have Jamba specifically, but we can simulate the experience by using their best long-context model (e.g., `gpt-4-turbo`) which can handle large context windows. The principles remain the same.
#
# --- 
#
# ## 1. Setup
#
# First, let's install the necessary libraries.

# + # %pip install openai python-dotenv

import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# - 

# ## 2. Loading the Document
#
# The core of this lab is our source document on "Quality by Design (QbD)". It's a comprehensive overview that is too long for many standard models to handle in a single pass. Let's load it into memory.

# + 
try:
    with open("data/quality_by_design_overview.txt", "r") as f:
        qbd_document_text = f.read()
    print("Successfully loaded the 'Quality by Design' document.")
    print(f"Document length: {len(qbd_document_text.split())} words")
except FileNotFoundError:
    print("Error: Make sure 'data/quality_by_design_overview.txt' exists.")
    qbd_document_text = ""
# - 

# ## 3. The Power of Long Context: Full Document Summarization
#
# With a traditional model (e.g., 4k context window), summarizing this document would require complex techniques like "Map-Reduce," where you summarize chunks and then summarize the summaries. This often leads to a loss of context and nuance.
#
# With a long-context model, we can feed the entire document in a single prompt.
#
# ### Helper Function
# Let's define the helper function we'll use to interact with the OpenAI API.

# +
def get_completion(prompt, system_prompt="You are a helpful assistant."):
    """
    A helper function to get a completion from the OpenAI API.
    """
    if not qbd_document_text:
        return "Document not loaded. Please check the file path."
        
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo", # This model supports a large context window
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"
# -

### Exercise 1: Generate a High-Level Summary
#
# **Your Task:** Write a prompt that asks the AI to provide a high-level, one-paragraph summary of the entire document.

# +
# YOUR PROMPT HERE
summary_prompt = f"""
Please provide a concise, one-paragraph executive summary of the following document on Quality by Design (QbD).

--- DOCUMENT ---
{qbd_document_text}
--- END DOCUMENT ---
"""

if qbd_document_text:
    print("Generating summary...")
    summary = get_completion(summary_prompt)
    print("\n--- Executive Summary ---")
    print(summary)
# - 

# ### Exercise 2: Structured Extraction from the Full Document
#
# A simple summary is useful, but the real power comes from extracting specific, structured information from the entire text at once.
#
# **Your Task:** Write a prompt that asks the AI to identify and list all the **Core Elements of Quality by Design** mentioned in the text. The output should be a numbered list.

# + 
# YOUR PROMPT HERE
structured_extraction_prompt = f"""
# Based on the document provided below, please identify and list the core elements of Quality by Design (QbD).
# The output should be a numbered list, with a brief one-sentence description for each element based on the text.
#
# --- DOCUMENT ---
# {qbd_document_text}
# --- END DOCUMENT ---
# """
#
# if qbd_document_text:
#     print("Extracting core elements...")
#     core_elements = get_completion(structured_extraction_prompt)
#     print("\n--- Core Elements of QbD ---")
#     print(core_elements)
# - 

# ## 4. Full-Context Question Answering
#
# Long context allows for sophisticated question-answering where the model can synthesize information from different parts of the document to form a coherent answer.
#
# ### Exercise 3: Answering a Complex Question
#
# **Your Task:** Ask a question that requires the model to connect at least two different concepts from the text. For example, ask how Risk Assessment relates to the Control Strategy.

# + 
# YOUR QUESTION HERE
complex_question = f"""
# According to the provided document, how does the 'Risk Assessment' phase influence the 'Control Strategy' in a Quality by Design framework?
#
# --- DOCUMENT ---
# {qbd_document_text}
# --- END DOCUMENT ---
# """
#
# if qbd_document_text:
#     print(f"Asking complex question: {complex_question.splitlines()[1]}")
#     answer = get_completion(complex_question)
#     print("\n--- AI's Answer ---")
#     print(answer)
# - 

# ### Why this is better than RAG (Retrieval-Augmented Generation) in some cases:
#
# For RAG, you would embed chunks of the document, find the most relevant chunks, and then feed only those to the model. This is great for very large document sets.
#
# However, if a question requires synthesizing information from the *introduction* and the *conclusion* of a document, RAG might fail to retrieve both chunks. A long-context model sees the entire document and can make these connections seamlessly.
#
# --- 
#
# ## Conclusion
#
# In this lab, you have seen the practical benefits of modern, long-context AI architectures like Jamba. By being able to process entire documents in a single pass, you can perform tasks that were previously much more complex or impossible:
#
# 1.  **Holistic Summarization:** Creating summaries that capture the full context of the document.
# 2.  **Complex Q&A:** Answering questions that require synthesizing information from multiple sections.
# 3.  **Simplified Workflows:** Avoiding the need for complex chunking and map-reduce logic.
#
# As these models become more widespread, the ability to build applications that leverage long-context understanding will be a critical skill.