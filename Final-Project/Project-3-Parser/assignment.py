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

# # Final Project 3: Clinical Trial Report Parser
#
# **Objective:** Your goal is to build a robust system that can extract structured information from unstructured text using an LLM with **function calling**. This is a highly valuable skill for turning messy, real-world documents into clean, database-ready data.
#
# **Estimated Time:** 1 hour
#
# **Core Task:** You will define a Pydantic schema for the data you want to extract from clinical trial press releases. Then, you will use LangChain and OpenAI's function calling ability to parse the text and populate your schema.
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

# +=
import json
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain

load_dotenv()
# -=

# ## 2. The Challenge: Unstructured Press Releases
#
# Our data consists of two `.txt` files containing fictional press releases about clinical trials. This text is a mix of formal announcements, scientific details, and forward-looking statements. Manually extracting key information is slow and error-prone. Our job is to automate it.
#
# ### Load the Data (Starter Code)

# +=
# Starter Code
try:
    with open("./data/trial_press_release_1.txt", "r") as f:
        press_release_1 = f.read()
    with open("./data/trial_press_release_2.txt", "r") as f:
        press_release_2 = f.read()
    print("--- Press Release 1 ---")
    print(press_release_1)
    print("\n--- Press Release 2 ---")
    print(press_release_2)
except FileNotFoundError:
    print("Error: Data files not found.")
    press_release_1, press_release_2 = "", ""
# -=

# ## 3. Define Your Data Schema
#
# Before we can extract anything, we must define the *structure* of the data we want. Pydantic is the perfect tool for this, as it allows us to define a data model with types and descriptions.
#
# **Your Task:**
# - Complete the Pydantic `ClinicalTrialResult` model below.
# - Add fields for `trial_phase`, `patient_count`, and `primary_endpoint`.
# - Use the `Field` function to provide a clear description for each new field. This description is crucial, as it tells the LLM what kind of information to look for.
# - For the `outcome` field, use the `Literal` type to constrain the LLM's output to one of three specific strings: `"Successful"`, `"Failed"`, or `"Ongoing"`.

# +=
# --- YOUR CODE HERE ---

class ClinicalTrialResult(BaseModel):
    """A structured representation of a clinical trial result."""
    drug_name: str = Field(..., description="The name of the drug being tested.")
    trial_phase: str = Field(..., description="The phase of the clinical trial (e.g., 'Phase 3', 'Phase 2b').")
    # Add patient_count field here
    # Add primary_endpoint field here
    outcome: Literal["Successful", "Failed", "Ongoing"] = Field(..., description="The final outcome of the trial.")

# --- END YOUR CODE ---
# -=

# ## 4. Build the Extraction Chain
#
# We will use a specialized LangChain function, `create_structured_output_chain`, which is designed for exactly this task. It takes our Pydantic model and an LLM and creates a chain that will reliably output a JSON object matching our schema.
#
# **Your Task:**
# - Create an instance of the `ChatOpenAI` model. Use a powerful model like `"gpt-4-turbo"` for best results.
# - Define a `ChatPromptTemplate`. It should instruct the AI to act as an expert data extractor and tell it to extract information from the user's input.
# - Use `create_structured_output_chain` to build the final chain.

# +=
# --- YOUR CODE HERE ---

# 1. Initialize the LLM
llm = None # Replace None

# 2. Create the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting key information from medical and scientific documents."),
    ("human", "Please extract the clinical trial data from the following text: {text}")
])

# 3. Create the structured output chain
# Pass your Pydantic model to the 'output_schema' argument
extraction_chain = None # Replace None

# --- END YOUR CODE ---
# -=

# ## 5. Run and Test Your Parser
#
# Now it's time to use your chain to parse the two press releases.
#
# **Your Task:**
# - Invoke the `extraction_chain` for both `press_release_1` and `press_release_2`.
# - The input to the chain should be a dictionary with the key `"text"`.
# - The output will be a dictionary. The structured data is in the `"function"` key.
# - Print the extracted Pydantic objects.

# +=
# --- YOUR CODE HERE ---

if 'extraction_chain' in locals() and extraction_chain:
    # Parse the first press release
    result_1 = None # Invoke the chain
    extracted_data_1 = None # Get the data from the 'function' key

    print("--- Extracted Data 1 ---")
    # print(extracted_data_1)

    # Parse the second press release
    result_2 = None # Invoke the chain
    extracted_data_2 = None # Get the data from the 'function' key

    print("\n--- Extracted Data 2 ---")
    # print(extracted_data_2)
else:
    print("Extraction chain not created. Please complete the previous step.")

# --- END YOUR CODE ---
# -=

# **Expected Output:**
# You should see two clean `ClinicalTrialResult` objects, with all the fields correctly populated from the unstructured text. This demonstrates how you can turn messy documents into clean, structured data that could be saved to a database or used in an analysis pipeline.
#
# ---
#
# ## Conclusion
#
# In this project, you have built a powerful and modern data extraction pipeline. You learned how to:
# 1.  Define a formal data schema using Pydantic to specify your desired output.
# 2.  Use LangChain's `create_structured_output_chain` to leverage LLM function calling.
# 3.  Reliably parse unstructured text into validated, structured objects.
#
# This technique is one of the most practical and high-impact applications of LLMs in a business or research context.
