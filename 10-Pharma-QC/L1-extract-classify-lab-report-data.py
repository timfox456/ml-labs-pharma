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

# # Lab: Extract and Classify Lab Report Data with AI
#
# **Goal:** In this lab, you will build an AI-powered pipeline to parse unstructured text from a pharmaceutical Quality Control (QC) lab report and convert it into structured, usable data. 
#
# **Key Concepts:**
# - **Unstructured vs. Structured Data:** Recognizing the challenge of working with free-form text.
# - **Function Calling / Tool Use:** Using modern LLM capabilities to reliably extract information.
# - **Data Schemas (Pydantic):** Defining a clear, validated structure for our desired output.
#
# ---
#
# ## 1. Setup
# 
# First, let's install the necessary libraries. We'll need `openai` to interact with the LLM and `pydantic` to define our data schemas.

# + 
# %pip install openai pydantic python-dotenv

import os
import openai
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# - 

# ## 2. The Challenge: Unstructured Lab Reports
# 
# In pharmaceutical QC, data often comes in formats like PDFs or plain text files. This is "unstructured" data. It's easy for a human to read but difficult for a computer to process.
# 
# Let's start by loading and examining our sample lab report.

# + 
# Load the sample lab report
try:
    with open("data/lab_report_01.txt", "r") as f:
        lab_report_text = f.read()
    print(lab_report_text)
except FileNotFoundError:
    print("Error: Make sure 'data/lab_report_01.txt' exists.")
    lab_report_text = ""
# - 

# ### The Problem with Traditional Methods
# 
# Traditionally, you might use Regular Expressions (Regex) to parse this text. However, regex is:
# - **Brittle:** A small change in the report format can break the entire script.
# - **Complex:** Writing and debugging regex for complex documents is very difficult.
# - **Not Scalable:** You need to write new patterns for every new report format.
# 
# AI offers a more flexible and robust solution.
#
# ---
#
# ## 3. Defining Our Target Data Structure with Pydantic
#
# Before we can extract data, we need to define what we want to extract. A data schema is a formal blueprint of our desired output. We will use Pydantic, a library for data validation and settings management using Python type annotations.
#
# We want to extract key information like the `batch_id`, `conclusion`, and a list of all the `tests` performed.

# + 
class TestResult(BaseModel):
    """A model to hold the results of a single QC test."""
    test_name: str = Field(..., description="The official name of the test, e.g., 'Assay (Purity)'.")
    method: Optional[str] = Field(None, description="The method or standard used for the test, e.g., 'HPLC-UV/VIS-001'.")
    specification: str = Field(..., description="The acceptance criteria for the test, e.g., '99.5% - 100.5%'.")
    result: str = Field(..., description="The actual measured result of the test, e.g., '99.87%' or 'Conforms'.")

class LabReport(BaseModel):
    """The top-level model for the entire Certificate of Analysis."""
    batch_id: str = Field(..., description="The unique identifier for the batch.")
    product_name: str = Field(..., description="The name of the product being tested.")
    conclusion: str = Field(..., description="The final conclusion of the report, e.g., 'meets all specifications'.")
    tests: List[TestResult] = Field(..., description="A list of all tests performed.")
# - 

# **Why Pydantic is great:**
# - **Clear & Readable:** The schema is self-documenting.
# - **Validation:** It automatically validates the data extracted by the LLM. If the LLM returns a `result` that isn't a string, Pydantic will raise an error.
# - **IDE Support:** Autocompletion and type-checking make your code more robust.
#
# ---
#
# ## 4. Using AI for Extraction (Function Calling)
#
# Now for the core of our lab. We will ask an LLM to act as our data extractor. We provide it with two things:
# 1. The **unstructured text** (our lab report).
# 2. The **schema** of the data we want (our `LabReport` Pydantic model).
#
# We will use the **OpenAI Function Calling** feature. We give the model a "tool" (a function) that it can call, and the parameters of that tool are defined by our Pydantic schema.

# + 
def extract_lab_data(report_text: str):
    """
    Uses an OpenAI LLM to parse a lab report and extract structured data.
    """
    client = openai.OpenAI()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly accurate data extraction assistant. Your task is to extract information from a pharmaceutical lab report and structure it according to the provided schema. Do not miss any details."
                },
                {
                    "role": "user",
                    "content": f"Please extract the data from the following lab report:\n\n---\n{report_text}\n---"
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "structure_lab_report",
                        "description": "Formats the extracted lab report data.",
                        "parameters": LabReport.model_json_schema()
                    }
                }
            ],
            tool_choice={"type": "function", "function": {"name": "structure_lab_report"}}
        )
        
        # The model's response includes the arguments for the function call
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            arguments = tool_calls[0].function.arguments
            # Parse the JSON arguments into our Pydantic model
            structured_data = LabReport.model_validate_json(arguments)
            return structured_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Let's run the extraction
if lab_report_text:
    structured_report = extract_lab_data(lab_report_text)
    if structured_report:
        print("Extraction Successful!")
# - 

# ## 5. Analyzing the Structured Output
# 
# If the extraction was successful, `structured_report` is now a Pydantic object. It's no longer just text; it's a Python object with attributes and types that we can work with programmatically.
# 
# Let's inspect the extracted data.

# + 
if 'structured_report' in locals() and structured_report:
    print(f"Batch ID: {structured_report.batch_id}")
    print(f"Product Name: {structured_report.product_name}")
    print("""--------------------""")
    
    # Loop through the extracted tests
    for test in structured_report.tests:
        print(f"Test: {test.test_name}")
        print(f"  - Specification: {test.specification}")
        print(f"  - Result: {test.result}")
        print(f"  - Method: {test.method or 'N/A'}")
    
    print("""--------------------""")
    print(f"Conclusion: {structured_report.conclusion}")
# - 

# ### Exercise 1: Automated Classification
# 
# Now that the data is structured, we can easily build logic on top of it.
# 
# **Your Task:** Write a function `classify_batch(report: LabReport)` that automatically determines if a batch should be **"APPROVED"** or **"FLAGGED FOR REVIEW"**.
# 
# A batch should be flagged if:
# 1. The `conclusion` contains words like "fail", "out of spec", or "does not meet".
# 2. Any test `result` is "Fails" or does not conform.
#
# *Hint: You can check for substrings in the `conclusion` and `result` fields.*

# + 
# YOUR CODE HERE
def classify_batch(report: LabReport) -> str:
    """
    Analyzes a structured lab report to provide a final classification.
    """
    # Check the main conclusion first
    negative_keywords = ["fail", "out of spec", "does not meet"]
    if any(keyword in report.conclusion.lower() for keyword in negative_keywords):
        return "FLAGGED FOR REVIEW"
        
    # Check each individual test result
    for test in report.tests:
        if "fail" in test.result.lower() or "does not conform" in test.result.lower():
            return "FLAGGED FOR REVIEW"
            
    return "APPROVED"

# Test your function
if 'structured_report' in locals() and structured_report:
    classification = classify_batch(structured_report)
    print(f"Automated Classification: {classification}")
# - 

# ### Exercise 2: Convert to a Dictionary or JSON
# 
# The structured data can be easily exported for storage in a database or for sending over an API.
# 
# **Your Task:** Convert the `structured_report` object into a JSON string.
#
# *Hint: Pydantic models have a built-in method for this. Look for a method like `model_dump_json`.*

# + 
if 'structured_report' in locals() and structured_report:
    # YOUR CODE HERE
    report_json = structured_report.model_dump_json(indent=2)
    
    print("Report as JSON:")
    print(report_json)
# - 

# ---
#
# ## Conclusion
#
# Congratulations! You have successfully built an AI pipeline to:
# 1. **Load** unstructured data from a text file.
# 2. **Define** a target data schema using Pydantic.
# 3. **Extract** the data using an LLM with function calling.
# 4. **Analyze** the resulting structured data to make automated decisions.
#
# This approach is significantly more powerful and adaptable than traditional parsing methods and is a foundational technique for building modern AI applications in the pharmaceutical industry.
