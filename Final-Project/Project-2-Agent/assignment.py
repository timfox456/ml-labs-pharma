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

# # Final Project 2: Drug Review Analysis Agent
#
# **Objective:** Your goal is to build an AI agent that can perform a complex, multi-step analysis of a drug review dataset. This project goes "Beyond RAG" by requiring the LLM to act as a reasoning engine that uses tools to gather information and synthesize a final answer.
#
# **Estimated Time:** 1 hour
#
# **Core Task:** You will create a set of Python functions that act as "tools" for an agent. You will then build a LangChain agent that can use these tools to answer a multi-hop question that a simple RAG system could not handle.
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
# Now, let's import the necessary libraries and load the data.

# + import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Load the dataset into a pandas DataFrame
try:
    df = pd.read_csv("./data/drug_reviews.csv")
    print("Dataset loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print("Error: data/drug_reviews.csv not found.")
    df = None
# - 

# ## 2. The Challenge: A Multi-Step Analysis
#
# A user wants to know the following:
# > "For the condition 'High Blood Pressure', what are the top 2 drugs with the highest average ratings? For each of those two drugs, provide a summary of the most common positive comments."
#
# A standard RAG system would fail here because it requires multiple steps and calculations:
# 1.  Filter the dataset by condition.
# 2.  Calculate the average rating for each drug in the filtered set.
# 3.  Identify the top 2 drugs.
# 4.  For each of the top drugs, retrieve all their positive reviews.
# 5.  Summarize those positive reviews.
#
# Your task is to build an agent that can perform this reasoning.
#
# ---
#
# ## 3. Create Your Tools
#
# An agent needs tools to interact with the world (in this case, our DataFrame). You need to create a set of Python functions that the agent can call.
#
# **Your Task:**
# - Complete the five functions below.
# - Add the `@tool` decorator to each one so that LangChain recognizes it as a tool.
# - The first tool is provided for you as an example.

# + 
# ---
# YOUR CODE HERE
# -

@tool
def get_drugs_for_condition(condition: str) -> str:
    """Returns a list of drugs used for a specific condition."""
    if df is not None:
        drugs = df[df['condition'].str.lower() == condition.lower()]['drugName'].unique()
        return f"The drugs for {condition} are: {', '.join(drugs)}"
    return "Dataset not loaded."

# @tool
def get_average_rating(drug_name: str) -> str:
    """Returns the average rating for a specific drug."""
    # Hint: Filter the DataFrame for the drug_name and calculate the mean of the 'rating' column.
    # Return a formatted string like "The average rating for [drug_name] is [avg_rating]."
    pass

# @tool
def get_reviews(drug_name: str, sentiment: str = 'positive') -> str:
    """
    Returns patient reviews for a specific drug.
    Sentiment can be 'positive' (rating > 7), 'negative' (rating < 4), or 'all'.
    """
    # Hint: Filter the DataFrame by drug_name.
    # Then, based on the 'sentiment' parameter, filter by the 'rating' column.
    # Join the 'review' texts into a single string and return it.
    pass

# @tool
def summarize_text(text: str) -> str:
    """Summarizes a long piece of text into a few key points."""
    # Hint: This tool will need to call an LLM.
    # Create an instance of the OpenAI LLM and pass it a prompt that includes the text to be summarized.
    pass

# Create a list of all your tools
tools = [] # Add your completed tool functions to this list

# ---
# END YOUR CODE
# -

# ## 4. Build the Agent
#
# Now that you have your tools, you need to create the agent that can use them.
#
# **Your Task:**
# - Create a `PromptTemplate` using the provided `prompt_template_string`.
# - Create a ReAct agent using `create_react_agent`.
# - Create an `AgentExecutor` to run the agent.

# + 
# Starter Code
prompt_template_string = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

# ---
# YOUR CODE HERE
# -

# 1. Create the Prompt Template
prompt = None # Replace None

# 2. Create the Agent
agent = None # Replace None

# 3. Create the Agent Executor
agent_executor = None # Replace None

# ---
# END YOUR CODE
# -

# ## 5. Run and Test Your Agent
#
# Now, let's give the agent the complex question and see if it can reason its way to the answer.
#
# **Your Task:**
# - Define the `complex_question` string.
# - Invoke the `agent_executor` with the question.
# - Print the final output.

# + 
# ---
# YOUR CODE HERE
# -

complex_question = "For the condition 'High Blood Pressure', what are the top 2 drugs with the highest average ratings? For each of those two drugs, provide a summary of the most common positive comments."

response = None # Invoke the agent

print("\n--- Agent's Final Response ---")
# print(response['output']) # The final answer is in the 'output' key of the response dictionary

# ---
# END YOUR CODE
# -

# **Expected Output:**
# If your agent works correctly, you should see a detailed thought process (because `verbose=True` is the default for `AgentExecutor`). The final answer should identify Valsartan and Guanfacine as the top drugs and provide a summary of their positive reviews.
#
# ---
#
# ## Conclusion
#
# You have successfully built an agentic system that goes beyond simple RAG. You learned how to:
# 1.  Decompose a complex problem into smaller, solvable steps.
# 2.  Encapsulate those steps into tools (functions) that an LLM can use.
# 3.  Build a reasoning agent that can plan and execute a series of actions to find an answer.
#
# This agent-and-tool-use pattern is fundamental to solving complex, real-world problems with AI.
