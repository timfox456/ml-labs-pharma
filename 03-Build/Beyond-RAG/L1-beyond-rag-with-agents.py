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

# # Lab: Beyond RAG - Agentic Systems for Complex Tasks
#
# **Goal:** In this lab, you will explore the limitations of standard Retrieval-Augmented Generation (RAG) and learn how to solve more complex problems using an **agentic approach** with tools.
#
# **Key Concepts:**
# - **RAG Limitations:** Understanding that RAG is for knowledge retrieval, not multi-step reasoning.
# - **Multi-hop Questions:** Identifying questions that require finding and connecting information from multiple sources.
# - **Agents and Tools:** Using LLMs as reasoning engines that can use tools (functions) to gather information and solve problems step-by-step.
#
# ---
#
# ## 1. Setup
#
# We will need several libraries from the LangChain ecosystem to build our RAG and Agent systems. We'll use a simple in-memory vector store for this lab.

# + # %pip install langchain langchain-openai python-dotenv faiss-cpu

import os
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
# Make sure your OPENAI_API_KEY is set in your .env file
# - 

# ## 2. The Challenge: A Multi-Document Q&A
#
# Our data is spread across three different documents: project descriptions, project assignments, and employee contact info. Our goal is to answer a question that requires information from all three.
#
# ### Load the Documents

# + # Load all documents from the data directory
loader_projects = TextLoader("./data/project_docs.txt")
documents_projects = loader_projects.load()

loader_assignments = TextLoader("./data/assignment_docs.txt")
documents_assignments = loader_assignments.load()

loader_employees = TextLoader("./data/employee_docs.txt")
documents_employees = loader_employees.load()

all_documents = documents_projects + documents_assignments + documents_employees

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(all_documents)

print(f"Loaded and split {len(texts)} document chunks.")
# - 

# ## 3. Approach 1: Standard RAG
#
# Let's build a standard RAG pipeline. We will embed all the document chunks into a vector store and use a retrieval chain to answer questions.

# + # Create embeddings and the vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
# - 

# ### Test RAG on a Simple Question
# A simple question that can be answered from a single retrieved document should work perfectly.

# + # simple_question = "What is the description of Project Alpha?"
# print(f"Asking RAG: {simple_question}")
# response = rag_chain.invoke(simple_question)
# print(f"Response: {response['result']}")
# - 

# ### Test RAG on a Multi-Hop Question
# Now, let's ask the question that requires "joining" information across all three documents.

# + # complex_question = "What is the email address of the manager for Project Alpha?"
# print(f"Asking RAG: {complex_question}")
# response = rag_chain.invoke(complex_question)
# print(f"Response: {response['result']}")
# - 

# ### Why did RAG fail?
#
# The RAG pipeline likely failed or gave a vague answer. Here's why:
# 1.  The query "email address of the manager for Project Alpha" is semantically closest to the `project_docs.txt` chunk about Project Alpha.
# 2.  The retriever finds that chunk, which says nothing about a manager or an email.
# 3.  The LLM is then asked to answer the question based *only* on that retrieved context and cannot find the answer.
#
# RAG is a **fixed pipeline**: `retrieve -> generate`. It cannot perform the multi-step reasoning required:
# - **Step 1:** Find the manager of Project Alpha (Alice Williams).
# - **Step 2:** Find the email for Alice Williams.
#
# ---
#
# ## 4. Approach 2: An Agent with Tools
#
# An agent is not a fixed pipeline. It's a reasoning loop (`think -> act -> observe -> think ...`). We can give it **tools** to find the information it needs, step-by-step.
#
# ### Define Our Tools
# We'll create simple Python functions that act as our "APIs" to the data. The `@tool` decorator makes them usable by a LangChain agent.

# + # For simplicity, we'll just search the raw text of our documents.
# # In a real system, these tools would query databases or APIs.

# @tool
# def get_project_details(project_name: str) -> str:
#     """Returns the description and status of a given project."""
#     for doc in documents_projects:
#         if project_name.lower() in doc.page_content.lower():
#             return doc.page_content
#     return "Project not found."

# @tool
# def get_manager_of_project(project_name: str) -> str:
#     """Returns the name of the manager for a given project."""
#     for doc in documents_assignments:
#         if project_name.lower() in doc.page_content.lower():
#             # Simple parsing for this example
#             parts = doc.page_content.split(" is managed by ")
#             return parts[1].replace('.', '')
#     return "Manager not found for that project."

# @tool
# def get_contact_info(employee_name: str) -> str:
#     """Returns the title and email address for a given employee."""
#     for doc in documents_employees:
#         if employee_name.lower() in doc.page_content.lower():
#             return doc.page_content
#     return "Employee contact info not found."

# tools = [get_project_details, get_manager_of_project, get_contact_info]
# - 

# ### Build the Agent
# We'll use a standard ReAct (Reasoning and Acting) agent. We provide it with the tools and a prompt that tells it how to think and use them.

# + # The prompt template tells the agent how to reason and use the tools.
# prompt_template = """
# Answer the following questions as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}
# """

# prompt = PromptTemplate.from_template(prompt_template)

# # Create the agent
# agent = create_react_agent(OpenAI(temperature=0), tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# - 

# ### Run the Agent on the Complex Question
# Now, let's give the agent the same multi-hop question that RAG failed on.

# + # response = agent_executor.invoke({"input": complex_question})
# print("\n--- Agent's Final Response ---")
# print(response['output'])
# - 

# ### Why did the Agent succeed?
#
# Look at the `verbose=True` output. You can see the agent's step-by-step reasoning:
# 1.  **Thought:** "I need to find the manager of Project Alpha."
# 2.  **Action:** `get_manager_of_project`, Input: `"Project Alpha"`
# 3.  **Observation:** "Alice Williams"
# 4.  **Thought:** "Now I have the manager's name. I need to find the contact info for Alice Williams."
# 5.  **Action:** `get_contact_info`, Input: `"Alice Williams"`
# 6.  **Observation:** "Employee: Alice Williams..."
# 7.  **Thought:** "I have all the information. The email is a.williams@aims_pharma.com."
# 8.  **Final Answer:** "The email address of the manager for Project Alpha is a.williams@aims_pharma.com."
#
# ---
#
# ## Conclusion
#
# This lab demonstrates the critical difference between RAG and Agentic systems:
#
# -   **RAG** is a powerful pattern for **knowledge retrieval**. It's best for when the answer to a question is likely contained within a single, retrievable chunk of text.
# -   **Agents** are a more general and powerful pattern for **problem-solving**. They are best for complex tasks that require planning, multi-step reasoning, and the ability to "join" information from different tools or sources.
#
# Understanding when to use each pattern is key to building effective and robust AI applications.
