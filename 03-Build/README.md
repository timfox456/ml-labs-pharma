# AIMS Pharma - Final Projects

This directory contains three final projects designed to test and integrate the concepts learned throughout the "AI for Developers in the Pharmaceutical Industry" course. Each project is a self-contained, one-hour coding assignment that tackles a realistic, domain-specific problem.

## Getting Started

Before beginning any project, you must set up a dedicated Python environment for it. This ensures you have the correct dependencies and avoids conflicts.

For each project directory (`Project-1-RAG`, `Project-2-Agent`, `Project-3-Parser`):

1.  **Navigate into the project directory:**
    ```bash
    cd Project-1-RAG  # Or Project-2-Agent, etc.
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your OpenAI API Key:**
    - Create a file named `.env` in the main `aims_pharma` directory (or ensure the existing one is there).
    - Add the following line to the `.env` file, replacing `your_key_here` with your actual key:
      ```
      OPENAI_API_KEY="your_key_here"
      ```

5.  **Start Jupyter Lab or Notebook** and open the `assignment.ipynb` file to begin.

---

## Project 1: Clinical Paper Q&A with RAG

-   **Directory:** `Project-1-RAG/`
-   **Goal:** Build a question-answering system that helps a researcher quickly get answers from a collection of clinical paper abstracts on diabetes.
-   **Concepts Integrated:** Retrieval-Augmented Generation (RAG), Embeddings, Vector Stores (FAISS), LangChain QA Chains, Prompt Engineering.
-   **Files:**
    -   `assignment.py/.ipynb`: The student notebook with instructions and starter code.
    -   `solution.py/.ipynb`: The complete instructor's solution.
    -   `data/diabetes_clinical_papers.csv`: The knowledge base for the RAG system.

---

## Project 2: Drug Review Analysis Agent

-   **Directory:** `Project-2-Agent/`
-   **Goal:** Build an AI agent that can perform a complex, multi-step analysis on a dataset of patient drug reviews. This project goes "Beyond RAG" by requiring the agent to use multiple tools in a sequence to solve a problem.
-   **Concepts Integrated:** Agents and Tools, Multi-hop Reasoning, Prompt Engineering, Summarization, Data Analysis with Pandas.
-   **Files:**
    -   `assignment.py/.ipynb`: The student notebook with instructions and starter code.
    -   `solution.py/.ipynb`: The complete instructor's solution.
    -   `data/drug_reviews.csv`: A sample dataset of drug reviews.

---

## Project 3: Clinical Trial Report Parser

-   **Directory:** `Project-3-Parser/`
-   **Goal:** Build a system to automatically extract structured information (like the drug name, trial phase, and outcome) from unstructured, narrative-style press releases about clinical trials.
-   **Concepts Integrated:** Structured Data Extraction, LLM Function Calling, Pydantic for data validation and schema definition, Prompt Engineering.
-   **Files:**
    -   `assignment.py/.ipynb`: The student notebook with instructions and starter code.
    -   `solution.py/.ipynb`: The complete instructor's solution.
    -   `data/`: Contains two `.txt` files with fictional press releases.
