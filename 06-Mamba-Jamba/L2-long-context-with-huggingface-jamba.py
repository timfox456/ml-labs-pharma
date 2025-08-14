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

# # Lab 2: Long-Context AI with Hugging Face Jamba
#
# **Goal:** In this lab, you will use the actual Jamba model from Hugging Face to process a lengthy, domain-specific document. This provides a real-world example of using a powerful, open-source, long-context model. 
#
# **Key Concepts:**
# - **Hugging Face Hub:** Loading models and tokenizers directly from the hub.
# - **Jamba Model:** Using a real hybrid (Transformer + Mamba) model for inference.
# - **Long-Context Q&A:** Leveraging the model's large context window to answer questions about a full document.
#
# ---
#
# ## 1. Setup
#
# We need the `transformers` library and its dependencies. We'll also need `accelerate` to help with model loading and `bitsandbytes` for quantization to help fit the model into memory.

# + 
# %pip install transformers torch accelerate bitsandbytes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# - 

# ## 2. Loading the Jamba Model
#
# We will load the Jamba model and its tokenizer directly from the Hugging Face Hub. Jamba is a large model, so we'll use 4-bit quantization (`load_in_4bit=True`) to reduce its memory footprint, which is a common practice for running large models.

# + 
# Define the model identifier
model_id = "ai21labs/Jamba-v0.1"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model
# We use 4-bit quantization to make it runnable on consumer hardware.
# This will download the model, which may take some time.
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
    )
    print("Jamba model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("This may be due to memory constraints. If you are on a GPU with limited VRAM, this model may be too large.")
    model = None
# - 

# ## 3. Loading the Document
#
# We will use the same "Quality by Design (QbD)" document from the previous lab. This allows us to directly compare the experience of using a placeholder model with the real thing.

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

# ## 4. Full-Context Question Answering with Jamba
#
# Now, let's use Jamba's long-context capability to answer a question that requires understanding the entire document. We will format the prompt to clearly separate the context (the document) from the question.
#
# ### Exercise: Ask a Complex Question
#
# **Your Task:** Formulate a prompt to ask Jamba a question about the QbD document. The question should require synthesizing information from different parts of the text.

# + 
def ask_jamba(question, context):
    """
    A helper function to format the prompt and get a response from the Jamba model.
    """
    if not model:
        return "Model not loaded. Cannot proceed."
        
    # Format the prompt with the context and the question
    prompt = f"""
<document>
{context}
</document>

Based on the document provided above, please answer the following question:
{question}
"""
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate the output
    # We set a max length to prevent overly long or runaway responses.
    output = model.generate(**inputs, max_new_tokens=250)
    
    # Decode and print the response
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # The response will include our prompt, so we find and print just the answer part.
    answer_start = response_text.find("answer the following question:") + len("answer the following question:")
    answer = response_text[answer_start:].strip()
    
    return answer

if qbd_document_text and model:
    # YOUR QUESTION HERE
    my_question = "What is the relationship between Critical Quality Attributes (CQAs) and the Design Space?"
    
    print(f"Asking Jamba: {my_question}\n")
    answer = ask_jamba(my_question, qbd_document_text)
    
    print("--- Jamba's Answer ---")
    print(answer)
# - 

# --- 
# ## Conclusion
#
# In this lab, you used a real, state-of-the-art hybrid model (Jamba) to perform a long-context task. You have learned how to:
#
# 1.  Load large, powerful models from the Hugging Face Hub.
# 2.  Use techniques like quantization (`load_in_4bit`) to make these models accessible.
# 3.  Apply a long-context model to a real-world, domain-specific document to answer complex questions.
#
# This hands-on experience is a direct application of the concepts discussed in the slides and demonstrates the practical power of architectures that go beyond the standard Transformer.
