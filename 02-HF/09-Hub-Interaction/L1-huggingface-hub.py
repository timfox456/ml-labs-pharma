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

# # Lab: Interacting with the Hugging Face Hub
#
# **Goal:** In this lab, you will learn how to share your fine-tuned models with the community by pushing them to the Hugging Face Hub. This is the standard way to save, version, and share models. 
#
# **Key Concepts:**
# - **Authentication:** Securely logging into your Hugging Face account from a notebook.
# - **Model Repositories:** Understanding that every model on the Hub is a Git repository.
# - **Pushing to Hub:** Uploading your model files, tokenizer configuration, and model card to a repository.
#
# **Prerequisites:**
# - You must have a Hugging Face account. If you don't, create one at [huggingface.co](https://huggingface.co/).
# - You must have completed the `07-Fine-Tuning` lab and have a saved model in `../07-Fine-Tuning/pharma_classifier_model`.
#
# ---
#
# ## 1. Setup
#
# We need the `transformers` and `huggingface_hub` libraries.

# + tags=['startup']
# %pip install transformers huggingface_hub

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import notebook_login
# - 

# ## 2. Logging into the Hugging Face Hub
#
# To push a model, you first need to authenticate. The `notebook_login` helper function provides a simple widget to do this.
#
# **ACTION REQUIRED:**
# 1.  Go to your Hugging Face account settings: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
# 2.  Create a new **Access Token** with the `write` role.
# 3.  Run the cell below and paste your token into the widget.

# + 
notebook_login()
# - 

# ## 3. Loading Your Fine-Tuned Model
#
# Before we can upload the model, we need to load it from the directory where our `Trainer` saved it in the previous lab.

# + 
# Path to the fine-tuned model checkpoint
# Note: You may need to adjust the checkpoint number
model_path = "../07-Fine-Tuning/pharma_classifier_model/checkpoint-9" 
repo_name = "pharma-text-classifier" # Choose a name for your model on the Hub

try:
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Successfully loaded fine-tuned model and tokenizer.")
except OSError:
    print(f"Error: Model not found at '{model_path}'. Please ensure you have run the 07-Fine-Tuning lab.")
    model = None
# - 

# ## 4. Pushing the Model to the Hub
#
# With the model loaded and authenticated, pushing it to the Hub is a single command. The `push_to_hub` method will:
# 1.  Create a new repository under your username with the name you provide (`repo_name`).
# 2.  Upload the model weights (`.bin` file), configuration (`config.json`), and tokenizer files.
# 3.  Create a basic `README.md` file (the model card).

# + 
if model:
    try:
        # Push the model to the hub
        print(f"Pushing model to repository: {repo_name}")
        model.push_to_hub(repo_name)
        print("Model push successful!")
        
        # Push the tokenizer to the same repository
        print(f"Pushing tokenizer to repository: {repo_name}")
        tokenizer.push_to_hub(repo_name)
        print("Tokenizer push successful!")
        
        # Get your username to construct the full model ID
        from huggingface_hub import HfApi
        user = HfApi().whoami()['name']
        model_id = f"{user}/{repo_name}"
        print(f"\nVisit your model on the Hub at: https://huggingface.co/{model_id}")
        
    except Exception as e:
        print(f"An error occurred during the push: {e}")
# - 

# ## 5. Verifying the Upload
#
# The ultimate test is to load the model *back* from the Hub, just as any other user would.
#
# ### Exercise: Load Your Model from the Hub
#
# **Your Task:** Use the `from_pretrained` method to load your newly uploaded model directly from the Hub using its full ID (e.g., `"YourUsername/pharma-text-classifier"`).

# + 
from transformers import pipeline

# YOUR CODE HERE
# Replace this with your actual model ID
if 'model_id' in locals():
    print(f"Loading model '{model_id}' from the Hub...")
    
    # Load the model into a pipeline
    hub_classifier = pipeline("text-classification", model=model_id)
    
    # Test it
    test_sentence = "A new batch record was initiated."
    result = hub_classifier(test_sentence)
    
    print(f"\nSentence: '{test_sentence}'")
    print(f"Prediction from Hub model: {result}")
else:
    print("Model ID not found. Please complete the previous step to push your model.")
# - 

# ---
# ## Conclusion
#
# Congratulations! You have successfully shared a model on the Hugging Face Hub. This is a fundamental skill for any modern AI practitioner. You have learned to:
# 
# 1.  Securely authenticate with the Hub from a notebook.
# 2.  Use `push_to_hub` to create a model repository and upload your files.
# 3.  Load a model back from the Hub to verify that it's accessible to everyone.
#
# Your fine-tuned model is now a part of the public AI ecosystem!
