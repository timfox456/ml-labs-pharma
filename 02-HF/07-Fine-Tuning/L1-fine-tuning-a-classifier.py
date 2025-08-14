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

# # Lab: Fine-Tuning a Transformer for Text Classification
#
# **Goal:** In this lab, you will learn the most important skill for adapting AI to a specific domain: **fine-tuning**. You will take a general-purpose pre-trained model and fine-tune it on a custom, pharma-related dataset to create a specialized classifier.
#
# **Key Concepts:**
# - **Datasets Library:** The standard way to load and process data in the Hugging Face ecosystem.
# - **Fine-Tuning:** The process of updating the weights of a pre-trained model on a new, specific task.
# - **Trainer API:** A high-level API that handles the entire training loop, including optimization, evaluation, and logging.
#
# ---
#
# ## 1. Setup
#
# We need several libraries from the Hugging Face ecosystem, including `datasets` to load our data and `accelerate` to speed up training. We also need `scikit-learn` to compute metrics.

# + # %pip install transformers datasets accelerate scikit-learn torch

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# - 

# ## 2. Loading the Custom Dataset
#
# The `datasets` library can load data from many sources, including the Hub and local files like CSVs, JSON, etc. We will load our custom `pharma_text_classification.csv` file.

# + # Load the dataset from the local CSV file
try:
    dataset = load_dataset('csv', data_files='data/pharma_text_classification.csv')
    print("Dataset loaded successfully:")
    print(dataset)
except FileNotFoundError:
    print("Error: Make sure 'data/pharma_text_classification.csv' exists.")
    dataset = None
# - 

# The dataset is currently a single split called `train`. We need to split it into a `train` set and a `test` set to properly evaluate our model's performance on unseen data.

# + 
if dataset:
    # Split the 'train' split into a 80% train and 20% test set
    dataset = dataset['train'].train_test_split(test_size=0.2)
    print("\nDataset after splitting:")
    print(dataset)
# - 

# ## 3. Preprocessing the Data
#
# Our model cannot process raw text. We need to **tokenize** it, converting the text into the numerical IDs the model expects. We'll use the tokenizer from the model we plan to fine-tune.

# + # Load the tokenizer for the model we want to fine-tune
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Create a function to tokenize the text
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

if dataset:
    # Apply the tokenization to the entire dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print("\nTokenized dataset sample:")
    print(tokenized_datasets["train"][0])
# - 

# ## 4. The Fine-Tuning Process
#
# Now we are ready to set up and run the fine-tuning job.
#
# ### Loading the Model
# We load `AutoModelForSequenceClassification`. It's crucial that we tell it how many labels we have. We also provide mappings from label IDs (0, 1, 2) to human-readable names.

# + # Define the label mappings
id2label = {0: "Regulatory", 1: "Manufacturing", 2: "Clinical"}
label2id = {"Regulatory": 0, "Manufacturing": 1, "Clinical": 2}

# Load the model with the correct number of labels and the mappings
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=3, id2label=id2label, label2id=label2id
)
# - 

# ### Defining Metrics
# During training, we want to see more than just the loss; we want to see human-understandable metrics like accuracy. We define a function to compute these metrics.

# + 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
# - 

# ### Training Arguments
# The `TrainingArguments` class lets us configure every aspect of the training process, such as the learning rate, number of epochs, and how often to save or evaluate the model.

# + # Define the training arguments
training_args = TrainingArguments(
    output_dir="pharma_classifier_model", # Where to save the model
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch",       # Save at the end of each epoch
    load_best_model_at_end=True,
)
# - 

# ### The Trainer
# The `Trainer` object brings everything together: the model, the arguments, the datasets, the tokenizer, and the metrics function.

# + 
if 'tokenized_datasets' in locals():
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
# - 

# ### Let's Train!
# Now, we just call `trainer.train()`. The Trainer API handles everything else.

# + 
if 'trainer' in locals():
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")
# - 

# ## 5. Using the Fine-Tuned Model
#
# The best version of our model has been saved to the `pharma_classifier_model` directory. We can now use it for inference just like any other model from the Hub, for example, by loading it into a `pipeline`.
#
# ### Exercise: Test Your Model
#
# **Your Task:** Load your fine-tuned model into a `text-classification` pipeline and test it on a new sentence. See if it correctly classifies the domain.

# + 
from transformers import pipeline

# YOUR CODE HERE
# 1. Define a new sentence to classify
new_sentence = "The new drug application was submitted to the regulatory authority."

# 2. Create a pipeline with your fine-tuned model
# The `model` argument should be the path to your saved model.
if 'trainer' in locals():
    finetuned_classifier = pipeline("text-classification", model="pharma_classifier_model/checkpoint-9") # Adjust checkpoint number if needed

    # 3. Get the prediction
    prediction = finetuned_classifier(new_sentence)

    print(f"Sentence: '{new_sentence}'")
    print(f"Predicted Label: {prediction}")
# - 

# ---
# ## Conclusion
#
# Congratulations! You have successfully fine-tuned a general-purpose Transformer model to become a specialist in classifying pharmaceutical texts. You have learned the standard workflow used by practitioners everywhere:
#
# 1.  Load a custom dataset using `datasets`.
# 2.  Preprocess and tokenize the data.
# 3.  Set up the `Trainer` with a model, data, and training arguments.
# 4.  Run the training loop.
# 5.  Use the final, specialized model for inference.
#
# This process is the key to unlocking the power of large language models for your specific domain and use cases.
