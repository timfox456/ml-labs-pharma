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

# # Lab: CNN vs. Vision Transformer (ViT) for Image Classification
#
# **Goal:** In this lab, you will build, train, and evaluate two different deep learning architectures—a classic Convolutional Neural Network (CNN) and a modern Vision Transformer (ViT)—on the same image classification task. This will provide a direct comparison of their performance, complexity, and training characteristics.
#
# **Key Concepts:**
# - **CNN Architecture:** Understanding the role of Conv2D and MaxPooling layers.
# - **ViT Architecture:** Understanding how Transformers process images as sequences of patches.
# - **Comparative Analysis:** Evaluating models based on metrics like accuracy, parameter count, and training time.
#
# **Prerequisites:**
# - You must have completed the `08-Vision` lab and have the image dataset available.
#
# ---
#
# ## 1. Setup and Data Preparation
#
# First, we need to set up our environment and prepare the dataset.
#
# **ACTION REQUIRED:** This lab requires the same image dataset from the `08-Vision` lab. Please **copy or symlink** the `data` directory from `labs/02-HF/08-Vision/` into this lab's directory (`labs/02-HF/10-CNN-vs-ViT/`).
#
# ```bash
# # From the root of the project, run:
# cp -r labs/02-HF/08-Vision/data labs/02-HF/10-CNN-vs-ViT/
# ```

# + 
# %pip install tensorflow transformers datasets scikit-learn Pillow torch accelerate

import os
import time
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow imports for the CNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# PyTorch and Hugging Face imports for the ViT
import torch
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
# - 

# ### Load the Dataset
# We will load the dataset once. We'll use the Hugging Face `datasets` library as it's convenient, and then convert it to a TensorFlow `tf.data.Dataset` for the CNN part.

# + 
try:
    # Load with Hugging Face datasets
    ds = load_dataset("imagefolder", data_dir="data")
    ds = ds['train'].train_test_split(test_size=0.2, seed=42)
    
    labels = ds["train"].features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    NUM_LABELS = len(labels)
    
    print(f"Dataset loaded successfully. Labels: {labels}")
except Exception as e:
    print(f"Could not load the dataset. Please ensure you have copied the images into the 'data' folder. Error: {e}")
    ds = None
# - 

# ---
# ## 2. Model 1: The Convolutional Neural Network (CNN)
#
# First, we'll build and train a standard CNN using TensorFlow/Keras.
#
# ### Prepare Data for TensorFlow

# + 
IMG_SIZE = 224 # Standard size for many vision models
BATCH_SIZE = 8 # Small batch size for limited memory

def preprocess_cnn(image, label):
    """Function to preprocess images for the CNN."""
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

if ds:
    # Convert Hugging Face dataset to TensorFlow dataset
    train_ds_cnn = ds['train'].to_tf_dataset(columns='image', label_cols='label', batch_size=BATCH_SIZE, shuffle=True)
    test_ds_cnn = ds['test'].to_tf_dataset(columns='image', label_cols='label', batch_size=BATCH_SIZE)

    # Apply preprocessing
    train_ds_cnn = train_ds_cnn.map(preprocess_cnn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds_cnn = test_ds_cnn.map(preprocess_cnn, num_parallel_calls=tf.data.AUTOTUNE)
# - 

# ### Build and Train the CNN

# + 
if 'train_ds_cnn' in locals():
    # Define the CNN model
    cnn_model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_LABELS, activation='softmax')
    ])

    cnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("--- CNN Model Summary ---")
    cnn_model.summary()

    # Train the model
    print("\n--- Training CNN ---")
    start_time = time.time()
    cnn_history = cnn_model.fit(train_ds_cnn, epochs=5, validation_data=test_ds_cnn)
    cnn_training_time = time.time() - start_time
    print(f"CNN training finished in {cnn_training_time:.2f} seconds.")
# - 

# ---
# ## 3. Model 2: The Vision Transformer (ViT)
#
# Now, we'll build and train a ViT using Hugging Face Transformers. This code is adapted from the `08-Vision` lab.
#
# ### Prepare Data for ViT

# + 
if ds:
    model_checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = ViTImageProcessor.from_pretrained(model_checkpoint)

    def transform_vit(example_batch):
        inputs = image_processor([x for x in example_batch['image']], return_tensors='pt')
        inputs['label'] = example_batch['label']
        return inputs

    prepared_ds_vit = ds.with_transform(transform_vit)

    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['label'] for x in batch])
        }
# - 

# ### Build and Train the ViT

# + 
if 'prepared_ds_vit' in locals():
    vit_model = ViTForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir="vit_vs_cnn_model",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=vit_model,
        args=training_args,
        train_dataset=prepared_ds_vit["train"],
        eval_dataset=prepared_ds_vit["test"],
        data_collator=collate_fn,
        compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))},
        tokenizer=image_processor,
    )

    print("\n--- Training ViT ---")
    start_time = time.time()
    trainer.train()
    vit_training_time = time.time() - start_time
    print(f"ViT training finished in {vit_training_time:.2f} seconds.")
# - 

# ---
# ## 4. Comparison and Analysis
#
# Now that both models are trained, let's compare them on key characteristics.
#
# ### Exercise: Compare the Models
#
# **Your Task:** Evaluate both models on their test sets and fill in the comparison table below with the results.

# + 
# Evaluate CNN
if 'cnn_model' in locals():
    cnn_loss, cnn_accuracy = cnn_model.evaluate(test_ds_cnn)
    cnn_params = cnn_model.count_params()
else:
    cnn_accuracy, cnn_params = 0, 0

# Evaluate ViT
if 'trainer' in locals():
    vit_eval_results = trainer.evaluate()
    vit_accuracy = vit_eval_results['eval_accuracy']
    vit_params = vit_model.num_parameters()
else:
    vit_accuracy, vit_params = 0, 0

print("--- Comparison Results ---")
print(f"CNN Model:")
print(f"  - Test Accuracy: {cnn_accuracy:.4f}")
print(f"  - Training Time: {cnn_training_time:.2f}s")
print(f"  - Total Parameters: {cnn_params:,}")

print(f"\nViT Model:")
print(f"  - Test Accuracy: {vit_accuracy:.4f}")
print(f"  - Training Time: {vit_training_time:.2f}s")
print(f"  - Total Parameters: {vit_params:,}")
# - 

# ### Analysis Questions
#
# 1.  **Accuracy:** Which model performed better on the test set? (Note: With small datasets and few epochs, results can vary. ViTs often need more data than CNNs to shine).
# 2.  **Model Size:** Look at the "Total Parameters." Which model is larger? What does this imply about its capacity and potential for overfitting?
# 3.  **Training Time:** Which model trained faster? Why might this be the case? (Consider the complexity of the operations in each architecture).
#
# ### Architectural Differences
#
# - **CNNs** use an **inductive bias** for images. They are hard-wired to look for local patterns (edges, textures) and build up a hierarchical understanding. This makes them very data-efficient.
# - **ViTs** treat an image as a generic sequence of patches. They have less built-in bias and learn the relationships between patches from scratch. This makes them incredibly powerful but often requires more data to learn effectively.
#
# ---
#
# ## Conclusion
#
# In this lab, you directly compared two powerful but fundamentally different vision architectures. You saw that there is no single "best" model; the choice depends on the task, the amount of data available, and the computational resources.
#
# - For smaller, specific datasets, a well-tuned **CNN** can be highly effective and efficient.
# - For larger, more diverse datasets, a pre-trained **ViT** can often achieve state-of-the-art performance due to its ability to learn global relationships between different parts of an image.

