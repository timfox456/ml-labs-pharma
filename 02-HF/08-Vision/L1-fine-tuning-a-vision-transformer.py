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

# # Lab: Fine-Tuning a Vision Transformer (ViT) for Image Classification
#
# **Goal:** In this lab, you will apply your fine-tuning skills to a new domain: computer vision. You will fine-tune a Vision Transformer (ViT) to classify images of different pharmaceutical dosage forms (pills, capsules, tablets).
#
# **Key Concepts:**
# - **Vision Transformer (ViT):** Understanding how the Transformer architecture can be applied to images.
# - **Image Data Loading:** Using the `datasets` library to work with image folder datasets.
# - **Feature Extraction:** Processing and augmenting images to prepare them for the model.
#
# ---
#
# ## 1. Setup
#
# We need the same core libraries as before, but with the addition of `Pillow` for image processing.

# + tags=['skip_test']
# %pip install transformers datasets accelerate scikit-learn torch Pillow

import torch
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# - 

# ## 2. Preparing the Dataset
#
# **ACTION REQUIRED:** This lab requires an image dataset. Please source your own images and place them in the following directory structure:
#
# ```
# labs/02-HF/08-Vision/data/
# ├── pills/
# │   ├── pill_01.jpg
# │   └── pill_02.jpg
# │   ...
# ├── capsules/
# │   ├── capsule_01.jpg
# │   └── capsule_02.jpg
# │   ...
# └── tablets/
#     ├── tablet_01.jpg
#     └── tablet_02.jpg
#     ...
# ```
#
# For a quick test, you can find 5-10 images for each category from a stock photo website.
#
# ---
#
# Once your data is in place, we can load it using the `datasets` library's `ImageFolder` feature, which automatically infers labels from the folder names.

# + tags=['skip_test']
# Load the dataset from the image folders
try:
    dataset = load_dataset("imagefolder", data_dir="data")
    print("Dataset loaded successfully:")
    print(dataset)

    # Split the 'train' split into a 80% train and 20% test set
    dataset = dataset['train'].train_test_split(test_size=0.2)
    print("\nDataset after splitting:")
    print(dataset)

    # Get the label names
    labels = dataset["train"].features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    print(f"\nLabels: {id2label}")

except Exception as e:
    print(f"Could not load the dataset. Please ensure your images are in the correct folder structure. Error: {e}")
    dataset = None
# - 

# ## 3. Preprocessing the Images
#
# Just like text, images need to be preprocessed. A `ViTImageProcessor` (also called a Feature Extractor) will:
# 1.  Resize the image to the model's expected size.
# 2.  Normalize the pixel values.
# 3.  Convert the image into a tensor.

# + tags=['skip_test']
# Load the feature extractor for the model we want to fine-tune
model_checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = ViTImageProcessor.from_pretrained(model_checkpoint)

# Create a transformation function
def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = image_processor([x for x in example_batch['image']], return_tensors='pt')
    # Don't forget to include the labels!
    inputs['label'] = example_batch['label']
    return inputs

if dataset:
    # Apply the transformation
    prepared_ds = dataset.with_transform(transform)
# - 

# ## 4. The Fine-Tuning Process
#
# This process is nearly identical to the text fine-tuning lab, demonstrating the power and consistency of the Hugging Face ecosystem.
#
# ### Loading the Model
# We load `ViTForImageClassification`, providing it with the number of labels and the label mappings from our dataset.

# + tags=['skip_test']
if 'labels' in locals():
    model = ViTForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True # Needed to replace the head of the pre-trained model
    )
# - 

# ### Training Arguments and Metrics
# These are the same as before. We define our training configuration and a function to compute accuracy and F1-score.

# + tags=['skip_test']
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def collate_fn(batch):
    """A custom collator to handle the already-tensorized image data."""
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

training_args = TrainingArguments(
    output_dir="pharma_image_classifier_model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False,
)
# - 

# ### The Trainer
# We bring everything together in the `Trainer` object. Note the use of a custom `collate_fn` because our data is already processed into tensors.

# + tags=['skip_test']
if 'prepared_ds' in locals():
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["test"],
        tokenizer=image_processor, # For vision models, the processor is passed here
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
# - 

# ### Let's Train!

# + tags=['skip_test']
if 'trainer' in locals():
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")
# - 

# ## 5. Using the Fine-Tuned Model
#
# Now you can use your specialized image classifier!
#
# ### Exercise: Test Your Model
#
# **Your Task:** Find one new image of a pill, capsule, or tablet that was **not** in your training data. Use your fine-tuned model to classify it.

# + tags=['skip_test']
from transformers import pipeline
from PIL import Image

# YOUR CODE HERE
# 1. Provide the path to your new test image
#    (You will need to upload this image to your lab environment)
# test_image_path = "path/to/your/new_image.jpg"

# try:
#     # 2. Load the image
#     image = Image.open(test_image_path)

#     # 3. Create a pipeline with your fine-tuned model
#     if 'trainer' in locals():
#         # Adjust checkpoint number based on your training output
#         vision_classifier = pipeline("image-classification", model="pharma_image_classifier_model/checkpoint-500") 

#         # 4. Get the prediction
#         prediction = vision_classifier(image)

#         print(f"Image: '{test_image_path}'")
#         print(f"Predicted Label: {prediction}")

# except FileNotFoundError:
#     print(f"Please update the 'test_image_path' to a valid image file.")
# - 

# ---
# ## Conclusion
#
# You have now successfully fine-tuned a Transformer model for both text and vision! This lab demonstrated that the core workflow is remarkably consistent across different modalities. You learned to:
#
# 1.  Load a custom image dataset from folders.
# 2.  Use a `ViTImageProcessor` to prepare images for the model.
# 3.  Fine-tune a ViT model using the same `Trainer` API you used for text.
# 4.  Use the final, specialized model for inference on new images.
#
# This powerful and flexible workflow allows you to adapt state-of-the-art models to a vast range of tasks in the pharmaceutical domain, from document analysis to visual quality control.
