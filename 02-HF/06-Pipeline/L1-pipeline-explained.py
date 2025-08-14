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

# # Pipeline Explained
#
# In this lab we will go behind the `pipeline` function and see how it works.
#
# #### Lab Goals:
#
# * Go deeper into the Pipeline.
# * Investigate what is going on behind the scenes.
#
# ---

# ## Step 1: Repeat sentiment analysis
#
# This is the same high-level abstraction from the previous lab.

# +
# %pip install transformers torch
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
# -

# * You should obtain this output:
#
# ```text
# [{'label': 'POSITIVE', 'score': 0.9598047137260437},
#  {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
# ```
#
# ---
# ## Behind the Scenes: Tokenizer, Model, and Post-processing
#
# The `pipeline` function is composed of three main steps:
# 1.  **Tokenizer**: Converts the raw text into a numerical representation (input IDs or tensors).
# 2.  **Model**: The core transformer model processes the numerical inputs and produces raw outputs, called "logits".
# 3.  **Post-processing**: Converts the raw logits into a human-readable format, like probabilities and labels.
#
# Let's investigate each step.

# ## Step 2: Investigate the tokenizer
#
# The tokenizer takes our raw text and turns it into numbers the model can understand, adding any special tokens the model requires.

# +
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Specify the type of tensors we want to get back (PyTorch tensors 'pt')
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
# -

# * The output `inputs` is a dictionary containing `input_ids` and `attention_mask`. These are the tensors that will be fed directly to the model.

# ## Step 3: Investigate the model
#
# Now we pass the tokenized inputs to the model.
#
# ### The Base Model
# If we use a base model (`AutoModel`), it returns the final hidden states, which are high-dimensional vectors for each input token. This is useful for feature extraction but not for classification directly.

# +
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

# The **inputs syntax unpacks the dictionary into keyword arguments,
# equivalent to running `model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])`
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
# -

# * The output shape `torch.Size([2, 20, 768])` means: (2 sentences, 20 tokens per sentence, 768 hidden dimensions per token).
#
# ### The "Head" Model
# To get classifications, we need a model with a "sequence classification head" on top of the base model. This head is a small neural network that takes the base model's output and converts it into classification scores (logits).

# +
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

# The output logits now have a much smaller dimensionality.
print(outputs.logits.shape)
# -

# * The shape `torch.Size([2, 2])` means: (2 sentences, 2 classification scores per sentence).

# ## Step 4: Post-Processing the Output
#
# The model outputs raw scores called **logits**. These are not probabilities.

# +
print(outputs.logits)
# -

# * The output should be similar to:
# ```text
# tensor([[-1.5607,  1.6123],
#         [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
# ```
#
# To convert these logits into probabilities, we need to apply a **SoftMax** function.

# +
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
# -

# * Now we have recognizable probability scores:
# ```text
# tensor([[4.0195e-02, 9.5980e-01],
#         [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
# ```
#
# But which score corresponds to which label? We can find this in the model's configuration.

# +
model.config.id2label
# -

# * The output `{0: 'NEGATIVE', 1: 'POSITIVE'}` tells us the mapping.
#
# * Now we can conclude that the model predicted the following:
#     * **First sentence:** NEGATIVE: 0.0402, POSITIVE: 0.9598
#     * **Second sentence:** NEGATIVE: 0.9995, POSITIVE: 0.0005
#
# This is exactly what the `pipeline` function does for us automatically!
