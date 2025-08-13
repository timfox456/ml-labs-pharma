# Pipeline explained

* In this lab we will go behind the Pipeline and see how it works

#### Lab Goals:

* Go deeper into the Pipeline
* Investigate what is going on behind the scenes


### Step 1: Repeat sentiment analysis

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
```

* Obtain this

```text
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```

### Step 2: Investigate the tokenizer

```python
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

* Specify the type of tensors we want to get back

```python
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```



* Here is your tensor


![](../images/03-tensor.png)

### Step 3: Investigate the model

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

* Let's dump the output of the model

```python
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
``` 

* Here is your output

![](../images/04-output.png)

### Step 3: Investigate the model

```python
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)
```

* Now the dimensionality is much lower
```text
torch.Size([2, 2])
```

### Step 4: Here are the "logits"

```python
print(outputs.logits)
```

```text
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```


* But what we need are the probabilities

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

```text
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```

* Now we can see that the model predicted [0.0402, 0.9598] for the first sentence and [0.9995, 0.0005] for the second one. These are recognizable probability scores.

* To get the labels corresponding to each position, we can inspect the id2label attribute of the model config (more on this later):

```python
model.config.id2label
```

```text
{0: 'NEGATIVE', 1: 'POSITIVE'}
```

* Now we can conclude that the model predicted the following:
    * First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598
    * Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005



