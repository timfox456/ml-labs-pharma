# Transformers

* In this lab we will practice the basic use of Transformers

### Lab Goals:

* Use Transformers for:
  * Classification
  * Text Generation
  * Entity name recognition
  * Question answering
  * Summarization
  * Translation

### How to do this lab

* Login to the server
* On the command line, open the Python interpreter
* We are not using the Jupyter notebook for this lab, and here is why:
  * Although the Jupyter notebook is a great tool, it is not the recommended practice for ML in production
  * Python scripts are better for performance
  * Architecting your systems in Python leads to better architecture and manageable code development
  * So, let us start

### Quizzes
* When you encounter a quiz, please provide your answer in private chat.
* For example
  * Quiz 1: What was your score?
  * You: `1: 99%`

### Step 1: Use a Pipeline for classification

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.") 
```

* Your output should be similar to this:

```text
[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```

### Quiz 1: What was your score?

### Step 2: Use a Pipeline for text classification

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

* Your output should be similar to this:

```text
{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}
```

### Quiz 2: What classification did you get?

### Step 3: Use a Pipeline for text generation

```python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

* Your output should look like this
```text
[{'generated_text': 'In this course, we will teach you how to understand and use '
                    'data flow and data interchange when handling user data. We '
                    'will be working with one or more of the most commonly used '
                    'data flows — data flows of various types, as seen by the '
                    'HTTP'}]

```

### Quiz 3: What was the generated text?

### Bonus: Generate more text sequences

* Use the num_return_sequences and max_length arguments to generate two sentences of 30 words each.

### Step 4: Use another model for text generation

```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

### Quiz 4: What was the generated text?

### Step 5: Fill in the missing words

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

### Quiz 5: What were the top two predictions?

### Step 6: Entity name recognition

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

### Quiz 6: 
* What were the recognized entities?
* Put in your name and place. Did it recognize you?

### Step 7: Question answering

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
```

* The answer should look as follows

```text
{'score': 0.6385916471481323, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
```

### Quiz 7: Formulate your question, let HF answer it. What was your question and HG's answer?

### Step 8: Summarization

```python
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
```

### Quiz 8: 
* Try a max_length or a min_length
* What was the format of your function call?


### Step 9: Translation

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
```

### Quiz 9: What was the translation?
* TODO Bonus: What other languages are enabled? Try your favorite one or one you know.
* Hint
```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-fr")
translator("This course was produced by Hugging Face.")
```
* Note: this may require tuning
