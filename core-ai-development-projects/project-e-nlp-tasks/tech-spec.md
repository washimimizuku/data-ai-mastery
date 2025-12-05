# Tech Spec: NLP Multi-Task System

## Stack
Hugging Face Transformers, spaCy, FastAPI

## Models
- Classification: `distilbert-base-uncased-finetuned-sst-2-english`
- NER: `dslim/bert-base-NER`
- Summarization: `facebook/bart-large-cnn`
- QA: `deepset/roberta-base-squad2`
- Translation: `Helsinki-NLP/opus-mt-en-de`
- Generation: `gpt2`

## Implementation
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
ner = pipeline("ner")
summarizer = pipeline("summarization")
qa = pipeline("question-answering")
translator = pipeline("translation_en_to_de")
generator = pipeline("text-generation")
```

## FastAPI Endpoints
```python
@app.post("/classify")
@app.post("/ner")
@app.post("/summarize")
@app.post("/qa")
@app.post("/translate")
@app.post("/generate")
```
