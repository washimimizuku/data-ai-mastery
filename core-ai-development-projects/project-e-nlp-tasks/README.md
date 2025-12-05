# Project E: NLP Multi-Task System

## Objective

Build a comprehensive NLP system covering text classification, named entity recognition, summarization, question answering, translation, and text generation using Hugging Face Transformers with FastAPI endpoints.

**What You'll Build**: A production-ready NLP API that performs 6 different NLP tasks (sentiment analysis, NER, summarization, QA, translation, generation) with FastAPI endpoints, model comparison, and a Gradio interface for testing.

**What You'll Learn**: Modern NLP techniques, Hugging Face Transformers, BERT variants, sequence-to-sequence models, FastAPI development, and building unified NLP applications.

## Time Estimate

**1-2 days (8-16 hours)** - Following the implementation plan

### Day 1 (8-16 hours)
- **Hours 1-2**: Setup Transformers and spaCy (install, download models)
- **Hours 3-5**: Implement classification, NER, summarization (3 tasks)
- **Hours 6-8**: Implement QA, translation, generation (3 tasks)
- **Hours 9-11**: Build FastAPI endpoints (6 endpoints + docs)
- **Hours 12-13**: Test all tasks (unit tests, integration tests)
- **Hours 14-15**: Create Gradio UI (tabbed interface)
- **Hour 16**: Documentation and polish (README, examples)

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 51-70
  - Days 51-60: ML fundamentals
  - Days 61-70: NLP basics
- [30 Days of Python](https://github.com/washimimizuku/30-days-python-data-ai) - Days 1-20

### Technical Requirements
- Python 3.11+ installed
- 8GB+ RAM (16GB recommended)
- Understanding of transformers and NLP
- Basic FastAPI knowledge

### Tools Needed
- Python with transformers, spacy, fastapi
- Postman or curl for API testing
- Git for version control

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install transformers torch

# Install NLP libraries
pip install spacy
python -m spacy download en_core_web_sm

# Install API framework
pip install fastapi uvicorn

# Install UI
pip install gradio

# Create project structure
mkdir -p nlp-system/{src,tests,data}
cd nlp-system
```

### Step 3: Implement Text Classification
```python
# src/classification.py
from transformers import pipeline

class TextClassifier:
    """Text classification for sentiment and topics"""
    
    def __init__(self):
        # Sentiment analysis
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Topic classification
        self.topic = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        print("✓ Loaded classification models")
    
    def analyze_sentiment(self, text: str):
        """Analyze sentiment (positive/negative)"""
        result = self.sentiment(text)[0]
        return {
            "label": result["label"],
            "score": result["score"],
            "text": text
        }
    
    def classify_topic(self, text: str, candidate_labels: list):
        """Classify text into topics"""
        result = self.topic(text, candidate_labels)
        return {
            "labels": result["labels"],
            "scores": result["scores"],
            "text": text
        }

# Usage
if __name__ == "__main__":
    classifier = TextClassifier()
    
    # Sentiment
    result = classifier.analyze_sentiment("I love this product!")
    print(f"Sentiment: {result['label']} ({result['score']:.2%})")
    
    # Topic
    result = classifier.classify_topic(
        "The stock market crashed today",
        ["business", "sports", "technology", "politics"]
    )
    print(f"Topic: {result['labels'][0]} ({result['scores'][0]:.2%})")
```

### Step 4: Implement Named Entity Recognition
```python
# src/ner.py
from transformers import pipeline
import spacy

class NamedEntityRecognizer:
    """NER using Transformers and spaCy"""
    
    def __init__(self, use_transformers: bool = True):
        self.use_transformers = use_transformers
        
        if use_transformers:
            self.ner = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
            print("✓ Loaded BERT NER")
        else:
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ Loaded spaCy NER")
    
    def extract_entities(self, text: str):
        """Extract named entities"""
        if self.use_transformers:
            entities = self.ner(text)
            return {
                "entities": [
                    {
                        "text": ent["word"],
                        "label": ent["entity_group"],
                        "score": ent["score"]
                    }
                    for ent in entities
                ],
                "count": len(entities)
            }
        else:
            doc = self.nlp(text)
            return {
                "entities": [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    }
                    for ent in doc.ents
                ],
                "count": len(doc.ents)
            }

# Usage
if __name__ == "__main__":
    ner = NamedEntityRecognizer()
    
    text = "Apple Inc. is located in Cupertino, California. Tim Cook is the CEO."
    result = ner.extract_entities(text)
    
    print(f"Found {result['count']} entities:")
    for ent in result['entities']:
        print(f"  {ent['text']}: {ent['label']}")
```

### Step 5: Implement Summarization
```python
# src/summarization.py
from transformers import pipeline

class TextSummarizer:
    """Text summarization using BART"""
    
    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        print("✓ Loaded BART summarizer")
    
    def summarize(self, text: str, max_length: int = 130, min_length: int = 30):
        """Summarize text"""
        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )[0]
        
        return {
            "summary": result["summary_text"],
            "original_length": len(text.split()),
            "summary_length": len(result["summary_text"].split()),
            "compression_ratio": len(result["summary_text"]) / len(text)
        }

# Usage
if __name__ == "__main__":
    summarizer = TextSummarizer()
    
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    """
    
    result = summarizer.summarize(long_text)
    print(f"Summary ({result['summary_length']} words):")
    print(result['summary'])
```

### Step 6: Implement Question Answering
```python
# src/qa.py
from transformers import pipeline

class QuestionAnswerer:
    """Question answering using RoBERTa"""
    
    def __init__(self):
        self.qa = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
        print("✓ Loaded RoBERTa QA")
    
    def answer(self, question: str, context: str):
        """Answer question based on context"""
        result = self.qa(question=question, context=context)
        
        return {
            "answer": result["answer"],
            "score": result["score"],
            "start": result["start"],
            "end": result["end"]
        }

# Usage
if __name__ == "__main__":
    qa = QuestionAnswerer()
    
    context = """
    The Amazon rainforest is a moist broadleaf tropical rainforest in the 
    Amazon biome that covers most of the Amazon basin of South America. 
    This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered 
    by the rainforest.
    """
    
    question = "How large is the Amazon rainforest?"
    result = qa.answer(question, context)
    
    print(f"Q: {question}")
    print(f"A: {result['answer']} (confidence: {result['score']:.2%})")
```

### Step 7: Implement Translation
```python
# src/translation.py
from transformers import pipeline

class Translator:
    """Translation using Helsinki-NLP models"""
    
    def __init__(self):
        # English to German
        self.en_to_de = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-de"
        )
        
        # English to French
        self.en_to_fr = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-fr"
        )
        
        print("✓ Loaded translation models")
    
    def translate(self, text: str, target_lang: str = "de"):
        """Translate text"""
        if target_lang == "de":
            result = self.en_to_de(text)[0]
        elif target_lang == "fr":
            result = self.en_to_fr(text)[0]
        else:
            raise ValueError(f"Unsupported language: {target_lang}")
        
        return {
            "original": text,
            "translation": result["translation_text"],
            "target_lang": target_lang
        }

# Usage
if __name__ == "__main__":
    translator = Translator()
    
    result = translator.translate("Hello, how are you?", "de")
    print(f"EN: {result['original']}")
    print(f"DE: {result['translation']}")
```

### Step 8: Implement Text Generation
```python
# src/generation.py
from transformers import pipeline

class TextGenerator:
    """Text generation using GPT-2"""
    
    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model="gpt2"
        )
        print("✓ Loaded GPT-2")
    
    def generate(self, prompt: str, max_length: int = 100, num_return: int = 1):
        """Generate text from prompt"""
        results = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return,
            temperature=0.7
        )
        
        return {
            "prompt": prompt,
            "generations": [r["generated_text"] for r in results]
        }

# Usage
if __name__ == "__main__":
    generator = TextGenerator()
    
    result = generator.generate("Once upon a time", max_length=50)
    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['generations'][0]}")
```

### Step 9: Build FastAPI Endpoints
```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.classification import TextClassifier
from src.ner import NamedEntityRecognizer
from src.summarization import TextSummarizer
from src.qa import QuestionAnswerer
from src.translation import Translator
from src.generation import TextGenerator

# Initialize FastAPI
app = FastAPI(
    title="NLP Multi-Task API",
    description="Comprehensive NLP system with 6 tasks",
    version="1.0.0"
)

# Load models
print("Loading models...")
classifier = TextClassifier()
ner = NamedEntityRecognizer()
summarizer = TextSummarizer()
qa = QuestionAnswerer()
translator = Translator()
generator = TextGenerator()
print("✓ All models loaded!")

# Request models
class TextRequest(BaseModel):
    text: str

class TopicRequest(BaseModel):
    text: str
    labels: List[str]

class QARequest(BaseModel):
    question: str
    context: str

class TranslationRequest(BaseModel):
    text: str
    target_lang: str = "de"

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    num_return: int = 1

# Endpoints
@app.post("/classify/sentiment")
async def classify_sentiment(request: TextRequest):
    """Analyze sentiment"""
    try:
        return classifier.analyze_sentiment(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/topic")
async def classify_topic(request: TopicRequest):
    """Classify topic"""
    try:
        return classifier.classify_topic(request.text, request.labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ner")
async def extract_entities(request: TextRequest):
    """Extract named entities"""
    try:
        return ner.extract_entities(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_text(request: TextRequest):
    """Summarize text"""
    try:
        return summarizer.summarize(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa")
async def answer_question(request: QARequest):
    """Answer question"""
    try:
        return qa.answer(request.question, request.context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """Translate text"""
    try:
        return translator.translate(request.text, request.target_lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    """Generate text"""
    try:
        return generator.generate(
            request.prompt,
            request.max_length,
            request.num_return
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "NLP Multi-Task API",
        "endpoints": [
            "/classify/sentiment",
            "/classify/topic",
            "/ner",
            "/summarize",
            "/qa",
            "/translate",
            "/generate"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 10: Create Gradio Interface
```python
# gradio_app.py
import gradio as gr
from src.classification import TextClassifier
from src.ner import NamedEntityRecognizer
from src.summarization import TextSummarizer
from src.qa import QuestionAnswerer
from src.translation import Translator
from src.generation import TextGenerator

# Load models
print("Loading models...")
classifier = TextClassifier()
ner = NamedEntityRecognizer()
summarizer = TextSummarizer()
qa = QuestionAnswerer()
translator = Translator()
generator = TextGenerator()
print("✓ All models loaded!")

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="NLP Multi-Task System") as demo:
    gr.Markdown("# 📝 NLP Multi-Task System")
    gr.Markdown("Perform 6 different NLP tasks using state-of-the-art models")
    
    with gr.Tabs():
        # Classification
        with gr.Tab("Classification"):
            with gr.Row():
                cls_input = gr.Textbox(label="Text", lines=3)
                cls_btn = gr.Button("Analyze Sentiment", variant="primary")
            cls_output = gr.JSON(label="Result")
            cls_btn.click(
                fn=lambda x: classifier.analyze_sentiment(x),
                inputs=cls_input,
                outputs=cls_output
            )
        
        # NER
        with gr.Tab("Named Entity Recognition"):
            with gr.Row():
                ner_input = gr.Textbox(label="Text", lines=3)
                ner_btn = gr.Button("Extract Entities", variant="primary")
            ner_output = gr.JSON(label="Entities")
            ner_btn.click(
                fn=lambda x: ner.extract_entities(x),
                inputs=ner_input,
                outputs=ner_output
            )
        
        # Summarization
        with gr.Tab("Summarization"):
            with gr.Row():
                sum_input = gr.Textbox(label="Text", lines=5)
                sum_btn = gr.Button("Summarize", variant="primary")
            sum_output = gr.JSON(label="Summary")
            sum_btn.click(
                fn=lambda x: summarizer.summarize(x),
                inputs=sum_input,
                outputs=sum_output
            )
        
        # QA
        with gr.Tab("Question Answering"):
            qa_context = gr.Textbox(label="Context", lines=5)
            qa_question = gr.Textbox(label="Question")
            qa_btn = gr.Button("Answer", variant="primary")
            qa_output = gr.JSON(label="Answer")
            qa_btn.click(
                fn=lambda q, c: qa.answer(q, c),
                inputs=[qa_question, qa_context],
                outputs=qa_output
            )
        
        # Translation
        with gr.Tab("Translation"):
            trans_input = gr.Textbox(label="English Text", lines=3)
            trans_lang = gr.Dropdown(["de", "fr"], label="Target Language", value="de")
            trans_btn = gr.Button("Translate", variant="primary")
            trans_output = gr.JSON(label="Translation")
            trans_btn.click(
                fn=lambda t, l: translator.translate(t, l),
                inputs=[trans_input, trans_lang],
                outputs=trans_output
            )
        
        # Generation
        with gr.Tab("Text Generation"):
            gen_input = gr.Textbox(label="Prompt", lines=2)
            gen_length = gr.Slider(50, 200, value=100, label="Max Length")
            gen_btn = gr.Button("Generate", variant="primary")
            gen_output = gr.JSON(label="Generated Text")
            gen_btn.click(
                fn=lambda p, l: generator.generate(p, l),
                inputs=[gen_input, gen_length],
                outputs=gen_output
            )

if __name__ == "__main__":
    demo.launch(share=False)
```

## Key Features to Implement

### 1. Text Classification
- Sentiment analysis (positive/negative)
- Topic classification (zero-shot)
- Multi-label classification

### 2. Named Entity Recognition
- Person, organization, location extraction
- BERT-based NER
- spaCy integration

### 3. Text Summarization
- Extractive and abstractive summarization
- BART model
- Configurable length

### 4. Question Answering
- Context-based QA
- RoBERTa SQuAD model
- Confidence scores

### 5. Translation
- English to German/French
- Helsinki-NLP models
- Multiple language pairs

### 6. Text Generation
- GPT-2 generation
- Prompt-based
- Temperature control

### 7. FastAPI Endpoints
- RESTful API
- 6+ endpoints
- OpenAPI documentation
- Error handling

### 8. Model Comparison
- Performance benchmarks
- Accuracy metrics
- Speed comparison

## Success Criteria

By the end of this project, you should have:

### Functionality
- [ ] Text classification working (sentiment + topic)
- [ ] NER working (BERT + spaCy)
- [ ] Summarization working (BART)
- [ ] QA working (RoBERTa)
- [ ] Translation working (Helsinki-NLP)
- [ ] Generation working (GPT-2)
- [ ] FastAPI with 6+ endpoints
- [ ] Gradio interface

### Quality Metrics
- [ ] **All NLP tasks working**: 6/6 tasks functional
- [ ] **FastAPI API deployed**: All endpoints responding
- [ ] **Model comparison documented**: Performance metrics
- [ ] **Code quality**: < 600 lines of code

### Deliverables
- [ ] 6 NLP task implementations
- [ ] FastAPI application
- [ ] Gradio interface
- [ ] Model comparison results
- [ ] API documentation
- [ ] Comprehensive README

## Learning Outcomes

After completing this project, you'll be able to:

- Use Hugging Face Transformers for NLP
- Implement multiple NLP tasks
- Build RESTful APIs with FastAPI
- Deploy NLP models
- Compare model performance
- Create interactive NLP demos

## Expected Performance

**Inference Speed (CPU)**:
```
Classification: 50-100ms
NER: 100-200ms
Summarization: 1-3 seconds
QA: 200-500ms
Translation: 500ms-1s
Generation: 2-5 seconds
```

**Model Sizes**:
```
DistilBERT: 250 MB
BERT-NER: 420 MB
BART: 1.6 GB
RoBERTa: 500 MB
Helsinki-NLP: 300 MB
GPT-2: 500 MB
```

## Project Structure

```
project-e-nlp-tasks/
├── src/
│   ├── __init__.py
│   ├── classification.py    # Sentiment + topic
│   ├── ner.py               # Named entity recognition
│   ├── summarization.py     # Text summarization
│   ├── qa.py                # Question answering
│   ├── translation.py       # Translation
│   └── generation.py        # Text generation
├── tests/
│   ├── test_classification.py
│   ├── test_ner.py
│   └── test_api.py
├── data/
│   └── samples/             # Test texts
├── results/
│   └── benchmarks.json      # Performance metrics
├── app.py                   # FastAPI application
├── gradio_app.py            # Gradio interface
├── requirements.txt
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges & Solutions

### Challenge 1: Model Download Size
**Problem**: Models total 3-4 GB
**Solution**: Download on-demand, cache models

### Challenge 2: Slow Inference
**Problem**: Some tasks take seconds
**Solution**: Use GPU, optimize batch size, use distilled models

### Challenge 3: API Timeout
**Problem**: Long-running requests timeout
**Solution**: Increase timeout, use async processing

## Next Steps

After completing this project:

1. **Add to Portfolio**: Document on GitHub with API examples
2. **Write Blog Post**: "Building a Multi-Task NLP API"
3. **Extend Features**: Add more languages, fine-tune models
4. **Build Project F**: Continue with Inference Optimization
5. **Production Use**: Deploy with Docker and load balancing

## Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [spaCy Documentation](https://spacy.io/)
- [Gradio Documentation](https://gradio.app/)

## Questions?

If you get stuck:
1. Review the tech-spec.md for model details
2. Check implementation-plan.md for step-by-step guide
3. Search Hugging Face forums for model issues
4. Review the 100 Days bootcamp materials on NLP
