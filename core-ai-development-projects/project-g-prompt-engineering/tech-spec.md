# Tech Spec: Prompt Engineering Framework

## Components
1. Template library (zero-shot, few-shot, CoT)
2. DSPy for optimization
3. Evaluation metrics
4. A/B testing

## Patterns
```python
# Zero-shot
"Classify the sentiment: {text}"

# Few-shot
"Examples:\nText: Great! Sentiment: Positive\nText: {text} Sentiment:"

# Chain-of-Thought
"Let's think step by step:\n1. {text}"

# ReAct
"Thought: {thought}\nAction: {action}\nObservation: {obs}"
```

## DSPy Optimization
```python
import dspy

class CoT(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.generate(question=question)
```
