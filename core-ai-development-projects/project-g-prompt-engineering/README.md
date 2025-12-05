# Project G: Prompt Engineering Framework

## Objective

Build a comprehensive prompt engineering framework demonstrating systematic prompt design, optimization techniques (zero-shot, few-shot, chain-of-thought, ReAct), automated optimization with DSPy, and rigorous evaluation to achieve 20-50% performance improvements on various tasks.

**What You'll Build**: A production-ready prompt engineering toolkit with template libraries, few-shot example selectors, chain-of-thought reasoning, ReAct agents, DSPy optimization, A/B testing framework, and comprehensive evaluation metrics.

**What You'll Learn**: Prompt engineering best practices, zero-shot vs few-shot learning, chain-of-thought reasoning, ReAct pattern, DSPy for automated optimization, prompt evaluation metrics, A/B testing, and systematic prompt improvement methodologies.

## Time Estimate

**2 days (16 hours)** - Following the implementation plan

### Day 1 (8 hours)
- **Hour 1**: Setup & basics (install dependencies, Ollama setup, basic templates)
- **Hours 2-3**: Template library (zero-shot, few-shot, CoT, ReAct patterns)
- **Hours 4-5**: Few-shot learning (example selection, dynamic examples, similarity-based)
- **Hours 6-7**: Chain-of-thought (step-by-step reasoning, self-consistency)
- **Hour 8**: ReAct pattern (thought-action-observation loops)

### Day 2 (8 hours)
- **Hours 1-2**: DSPy optimization (automatic prompt tuning, teleprompters)
- **Hours 3-4**: Evaluation framework (accuracy, consistency, cost metrics)
- **Hours 5-6**: A/B testing (compare prompts, statistical significance)
- **Hour 7**: Gradio UI (interactive prompt testing, comparison tool)
- **Hour 8**: Documentation & examples (best practices guide, use cases)

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 71-85
  - Days 71-75: LLM fundamentals and prompting basics
  - Days 76-80: Advanced prompting techniques
  - Days 81-85: Prompt optimization and evaluation
- [30 Days of Python](https://github.com/washimimizuku/30-days-python-data-ai) - Days 1-20

### Technical Requirements
- Python 3.11+ installed
- 8GB+ RAM for local LLMs
- Understanding of LLM capabilities and limitations
- Basic knowledge of prompt engineering concepts

### Tools Needed
- Python with langchain, dspy, ollama
- Ollama for local LLM serving
- Gradio for UI
- Git for version control

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Install Ollama and Models
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com for Windows

# Pull LLM models
ollama pull llama3:8b      # 4.7GB - Best for reasoning
ollama pull mistral:7b     # 4.1GB - Fast and good
ollama pull phi3:mini      # 2.3GB - Efficient

# Verify installation
ollama list
```

### Step 3: Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain langchain-community ollama \
    dspy-ai gradio pandas matplotlib \
    scikit-learn sentence-transformers

# Create project structure
mkdir -p prompt-framework/{src,templates,examples,evaluations}
cd prompt-framework
```

### Step 4: Build Basic Template System
```python
# src/templates/base_templates.py
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class PromptExample:
    """Single example for few-shot learning"""
    input: str
    output: str
    explanation: str = ""

class PromptTemplateLibrary:
    """Library of reusable prompt templates"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize common prompt templates"""
        
        # Zero-shot classification
        self.templates['zero_shot_classify'] = PromptTemplate(
            input_variables=["text", "categories"],
            template="""Classify the following text into one of these categories: {categories}

Text: {text}

Category:"""
        )
        
        # Zero-shot extraction
        self.templates['zero_shot_extract'] = PromptTemplate(
            input_variables=["text", "entities"],
            template="""Extract the following entities from the text: {entities}

Text: {text}

Extracted entities (JSON format):"""
        )
        
        # Zero-shot summarization
        self.templates['zero_shot_summarize'] = PromptTemplate(
            input_variables=["text", "max_words"],
            template="""Summarize the following text in {max_words} words or less:

{text}

Summary:"""
        )
        
        # Zero-shot question answering
        self.templates['zero_shot_qa'] = PromptTemplate(
            input_variables=["context", "question"],
            template="""Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""
        )
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get template by name"""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]
    
    def add_template(self, name: str, template: PromptTemplate):
        """Add custom template"""
        self.templates[name] = template
    
    def list_templates(self) -> List[str]:
        """List all available templates"""
        return list(self.templates.keys())

# Usage
if __name__ == "__main__":
    library = PromptTemplateLibrary()
    
    # Use zero-shot classification
    template = library.get_template('zero_shot_classify')
    prompt = template.format(
        text="This movie was absolutely fantastic!",
        categories="positive, negative, neutral"
    )
    print(prompt)
```

### Step 5: Implement Few-Shot Learning
```python
# src/templates/few_shot.py
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from typing import List, Dict

class FewShotPromptBuilder:
    """Build few-shot prompts with dynamic example selection"""
    
    def __init__(self, use_semantic_selection: bool = False):
        self.use_semantic_selection = use_semantic_selection
        if use_semantic_selection:
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    def create_few_shot_prompt(
        self,
        examples: List[Dict[str, str]],
        example_template: str,
        prefix: str,
        suffix: str,
        input_variables: List[str],
        k: int = 3
    ) -> FewShotPromptTemplate:
        """Create few-shot prompt with optional semantic selection"""
        
        # Create example prompt template
        example_prompt = PromptTemplate(
            input_variables=list(examples[0].keys()),
            template=example_template
        )
        
        if self.use_semantic_selection:
            # Use semantic similarity for example selection
            example_selector = SemanticSimilarityExampleSelector.from_examples(
                examples,
                self.embeddings,
                Chroma,
                k=k
            )
            
            prompt = FewShotPromptTemplate(
                example_selector=example_selector,
                example_prompt=example_prompt,
                prefix=prefix,
                suffix=suffix,
                input_variables=input_variables
            )
        else:
            # Use all examples
            prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example_prompt,
                prefix=prefix,
                suffix=suffix,
                input_variables=input_variables
            )
        
        return prompt

# Example: Sentiment Classification with Few-Shot
class SentimentFewShotPrompt:
    """Few-shot prompt for sentiment classification"""
    
    def __init__(self):
        self.examples = [
            {
                "text": "This movie was absolutely fantastic! Best film I've seen all year.",
                "sentiment": "positive",
                "confidence": "high"
            },
            {
                "text": "Terrible experience. Would not recommend to anyone.",
                "sentiment": "negative",
                "confidence": "high"
            },
            {
                "text": "It was okay, nothing special but not bad either.",
                "sentiment": "neutral",
                "confidence": "medium"
            },
            {
                "text": "I loved the cinematography but the plot was weak.",
                "sentiment": "mixed",
                "confidence": "medium"
            },
            {
                "text": "Absolutely brilliant! A masterpiece of modern cinema.",
                "sentiment": "positive",
                "confidence": "high"
            }
        ]
        
        self.example_template = """Text: {text}
Sentiment: {sentiment}
Confidence: {confidence}"""
        
        self.prefix = """Classify the sentiment of the following texts. Provide the sentiment (positive, negative, neutral, or mixed) and confidence level (high, medium, low).

Examples:"""
        
        self.suffix = """
Text: {text}
Sentiment:"""
    
    def build_prompt(self, use_semantic: bool = False, k: int = 3):
        """Build the few-shot prompt"""
        builder = FewShotPromptBuilder(use_semantic_selection=use_semantic)
        
        return builder.create_few_shot_prompt(
            examples=self.examples,
            example_template=self.example_template,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=["text"],
            k=k
        )

# Usage
if __name__ == "__main__":
    sentiment_prompt = SentimentFewShotPrompt()
    
    # Static few-shot (all examples)
    prompt = sentiment_prompt.build_prompt(use_semantic=False)
    result = prompt.format(text="The service was decent but the food was cold.")
    print(result)
    
    # Dynamic few-shot (semantic similarity)
    prompt_semantic = sentiment_prompt.build_prompt(use_semantic=True, k=2)
    result = prompt_semantic.format(text="Amazing experience, highly recommend!")
    print(result)
```

### Step 6: Implement Chain-of-Thought
```python
# src/templates/chain_of_thought.py
from langchain.prompts import PromptTemplate
from typing import List, Dict
import re

class ChainOfThoughtPrompt:
    """Chain-of-thought prompting for complex reasoning"""
    
    def __init__(self):
        self.cot_template = PromptTemplate(
            input_variables=["question"],
            template="""Let's solve this step by step:

Question: {question}

Let's think through this carefully:
1. First, I need to understand what is being asked
2. Then, I'll identify the key information
3. Next, I'll work through the logic
4. Finally, I'll provide the answer

Step-by-step solution:"""
        )
    
    def create_cot_prompt(self, question: str) -> str:
        """Create chain-of-thought prompt"""
        return self.cot_template.format(question=question)
    
    def create_few_shot_cot(self, examples: List[Dict], question: str) -> str:
        """Create few-shot chain-of-thought prompt"""
        prompt = "Solve these problems step by step:\n\n"
        
        # Add examples with reasoning
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Reasoning: {ex['reasoning']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"
        
        # Add the actual question
        prompt += f"Now solve this:\n"
        prompt += f"Question: {question}\n"
        prompt += f"Reasoning:"
        
        return prompt

class SelfConsistencyCoT:
    """Self-consistency with chain-of-thought"""
    
    def __init__(self, llm, num_samples: int = 5):
        self.llm = llm
        self.num_samples = num_samples
        self.cot = ChainOfThoughtPrompt()
    
    def generate_multiple_reasoning_paths(self, question: str) -> List[str]:
        """Generate multiple reasoning paths"""
        prompt = self.cot.create_cot_prompt(question)
        
        responses = []
        for _ in range(self.num_samples):
            response = self.llm.invoke(prompt)
            responses.append(response)
        
        return responses
    
    def extract_final_answer(self, response: str) -> str:
        """Extract final answer from response"""
        # Look for patterns like "Answer:", "Therefore:", "Final answer:"
        patterns = [
            r"Answer:\s*(.+?)(?:\n|$)",
            r"Therefore,?\s*(.+?)(?:\n|$)",
            r"Final answer:\s*(.+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, return last line
        return response.strip().split('\n')[-1]
    
    def get_consensus_answer(self, question: str) -> Dict:
        """Get consensus answer using self-consistency"""
        responses = self.generate_multiple_reasoning_paths(question)
        answers = [self.extract_final_answer(r) for r in responses]
        
        # Count answer frequencies
        from collections import Counter
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0]
        
        return {
            'question': question,
            'all_answers': answers,
            'consensus_answer': most_common[0],
            'confidence': most_common[1] / len(answers),
            'reasoning_paths': responses
        }

# Math reasoning examples
MATH_COT_EXAMPLES = [
    {
        "question": "If a train travels 120 miles in 2 hours, what is its average speed?",
        "reasoning": "To find average speed, I divide distance by time. Distance = 120 miles, Time = 2 hours. Speed = 120 / 2 = 60 miles per hour.",
        "answer": "60 miles per hour"
    },
    {
        "question": "A store has 45 apples. If they sell 3/5 of them, how many apples are left?",
        "reasoning": "First, find 3/5 of 45: (3/5) × 45 = 27 apples sold. Then subtract from total: 45 - 27 = 18 apples remaining.",
        "answer": "18 apples"
    }
]

# Usage
if __name__ == "__main__":
    from langchain.llms import Ollama
    
    llm = Ollama(model="llama3:8b")
    
    # Basic CoT
    cot = ChainOfThoughtPrompt()
    prompt = cot.create_cot_prompt(
        "If 5 machines can produce 5 widgets in 5 minutes, how long would it take 100 machines to produce 100 widgets?"
    )
    print(prompt)
    
    # Self-consistency CoT
    sc_cot = SelfConsistencyCoT(llm, num_samples=5)
    result = sc_cot.get_consensus_answer(
        "If 5 machines can produce 5 widgets in 5 minutes, how long would it take 100 machines to produce 100 widgets?"
    )
    print(f"\nConsensus Answer: {result['consensus_answer']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Step 7: Implement ReAct Pattern
```python
# src/templates/react_pattern.py
from langchain.prompts import PromptTemplate
from typing import List, Dict, Callable
import re

class ReActAgent:
    """ReAct (Reasoning + Acting) agent pattern"""
    
    def __init__(self, llm, tools: Dict[str, Callable]):
        self.llm = llm
        self.tools = tools
        self.max_iterations = 10
        
        self.react_template = PromptTemplate(
            input_variables=["question", "tools", "history"],
            template="""Answer the following question using the available tools.

Available tools:
{tools}

Question: {question}

{history}

Thought: Let me think about what to do next.
Action:"""
        )
    
    def run(self, question: str) -> Dict:
        """Run ReAct loop"""
        history = []
        
        for i in range(self.max_iterations):
            # Generate thought and action
            prompt = self._build_prompt(question, history)
            response = self.llm.invoke(prompt)
            
            # Parse response
            thought, action, action_input = self._parse_response(response)
            
            if action.lower() == "final answer":
                return {
                    'question': question,
                    'answer': action_input,
                    'steps': history,
                    'iterations': i + 1
                }
            
            # Execute action
            observation = self._execute_action(action, action_input)
            
            # Add to history
            history.append({
                'thought': thought,
                'action': action,
                'action_input': action_input,
                'observation': observation
            })
        
        return {
            'question': question,
            'answer': "Max iterations reached",
            'steps': history,
            'iterations': self.max_iterations
        }
    
    def _build_prompt(self, question: str, history: List[Dict]) -> str:
        """Build ReAct prompt with history"""
        tools_desc = "\n".join([f"- {name}: {func.__doc__}" 
                               for name, func in self.tools.items()])
        
        history_text = ""
        for step in history:
            history_text += f"\nThought: {step['thought']}"
            history_text += f"\nAction: {step['action']}[{step['action_input']}]"
            history_text += f"\nObservation: {step['observation']}\n"
        
        return self.react_template.format(
            question=question,
            tools=tools_desc,
            history=history_text
        )
    
    def _parse_response(self, response: str) -> tuple:
        """Parse LLM response into thought, action, action_input"""
        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", response)
        thought = thought_match.group(1) if thought_match else ""
        
        # Extract action
        action_match = re.search(r"Action:\s*(\w+)\[(.+?)\]", response)
        if action_match:
            action = action_match.group(1)
            action_input = action_match.group(2)
        else:
            action = "Final Answer"
            action_input = response.split("Answer:")[-1].strip() if "Answer:" in response else response
        
        return thought, action, action_input
    
    def _execute_action(self, action: str, action_input: str) -> str:
        """Execute tool action"""
        if action in self.tools:
            try:
                result = self.tools[action](action_input)
                return str(result)
            except Exception as e:
                return f"Error executing {action}: {str(e)}"
        else:
            return f"Unknown action: {action}"

# Example tools
def calculator(expression: str) -> float:
    """Calculate mathematical expressions"""
    try:
        return eval(expression)
    except:
        return "Invalid expression"

def search(query: str) -> str:
    """Search for information (mock)"""
    # In real implementation, this would call a search API
    mock_results = {
        "population of france": "67 million (2023)",
        "capital of japan": "Tokyo",
        "speed of light": "299,792,458 meters per second"
    }
    return mock_results.get(query.lower(), "No results found")

def wikipedia(topic: str) -> str:
    """Get Wikipedia summary (mock)"""
    # In real implementation, this would call Wikipedia API
    return f"Wikipedia summary for '{topic}': [Mock content]"

# Usage
if __name__ == "__main__":
    from langchain.llms import Ollama
    
    llm = Ollama(model="llama3:8b")
    
    tools = {
        "Calculator": calculator,
        "Search": search,
        "Wikipedia": wikipedia
    }
    
    agent = ReActAgent(llm, tools)
    
    result = agent.run(
        "What is the population of France divided by 10?"
    )
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"\nSteps taken: {result['iterations']}")
    for i, step in enumerate(result['steps'], 1):
        print(f"\nStep {i}:")
        print(f"  Thought: {step['thought']}")
        print(f"  Action: {step['action']}[{step['action_input']}]")
        print(f"  Observation: {step['observation']}")
```

### Step 8: Implement DSPy Optimization

```python
# src/optimization/dspy_optimizer.py
import dspy
from dspy.teleprompt import BootstrapFewShot
from typing import List, Dict

class DSPyPromptOptimizer:
    """Automatic prompt optimization using DSPy"""
    
    def __init__(self, model_name: str = "ollama/llama3:8b"):
        # Configure DSPy with Ollama
        self.lm = dspy.OllamaLocal(model=model_name)
        dspy.settings.configure(lm=self.lm)
    
    def optimize_classification(
        self,
        train_examples: List[Dict],
        metric_fn: callable
    ):
        """Optimize classification prompt"""
        
        # Define DSPy signature
        class Classify(dspy.Signature):
            """Classify text into categories"""
            text = dspy.InputField()
            category = dspy.OutputField()
        
        # Create module
        class ClassificationModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.classify = dspy.ChainOfThought(Classify)
            
            def forward(self, text):
                return self.classify(text=text)
        
        # Convert examples to DSPy format
        dspy_examples = [
            dspy.Example(text=ex['text'], category=ex['category']).with_inputs('text')
            for ex in train_examples
        ]
        
        # Optimize with BootstrapFewShot
        teleprompter = BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=4)
        optimized_module = teleprompter.compile(
            ClassificationModule(),
            trainset=dspy_examples
        )
        
        return optimized_module
    
    def optimize_qa(
        self,
        train_examples: List[Dict],
        metric_fn: callable
    ):
        """Optimize question answering prompt"""
        
        class QA(dspy.Signature):
            """Answer questions based on context"""
            context = dspy.InputField()
            question = dspy.InputField()
            answer = dspy.OutputField()
        
        class QAModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.qa = dspy.ChainOfThought(QA)
            
            def forward(self, context, question):
                return self.qa(context=context, question=question)
        
        dspy_examples = [
            dspy.Example(
                context=ex['context'],
                question=ex['question'],
                answer=ex['answer']
            ).with_inputs('context', 'question')
            for ex in train_examples
        ]
        
        teleprompter = BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=3)
        optimized_module = teleprompter.compile(
            QAModule(),
            trainset=dspy_examples
        )
        
        return optimized_module

# Example: Sentiment classification with DSPy
def sentiment_metric(example, pred, trace=None):
    """Metric for sentiment classification"""
    return example.category.lower() == pred.category.lower()

# Training data
SENTIMENT_TRAIN = [
    {"text": "This is amazing!", "category": "positive"},
    {"text": "Terrible experience", "category": "negative"},
    {"text": "It's okay", "category": "neutral"},
    {"text": "Absolutely love it!", "category": "positive"},
    {"text": "Worst product ever", "category": "negative"},
]

# Usage
if __name__ == "__main__":
    optimizer = DSPyPromptOptimizer()
    
    # Optimize sentiment classification
    optimized_classifier = optimizer.optimize_classification(
        train_examples=SENTIMENT_TRAIN,
        metric_fn=sentiment_metric
    )
    
    # Test optimized prompt
    result = optimized_classifier(text="This product exceeded my expectations!")
    print(f"Category: {result.category}")
```

## Key Features to Implement

### 1. Template Library
- **Zero-shot templates**: Classification, extraction, summarization, QA
- **Few-shot templates**: With static or dynamic example selection
- **Chain-of-thought templates**: Step-by-step reasoning
- **ReAct templates**: Thought-action-observation loops
- **Custom templates**: User-defined patterns

### 2. Few-Shot Learning
- **Static examples**: Fixed set of examples
- **Dynamic selection**: Semantic similarity-based selection
- **Example quality**: Diverse, representative examples
- **K-shot tuning**: Optimal number of examples

### 3. Chain-of-Thought
- **Basic CoT**: "Let's think step by step"
- **Few-shot CoT**: Examples with reasoning
- **Self-consistency**: Multiple reasoning paths
- **Least-to-most**: Break down complex problems

### 4. ReAct Pattern
- **Tool integration**: Calculator, search, Wikipedia
- **Reasoning loop**: Thought → Action → Observation
- **Error handling**: Graceful failure recovery
- **Max iterations**: Prevent infinite loops

### 5. DSPy Optimization
- **Automatic tuning**: Optimize prompts on training data
- **Teleprompters**: BootstrapFewShot, MIPRO
- **Metric-driven**: Optimize for specific metrics
- **Few-shot generation**: Automatic example selection

### 6. Evaluation Framework
- **Accuracy**: Correct vs total predictions
- **Consistency**: Same input → same output
- **Cost**: Token usage and API costs
- **Latency**: Response time
- **Quality**: Human evaluation scores

### 7. A/B Testing
- **Compare prompts**: Statistical significance testing
- **Metrics tracking**: Performance over time
- **Winner selection**: Automated or manual
- **Deployment**: Gradual rollout

## Comprehensive Evaluation Framework

```python
# src/evaluation/prompt_evaluator.py
from typing import List, Dict, Callable
import pandas as pd
from collections import Counter
import time

class PromptEvaluator:
    """Evaluate and compare prompts"""
    
    def __init__(self, llm):
        self.llm = llm
        self.results = []
    
    def evaluate_prompt(
        self,
        prompt_template: str,
        test_cases: List[Dict],
        metric_fn: Callable,
        prompt_name: str = "unnamed"
    ) -> Dict:
        """Evaluate single prompt on test cases"""
        
        correct = 0
        total = len(test_cases)
        latencies = []
        token_counts = []
        responses = []
        
        for test_case in test_cases:
            # Format prompt
            prompt = prompt_template.format(**test_case['input'])
            
            # Measure latency
            start = time.time()
            response = self.llm.invoke(prompt)
            latency = time.time() - start
            
            # Evaluate
            is_correct = metric_fn(test_case['expected'], response)
            
            correct += int(is_correct)
            latencies.append(latency)
            token_counts.append(len(response.split()))
            responses.append({
                'input': test_case['input'],
                'expected': test_case['expected'],
                'actual': response,
                'correct': is_correct
            })
        
        accuracy = correct / total
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(token_counts) / len(token_counts)
        
        result = {
            'prompt_name': prompt_name,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_latency': avg_latency,
            'avg_tokens': avg_tokens,
            'responses': responses
        }
        
        self.results.append(result)
        return result
    
    def compare_prompts(
        self,
        prompts: Dict[str, str],
        test_cases: List[Dict],
        metric_fn: Callable
    ) -> pd.DataFrame:
        """Compare multiple prompts"""
        
        comparison_results = []
        
        for name, template in prompts.items():
            print(f"\nEvaluating: {name}")
            result = self.evaluate_prompt(template, test_cases, metric_fn, name)
            comparison_results.append({
                'Prompt': name,
                'Accuracy': f"{result['accuracy']:.2%}",
                'Correct': f"{result['correct']}/{result['total']}",
                'Avg Latency': f"{result['avg_latency']:.2f}s",
                'Avg Tokens': f"{result['avg_tokens']:.0f}"
            })
        
        df = pd.DataFrame(comparison_results)
        return df
    
    def test_consistency(
        self,
        prompt_template: str,
        test_input: Dict,
        num_runs: int = 10
    ) -> Dict:
        """Test prompt consistency"""
        
        prompt = prompt_template.format(**test_input)
        responses = []
        
        for _ in range(num_runs):
            response = self.llm.invoke(prompt)
            responses.append(response.strip())
        
        # Calculate consistency
        response_counts = Counter(responses)
        most_common = response_counts.most_common(1)[0]
        consistency = most_common[1] / num_runs
        
        return {
            'input': test_input,
            'unique_responses': len(response_counts),
            'most_common_response': most_common[0],
            'consistency_score': consistency,
            'all_responses': responses
        }
    
    def calculate_cost(
        self,
        prompt_template: str,
        test_cases: List[Dict],
        cost_per_1k_tokens: float = 0.002
    ) -> Dict:
        """Calculate prompt cost"""
        
        total_input_tokens = 0
        total_output_tokens = 0
        
        for test_case in test_cases:
            prompt = prompt_template.format(**test_case['input'])
            response = self.llm.invoke(prompt)
            
            input_tokens = len(prompt.split())
            output_tokens = len(response.split())
            
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
        
        total_tokens = total_input_tokens + total_output_tokens
        total_cost = (total_tokens / 1000) * cost_per_1k_tokens
        cost_per_request = total_cost / len(test_cases)
        
        return {
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'cost_per_request': cost_per_request,
            'num_requests': len(test_cases)
        }
    
    def generate_report(self) -> str:
        """Generate evaluation report"""
        if not self.results:
            return "No evaluation results available"
        
        report = "="*60 + "\n"
        report += "PROMPT EVALUATION REPORT\n"
        report += "="*60 + "\n\n"
        
        for result in self.results:
            report += f"Prompt: {result['prompt_name']}\n"
            report += f"  Accuracy: {result['accuracy']:.2%}\n"
            report += f"  Correct: {result['correct']}/{result['total']}\n"
            report += f"  Avg Latency: {result['avg_latency']:.2f}s\n"
            report += f"  Avg Tokens: {result['avg_tokens']:.0f}\n\n"
        
        # Best prompt
        best = max(self.results, key=lambda x: x['accuracy'])
        report += f"🏆 Best Prompt: {best['prompt_name']} ({best['accuracy']:.2%})\n"
        
        return report

# Example test cases
SENTIMENT_TEST_CASES = [
    {
        'input': {'text': 'This is the best product ever!'},
        'expected': 'positive'
    },
    {
        'input': {'text': 'Absolutely terrible, waste of money.'},
        'expected': 'negative'
    },
    {
        'input': {'text': 'It works as expected, nothing special.'},
        'expected': 'neutral'
    },
]

def sentiment_metric(expected: str, actual: str) -> bool:
    """Check if sentiment matches"""
    return expected.lower() in actual.lower()

# Usage
if __name__ == "__main__":
    from langchain.llms import Ollama
    
    llm = Ollama(model="llama3:8b")
    evaluator = PromptEvaluator(llm)
    
    # Define prompts to compare
    prompts = {
        'zero_shot': "Classify the sentiment of this text: {text}\nSentiment:",
        'few_shot': """Classify sentiment:
Examples:
Text: "Great product!" → positive
Text: "Terrible quality" → negative
Text: "It's okay" → neutral

Text: {text}
Sentiment:""",
        'cot': """Classify the sentiment step by step:
Text: {text}

Let's analyze:
1. Identify key words
2. Determine tone
3. Classify sentiment

Sentiment:"""
    }
    
    # Compare prompts
    df = evaluator.compare_prompts(prompts, SENTIMENT_TEST_CASES, sentiment_metric)
    print(df)
    
    # Test consistency
    consistency = evaluator.test_consistency(
        prompts['zero_shot'],
        {'text': 'This is amazing!'},
        num_runs=10
    )
    print(f"\nConsistency: {consistency['consistency_score']:.2%}")
```

## A/B Testing Framework

```python
# src/evaluation/ab_testing.py
from scipy import stats
import numpy as np
from typing import List, Dict

class ABTester:
    """A/B testing for prompt comparison"""
    
    def __init__(self):
        self.experiments = []
    
    def run_ab_test(
        self,
        prompt_a: str,
        prompt_b: str,
        test_cases: List[Dict],
        metric_fn: callable,
        llm,
        name: str = "AB Test"
    ) -> Dict:
        """Run A/B test between two prompts"""
        
        results_a = []
        results_b = []
        
        for test_case in test_cases:
            # Test prompt A
            prompt = prompt_a.format(**test_case['input'])
            response_a = llm.invoke(prompt)
            score_a = metric_fn(test_case['expected'], response_a)
            results_a.append(score_a)
            
            # Test prompt B
            prompt = prompt_b.format(**test_case['input'])
            response_b = llm.invoke(prompt)
            score_b = metric_fn(test_case['expected'], response_b)
            results_b.append(score_b)
        
        # Calculate statistics
        mean_a = np.mean(results_a)
        mean_b = np.mean(results_b)
        
        # T-test for statistical significance
        t_stat, p_value = stats.ttest_ind(results_a, results_b)
        
        # Determine winner
        if p_value < 0.05:  # 95% confidence
            if mean_a > mean_b:
                winner = "Prompt A"
                improvement = (mean_a - mean_b) / mean_b * 100
            else:
                winner = "Prompt B"
                improvement = (mean_b - mean_a) / mean_a * 100
        else:
            winner = "No significant difference"
            improvement = 0
        
        result = {
            'name': name,
            'prompt_a_score': mean_a,
            'prompt_b_score': mean_b,
            'p_value': p_value,
            'winner': winner,
            'improvement': improvement,
            'statistically_significant': p_value < 0.05,
            'sample_size': len(test_cases)
        }
        
        self.experiments.append(result)
        return result
    
    def print_results(self, result: Dict):
        """Print A/B test results"""
        print("\n" + "="*60)
        print(f"A/B Test: {result['name']}")
        print("="*60)
        print(f"Prompt A Score: {result['prompt_a_score']:.2%}")
        print(f"Prompt B Score: {result['prompt_b_score']:.2%}")
        print(f"P-value: {result['p_value']:.4f}")
        print(f"Winner: {result['winner']}")
        if result['statistically_significant']:
            print(f"Improvement: {result['improvement']:.1f}%")
            print("✓ Statistically significant (p < 0.05)")
        else:
            print("✗ Not statistically significant")
        print(f"Sample size: {result['sample_size']}")

# Usage
if __name__ == "__main__":
    from langchain.llms import Ollama
    
    llm = Ollama(model="llama3:8b")
    tester = ABTester()
    
    prompt_a = "Classify: {text}\nSentiment:"
    prompt_b = "Analyze the sentiment of: {text}\nProvide sentiment (positive/negative/neutral):"
    
    result = tester.run_ab_test(
        prompt_a,
        prompt_b,
        SENTIMENT_TEST_CASES,
        sentiment_metric,
        llm,
        name="Sentiment Classification"
    )
    
    tester.print_results(result)
```

## Success Criteria

By the end of this project, you should have:

- [ ] Template library with 10+ reusable templates
- [ ] Zero-shot prompts for classification, extraction, QA, summarization
- [ ] Few-shot learning with static and dynamic example selection
- [ ] Chain-of-thought prompting with self-consistency
- [ ] ReAct pattern with tool integration
- [ ] DSPy optimization working
- [ ] Comprehensive evaluation framework
- [ ] A/B testing capability
- [ ] Consistency testing
- [ ] Cost calculation
- [ ] Gradio UI for interactive testing
- [ ] Documentation with best practices
- [ ] 20-50% performance improvement demonstrated
- [ ] GitHub repository with examples

## Learning Outcomes

After completing this project, you'll be able to:

- Design effective prompts for various tasks
- Choose between zero-shot, few-shot, and CoT approaches
- Implement dynamic example selection
- Use chain-of-thought for complex reasoning
- Build ReAct agents with tool use
- Optimize prompts automatically with DSPy
- Evaluate prompts rigorously
- Conduct A/B tests for prompt comparison
- Calculate and optimize prompt costs
- Explain prompt engineering best practices
- Debug and improve underperforming prompts

## Expected Performance Improvements

Based on typical results:

**Zero-shot → Few-shot**:
- Classification: 65% → 82% (+26% improvement)
- Extraction: 58% → 75% (+29% improvement)
- QA: 70% → 85% (+21% improvement)

**Few-shot → Chain-of-Thought**:
- Math reasoning: 45% → 78% (+73% improvement)
- Logic problems: 52% → 81% (+56% improvement)
- Multi-step tasks: 48% → 76% (+58% improvement)

**Manual → DSPy Optimized**:
- Overall accuracy: +15-25% improvement
- Consistency: +30-40% improvement
- Token efficiency: -20-30% reduction

## Project Structure

```
project-g-prompt-engineering/
├── src/
│   ├── templates/
│   │   ├── base_templates.py
│   │   ├── few_shot.py
│   │   ├── chain_of_thought.py
│   │   └── react_pattern.py
│   ├── optimization/
│   │   ├── dspy_optimizer.py
│   │   └── manual_tuning.py
│   ├── evaluation/
│   │   ├── prompt_evaluator.py
│   │   ├── ab_testing.py
│   │   └── metrics.py
│   └── ui/
│       └── gradio_app.py
├── templates/
│   ├── classification.json
│   ├── extraction.json
│   ├── qa.json
│   └── summarization.json
├── examples/
│   ├── sentiment_analysis.py
│   ├── math_reasoning.py
│   ├── code_generation.py
│   └── react_agent.py
├── evaluations/
│   ├── test_cases.json
│   ├── benchmark_results.csv
│   └── ab_test_results.json
├── notebooks/
│   ├── 01_template_basics.ipynb
│   ├── 02_few_shot_learning.ipynb
│   ├── 03_chain_of_thought.ipynb
│   ├── 04_dspy_optimization.ipynb
│   └── 05_evaluation.ipynb
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges & Solutions

### Challenge 1: Inconsistent Outputs
**Problem**: Same prompt produces different outputs
**Solution**: Use temperature=0, add explicit formatting instructions, use self-consistency
```python
# Set temperature to 0 for deterministic outputs
llm = Ollama(model="llama3:8b", temperature=0)

# Add explicit format instructions
prompt = """Classify sentiment. Respond with ONLY one word: positive, negative, or neutral.

Text: {text}
Sentiment:"""
```

### Challenge 2: Poor Few-Shot Example Selection
**Problem**: Examples don't help model performance
**Solution**: Use diverse, representative examples with semantic similarity selection
```python
# Use semantic similarity for better examples
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    Chroma,
    k=3  # Select 3 most similar examples
)
```

### Challenge 3: Chain-of-Thought Not Working
**Problem**: Model doesn't follow reasoning steps
**Solution**: Provide explicit step-by-step examples, use "Let's think step by step"
```python
prompt = """Let's solve this step by step:

Question: {question}

Step 1: Understand the problem
Step 2: Identify key information
Step 3: Apply logic
Step 4: Provide answer

Solution:"""
```

### Challenge 4: High Token Costs
**Problem**: Prompts are too long and expensive
**Solution**: Optimize prompt length, use smaller models, cache responses
```python
# Shorter, more efficient prompt
prompt = "Classify: {text}\nSentiment:"  # Instead of verbose instructions

# Use smaller model for simple tasks
llm = Ollama(model="phi3:mini")  # 2.3GB vs 4.7GB
```

## Advanced Techniques

### 1. Prompt Chaining
```python
# Break complex task into steps
step1_prompt = "Extract key facts from: {text}"
step2_prompt = "Summarize these facts: {facts}"
step3_prompt = "Generate title from summary: {summary}"

# Chain them together
facts = llm.invoke(step1_prompt.format(text=text))
summary = llm.invoke(step2_prompt.format(facts=facts))
title = llm.invoke(step3_prompt.format(summary=summary))
```

### 2. Prompt Ensembling
```python
# Use multiple prompts and vote
prompts = [prompt1, prompt2, prompt3]
results = [llm.invoke(p.format(text=text)) for p in prompts]

# Majority vote
from collections import Counter
final_answer = Counter(results).most_common(1)[0][0]
```

### 3. Retrieval-Augmented Prompting
```python
# Retrieve relevant examples dynamically
relevant_examples = retrieve_similar_examples(query, k=3)

# Build prompt with retrieved examples
prompt = build_few_shot_prompt(relevant_examples, query)
```

### 4. Meta-Prompting
```python
# Use LLM to generate prompts
meta_prompt = """Generate an effective prompt for this task: {task_description}

The prompt should:
1. Be clear and specific
2. Include examples if helpful
3. Specify output format

Generated prompt:"""

optimized_prompt = llm.invoke(meta_prompt.format(task_description=task))
```

## Troubleshooting

### Installation Issues

**Issue**: DSPy installation fails
```bash
# Solution: Install from GitHub
pip install git+https://github.com/stanfordnlp/dspy.git
```

**Issue**: Ollama connection errors
```bash
# Solution: Verify Ollama is running
ollama list
ollama serve  # Start Ollama server
```

### Runtime Issues

**Issue**: "Model not found" error
```python
# Solution: Pull model first
import subprocess
subprocess.run(["ollama", "pull", "llama3:8b"])
```

**Issue**: Slow inference
```python
# Solution: Use smaller model or reduce max_tokens
llm = Ollama(model="phi3:mini", num_predict=100)
```

## Production Deployment

### Gradio UI
```python
# src/ui/gradio_app.py
import gradio as gr
from langchain.llms import Ollama

llm = Ollama(model="llama3:8b")
library = PromptTemplateLibrary()

def test_prompt(template_name, input_text):
    template = library.get_template(template_name)
    prompt = template.format(text=input_text)
    response = llm.invoke(prompt)
    return response

demo = gr.Interface(
    fn=test_prompt,
    inputs=[
        gr.Dropdown(library.list_templates(), label="Template"),
        gr.Textbox(label="Input Text", lines=3)
    ],
    outputs=gr.Textbox(label="Output", lines=5),
    title="Prompt Engineering Framework",
    description="Test different prompt templates"
)

demo.launch()
```

### API Server
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PromptRequest(BaseModel):
    template: str
    inputs: dict

@app.post("/generate")
async def generate(request: PromptRequest):
    template = library.get_template(request.template)
    prompt = template.format(**request.inputs)
    response = llm.invoke(prompt)
    return {"response": response}
```

## Resources

### Documentation
- [Prompt Engineering Guide](https://www.promptingguide.ai/) - Comprehensive guide
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering) - Official guide
- [DSPy Documentation](https://dspy-docs.vercel.app/) - Automatic optimization
- [LangChain Prompts](https://python.langchain.com/docs/modules/model_io/prompts/) - Template library

### Papers
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - Original paper
- [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629) - ReAct pattern
- [Self-Consistency](https://arxiv.org/abs/2203.11171) - Consistency technique
- [DSPy: Programming with Foundation Models](https://arxiv.org/abs/2310.03714) - DSPy paper

### Tutorials
- [Prompt Engineering Course](https://learnprompting.org/) - Free course
- [LangChain Prompt Tutorial](https://python.langchain.com/docs/modules/model_io/prompts/quick_start) - Quick start
- [Few-Shot Learning Guide](https://www.promptingguide.ai/techniques/fewshot) - Techniques

## Questions?

If you get stuck:
1. Review the `tech-spec.md` for detailed patterns
2. Check prompt engineering guides for best practices
3. Test with simpler prompts first
4. Review the 100 Days bootcamp materials
5. Compare your prompts with examples
6. Use evaluation framework to identify issues

## Related Projects

After completing this project, consider:
- **Project A**: Local RAG - Apply prompt engineering to RAG
- **Project F**: Inference Optimization - Optimize prompt inference
- **Project H**: Embeddings Search - Optimize retrieval prompts
- Build a prompt marketplace or library for your team
