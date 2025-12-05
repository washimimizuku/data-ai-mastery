# Technical Specification: LLM Fine-Tuning & Inference Optimization

## Architecture
```
Dataset → Preprocessing → LoRA Fine-Tuning → Evaluation → Model Registry
                                                              ↓
                                                         vLLM Server
                                                              ↓
                                                         FastAPI
```

## Technology Stack
- **Framework**: Hugging Face Transformers, PEFT
- **Training**: PyTorch, bitsandbytes
- **Inference**: vLLM, TGI (Text Generation Inference)
- **API**: FastAPI
- **Platform**: AWS SageMaker, Modal, or RunPod
- **Tracking**: MLflow, Weights & Biases

## Model Selection
- Llama 3 (8B, 70B)
- Mistral 7B
- Phi-3
- CodeLlama (for code tasks)

## Fine-Tuning with LoRA

### Dataset Preparation
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="training_data.jsonl")

def format_instruction(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

dataset = dataset.map(format_instruction)
```

### LoRA Configuration
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
```

### Training with QLoRA
```python
from transformers import TrainingArguments, Trainer
import bitsandbytes as bnb

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

## Inference Optimization

### vLLM Setup
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="path/to/finetuned/model",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

outputs = llm.generate(prompts, sampling_params)
```

### FastAPI with Streaming
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.post("/generate")
async def generate(prompt: str):
    async def stream_response():
        async for token in llm.generate_stream(prompt):
            yield f"data: {token}\n\n"
    
    return StreamingResponse(stream_response(), media_type="text/event-stream")

@app.post("/generate/batch")
async def generate_batch(prompts: List[str]):
    outputs = llm.generate(prompts, sampling_params)
    return {"results": [output.text for output in outputs]}
```

## Evaluation

### Perplexity
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.max_position_embeddings
    
    nlls = []
    for i in range(0, encodings.input_ids.size(1), max_length):
        begin_loc = i
        end_loc = min(i + max_length, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll = outputs.loss
        
        nlls.append(nll)
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
```

### Task-Specific Evaluation
```python
def evaluate_sql_generation(model, test_cases):
    correct = 0
    for case in test_cases:
        generated_sql = model.generate(case['question'])
        if validate_sql(generated_sql, case['expected']):
            correct += 1
    
    accuracy = correct / len(test_cases)
    return accuracy
```

## Model Comparison

### Benchmarking
```python
import time

def benchmark_model(model, prompts, num_runs=100):
    latencies = []
    
    for _ in range(num_runs):
        start = time.time()
        model.generate(prompts)
        latencies.append(time.time() - start)
    
    return {
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "throughput": num_runs / sum(latencies)
    }
```

## Cost Analysis

### Training Costs
```python
# Example for AWS SageMaker
instance_type = "ml.g5.2xlarge"  # $1.52/hour
training_hours = 4
cost = 1.52 * training_hours  # $6.08
```

### Inference Costs
```python
# vLLM on g5.xlarge
hourly_cost = 1.006
requests_per_hour = 3600  # 1 req/sec
cost_per_1k_requests = (hourly_cost / requests_per_hour) * 1000
# $0.28 per 1K requests
```

## Deployment Options

### Modal Deployment
```python
import modal

stub = modal.Stub("llm-inference")

@stub.function(
    gpu="A10G",
    image=modal.Image.debian_slim().pip_install("vllm", "fastapi")
)
@modal.asgi_app()
def fastapi_app():
    return app
```

### Docker Deployment
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN pip install vllm fastapi uvicorn

COPY model /model
COPY app.py /app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Performance Benchmarks

| Configuration | Latency (ms) | Throughput (req/s) | Cost ($/1M tokens) |
|---------------|--------------|-------------------|-------------------|
| Base Model (API) | 2000 | 0.5 | $10.00 |
| Fine-tuned (vLLM) | 800 | 1.25 | $2.50 |
| Quantized (4-bit) | 400 | 2.5 | $1.25 |

## Monitoring
- Token generation speed
- GPU utilization
- Memory usage
- Request latency
- Error rates
