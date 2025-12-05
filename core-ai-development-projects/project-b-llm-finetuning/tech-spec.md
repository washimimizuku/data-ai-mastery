# Technical Specification: Fine-Tuning Small LLM

## Architecture
```
Base Model → Dataset Preparation → LoRA/QLoRA Fine-tuning → Evaluation → Export
```

## Technology Stack
- **Python**: 3.11+
- **Framework**: Hugging Face Transformers, PEFT
- **Training**: PyTorch, bitsandbytes (for QLoRA)
- **Models**: Phi-3, Mistral-7B, Llama-3-8B
- **UI**: Gradio

## Models to Fine-Tune

### Option 1: Phi-3 Mini (3.8B)
- Size: ~2.3GB (4-bit)
- Fast training
- Good for CPU inference

### Option 2: Mistral 7B
- Size: ~4.1GB (4-bit)
- High quality
- Balanced performance

### Option 3: Llama-3-8B
- Size: ~4.7GB (4-bit)
- Best quality
- Slower training

## Fine-Tuning Techniques

### LoRA (Low-Rank Adaptation)
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,          # Scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
```

### QLoRA (Quantized LoRA)
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

## Dataset Preparation

### Task Options
1. **SQL Generation** - Text to SQL queries
2. **Code Review** - Review and improve code
3. **Technical Q&A** - Answer technical questions
4. **Summarization** - Summarize technical docs

### Dataset Format (Instruction Tuning)
```json
{
  "instruction": "Generate a SQL query for the following request",
  "input": "Get all users who signed up in the last 30 days",
  "output": "SELECT * FROM users WHERE signup_date >= CURRENT_DATE - INTERVAL '30 days'"
}
```

### Data Processing
```python
def format_instruction(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
```

## Training Configuration

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    warmup_steps=100,
    lr_scheduler_type="cosine"
)
```

## Evaluation Metrics

### Quantitative
- Perplexity
- Task-specific accuracy
- BLEU/ROUGE (for generation tasks)
- Exact match (for structured outputs)

### Qualitative
- Manual review of outputs
- Comparison with base model
- Edge case handling

## Model Export

### Save LoRA Weights
```python
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
```

### Merge and Export
```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, "./fine-tuned-model")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

### Convert to GGUF (for Ollama)
```bash
# Using llama.cpp converter
python convert.py ./merged-model --outtype f16 --outfile model.gguf
```

## Project Structure
```
project-b-llm-finetuning/
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── export.py
├── data/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── configs/
│   ├── lora_config.yaml
│   └── training_config.yaml
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── models/
│   ├── base/
│   ├── fine-tuned/
│   └── merged/
├── app.py              # Gradio demo
└── README.md
```

## Hardware Requirements

### Minimum (CPU only)
- RAM: 16GB
- Training time: 8-12 hours

### Recommended (GPU)
- GPU: 8GB+ VRAM
- RAM: 16GB
- Training time: 2-4 hours

### Optimal
- GPU: 24GB+ VRAM (RTX 4090, A100)
- RAM: 32GB
- Training time: 1-2 hours
