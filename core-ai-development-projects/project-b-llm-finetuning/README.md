# Project B: LLM Fine-Tuning with LoRA/QLoRA

## Objective

Fine-tune a small open-source LLM (Phi-3, Mistral-7B, or Llama-3-8B) using parameter-efficient techniques (LoRA/QLoRA) for a specific task, demonstrating modern fine-tuning practices with measurable improvements.

**What You'll Build**: A complete fine-tuning pipeline that takes a base model, fine-tunes it on a custom dataset using LoRA/QLoRA, evaluates performance improvements, exports the model, and deploys it with a Gradio interface for comparison.

**What You'll Learn**: Parameter-efficient fine-tuning (LoRA, QLoRA), instruction tuning, dataset preparation, Hugging Face Trainer, evaluation metrics, model merging, GGUF export for Ollama, and deployment strategies.

## Time Estimate

**2 days (16 hours)** - Following the implementation plan

### Day 1 (8 hours)
- **Hour 1**: Setup & model selection (install dependencies, choose model, test base)
- **Hours 2-3**: Dataset preparation (select task, create 500-1000 examples, format, split)
- **Hours 4-5**: LoRA configuration (parameters, training args, test on subset)
- **Hours 6-8**: Training (full run, monitor metrics, save checkpoints, validation)

### Day 2 (8 hours)
- **Hours 1-2**: Evaluation (test set metrics, compare with base, qualitative analysis)
- **Hours 3-4**: Model export (save LoRA, merge weights, export, convert to GGUF)
- **Hours 5-6**: Gradio demo (inference function, UI, base vs fine-tuned comparison)
- **Hour 7**: Documentation (training report, hyperparameters, results, usage guide)
- **Hour 8**: Notebooks & polish (Jupyter notebooks, visualizations, cleanup)

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 71-92
  - Days 71-80: LLM fundamentals
  - Days 81-92: Advanced GenAI patterns
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 40-53
  - Days 40-45: Fine-tuning techniques
  - Days 46-53: Model optimization

### Technical Requirements
- Python 3.11+ installed
- 16GB+ RAM (32GB recommended)
- GPU with 8GB+ VRAM (or Google Colab)
- Understanding of transformers and LLMs
- Basic PyTorch knowledge

### Tools Needed
- Python with transformers, peft, bitsandbytes
- Hugging Face account (for model access)
- GPU (or Google Colab with T4/A100)
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
pip install torch transformers peft bitsandbytes accelerate datasets

# Install additional tools
pip install gradio wandb scipy

# For model export (optional)
pip install llama-cpp-python

# Create project structure
mkdir -p llm-finetuning/{src,data,configs,models,notebooks}
cd llm-finetuning
```

### Step 3: Choose Base Model and Task

**Model Options**:
```python
MODELS = {
    "phi-3": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "size": "3.8B parameters",
        "4bit_size": "~2.3GB",
        "training_time": "2-3 hours (T4 GPU)",
        "best_for": "Fast training, CPU inference"
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-v0.1",
        "size": "7B parameters",
        "4bit_size": "~4.1GB",
        "training_time": "4-6 hours (T4 GPU)",
        "best_for": "Balanced quality and speed"
    },
    "llama-3": {
        "name": "meta-llama/Meta-Llama-3-8B",
        "size": "8B parameters",
        "4bit_size": "~4.7GB",
        "training_time": "5-7 hours (T4 GPU)",
        "best_for": "Best quality outputs"
    }
}
```

**Task Options**:
1. **SQL Generation**: Text to SQL queries (structured output)
2. **Code Review**: Review and improve code (technical analysis)
3. **Technical Q&A**: Answer technical questions (knowledge-based)
4. **Summarization**: Summarize technical documents (compression)

### Step 4: Prepare Dataset
```python
# src/prepare_data.py
from datasets import Dataset
import json

# Example: SQL Generation dataset
def create_sql_dataset():
    """Create instruction-tuning dataset for SQL generation"""
    data = [
        {
            "instruction": "Generate a SQL query for the following request",
            "input": "Get all users who signed up in the last 30 days",
            "output": "SELECT * FROM users WHERE signup_date >= CURRENT_DATE - INTERVAL '30 days'"
        },
        {
            "instruction": "Generate a SQL query for the following request",
            "input": "Find the top 10 products by revenue",
            "output": "SELECT product_id, product_name, SUM(revenue) as total_revenue FROM sales GROUP BY product_id, product_name ORDER BY total_revenue DESC LIMIT 10"
        },
        # Add 500-1000 examples
    ]
    
    return data

def format_instruction(example):
    """Format example for instruction tuning"""
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    }

# Create and save dataset
data = create_sql_dataset()
dataset = Dataset.from_list(data)

# Format for training
formatted_dataset = dataset.map(format_instruction)

# Split into train/val/test
train_test = formatted_dataset.train_test_split(test_size=0.2, seed=42)
test_val = train_test['test'].train_test_split(test_size=0.5, seed=42)

train_dataset = train_test['train']
val_dataset = test_val['train']
test_dataset = test_val['test']

# Save
train_dataset.to_json('data/train.json')
val_dataset.to_json('data/val.json')
test_dataset.to_json('data/test.json')

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
```

### Step 5: Configure LoRA/QLoRA
```python
# src/train.py
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# QLoRA configuration (4-bit quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load base model
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
    r=16,                           # Rank (higher = more parameters)
    lora_alpha=32,                  # Scaling factor
    target_modules=[                # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
```

### Step 6: Train Model
```python
# Training arguments
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
    lr_scheduler_type="cosine",
    report_to="wandb",  # Optional: for tracking
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Start training
print("Starting training...")
trainer.train()

# Save model
model.save_pretrained("./models/fine-tuned")
tokenizer.save_pretrained("./models/fine-tuned")
print("✓ Training complete! Model saved.")
```

### Step 7: Evaluate Model
```python
# src/evaluate.py
from peft import PeftModel
import torch

def evaluate_model(base_model_name, finetuned_path, test_dataset):
    """Compare base vs fine-tuned model"""
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load fine-tuned model
    finetuned_model = PeftModel.from_pretrained(base_model, finetuned_path)
    
    results = {"base": [], "finetuned": []}
    
    for example in test_dataset:
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Base model
        with torch.no_grad():
            base_output = base_model.generate(**inputs, max_new_tokens=256)
            base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)
            results["base"].append(base_text)
        
        # Fine-tuned model
        with torch.no_grad():
            ft_output = finetuned_model.generate(**inputs, max_new_tokens=256)
            ft_text = tokenizer.decode(ft_output[0], skip_special_tokens=True)
            results["finetuned"].append(ft_text)
    
    return results

# Run evaluation
results = evaluate_model(
    "microsoft/Phi-3-mini-4k-instruct",
    "./models/fine-tuned",
    test_dataset
)

# Calculate metrics (example: exact match for SQL)
exact_matches_base = sum(1 for i, ex in enumerate(test_dataset) 
                         if ex['output'] in results['base'][i])
exact_matches_ft = sum(1 for i, ex in enumerate(test_dataset) 
                       if ex['output'] in results['finetuned'][i])

print(f"Base model exact matches: {exact_matches_base}/{len(test_dataset)}")
print(f"Fine-tuned exact matches: {exact_matches_ft}/{len(test_dataset)}")
print(f"Improvement: +{exact_matches_ft - exact_matches_base}")
```

### Step 8: Export Model
```python
# src/export.py
from peft import PeftModel

# Load and merge
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(base_model, "./models/fine-tuned")

# Merge LoRA weights into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./models/merged")
tokenizer.save_pretrained("./models/merged")

print("✓ Model merged and saved")

# Optional: Convert to GGUF for Ollama
# Requires llama.cpp tools
# python convert.py ./models/merged --outtype f16 --outfile model.gguf
```

### Step 9: Create Gradio Demo
```python
# app.py
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load models
print("Loading models...")
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

finetuned_model = PeftModel.from_pretrained(base_model, "./models/fine-tuned")
print("✓ Models loaded")

def generate_response(instruction, input_text, use_finetuned=True):
    """Generate response from model"""
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    model = finetuned_model if use_finetuned else base_model
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = response.split("### Response:")[-1].strip()
    
    return response

def compare_models(instruction, input_text):
    """Compare base vs fine-tuned model"""
    base_response = generate_response(instruction, input_text, use_finetuned=False)
    ft_response = generate_response(instruction, input_text, use_finetuned=True)
    
    return base_response, ft_response

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Fine-Tuned LLM Comparison")
    gr.Markdown("Compare base model vs fine-tuned model outputs")
    
    with gr.Row():
        with gr.Column():
            instruction = gr.Textbox(
                label="Instruction",
                placeholder="Generate a SQL query for the following request",
                lines=2
            )
            input_text = gr.Textbox(
                label="Input",
                placeholder="Get all users who signed up in the last 30 days",
                lines=3
            )
            submit_btn = gr.Button("Generate", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Base Model")
            base_output = gr.Textbox(label="Output", lines=10)
        
        with gr.Column():
            gr.Markdown("### Fine-Tuned Model")
            ft_output = gr.Textbox(label="Output", lines=10)
    
    submit_btn.click(
        fn=compare_models,
        inputs=[instruction, input_text],
        outputs=[base_output, ft_output]
    )
    
    gr.Examples(
        examples=[
            ["Generate a SQL query", "Find top 10 customers by revenue"],
            ["Generate a SQL query", "Get average order value by month"],
            ["Generate a SQL query", "List products with low stock"]
        ],
        inputs=[instruction, input_text]
    )

if __name__ == "__main__":
    demo.launch(share=False)
```

## Key Features to Implement

### 1. Model Selection
- Choose from Phi-3, Mistral-7B, or Llama-3-8B
- Evaluate base model performance
- Document model characteristics

### 2. Dataset Preparation
- Select/create fine-tuning dataset (500-1000 examples)
- Format for instruction tuning
- Train/validation/test split (80/10/10)
- Data quality checks

### 3. Fine-Tuning
- **LoRA**: Parameter-efficient adaptation
- **QLoRA**: 4-bit quantization for memory efficiency
- Configure hyperparameters (rank, alpha, dropout)
- Training with Hugging Face Trainer
- Monitor training metrics (loss, perplexity)

### 4. Evaluation
- Compare base vs fine-tuned model
- **Quantitative metrics**:
  - Perplexity
  - Task-specific accuracy
  - BLEU/ROUGE (for generation)
  - Exact match (for structured outputs)
- **Qualitative evaluation**:
  - Manual review of outputs
  - Edge case handling

### 5. Model Export
- Save LoRA weights
- Merge LoRA with base model
- Export merged model
- Convert to GGUF for Ollama (optional)
- Quantization options

### 6. Inference & Demo
- Load and test fine-tuned model
- Compare inference speed
- Gradio demo interface with side-by-side comparison

## Success Criteria

By the end of this project, you should have:

### Functionality
- [ ] Base model selected and tested
- [ ] Dataset prepared (500-1000 examples, formatted, split)
- [ ] LoRA/QLoRA configuration implemented
- [ ] Training completed successfully
- [ ] Model evaluated on test set
- [ ] LoRA weights saved
- [ ] Model merged and exported
- [ ] Gradio demo working

### Quality Metrics
- [ ] **Measurable improvement** over base model (>10% on task metric)
- [ ] **Training time**: < 4 hours on GPU
- [ ] **Code quality**: < 600 lines of code
- [ ] **Trainable parameters**: < 1% of total parameters (LoRA efficiency)

### Deliverables
- [ ] Fine-tuned model files
- [ ] Training scripts (prepare_data.py, train.py, evaluate.py, export.py)
- [ ] Evaluation results with comparison
- [ ] Gradio demo application
- [ ] 3 Jupyter notebooks (data prep, training, evaluation)
- [ ] Training report with hyperparameters
- [ ] Comprehensive documentation

## Learning Outcomes

After completing this project, you'll be able to:

- Understand parameter-efficient fine-tuning (LoRA, QLoRA)
- Prepare datasets for instruction tuning
- Configure and train models with Hugging Face
- Evaluate fine-tuned models quantitatively and qualitatively
- Merge LoRA weights and export models
- Deploy fine-tuned models with Gradio
- Explain trade-offs between LoRA and full fine-tuning
- Optimize training for limited GPU resources

## Expected Results

**Training Performance**:
```
Model: Phi-3 Mini (3.8B)
GPU: T4 (16GB VRAM)
Dataset: 800 examples

Training time: 2-3 hours
Memory usage: 6-8GB VRAM (QLoRA)
Trainable parameters: 8.4M (0.22% of total)

Base model accuracy: 45%
Fine-tuned accuracy: 78%
Improvement: +33 percentage points
```

**Model Comparison**:
```
Phi-3 Mini:
  Training: 2-3 hours
  Quality: Good
  Best for: Fast iteration

Mistral 7B:
  Training: 4-6 hours
  Quality: Better
  Best for: Balanced

Llama-3 8B:
  Training: 5-7 hours
  Quality: Best
  Best for: Maximum quality
```

## Project Structure

```
project-b-llm-finetuning/
├── src/
│   ├── prepare_data.py      # Dataset creation and formatting
│   ├── train.py             # LoRA/QLoRA training script
│   ├── evaluate.py          # Model evaluation
│   └── export.py            # Model merging and export
├── data/
│   ├── train.json           # Training data (80%)
│   ├── val.json             # Validation data (10%)
│   └── test.json            # Test data (10%)
├── configs/
│   ├── lora_config.yaml     # LoRA hyperparameters
│   └── training_config.yaml # Training arguments
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── models/
│   ├── base/                # Base model cache
│   ├── fine-tuned/          # LoRA weights
│   └── merged/              # Merged model
├── results/
│   ├── training_report.md   # Training metrics and logs
│   ├── evaluation_results.json
│   └── comparison_chart.png
├── app.py                   # Gradio demo
├── requirements.txt
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Hardware Requirements

### Minimum (CPU only)
- RAM: 16GB
- Training time: 8-12 hours
- Not recommended for production

### Recommended (GPU)
- GPU: 8GB+ VRAM (RTX 3060, T4)
- RAM: 16GB
- Training time: 2-4 hours
- Good for development

### Optimal
- GPU: 24GB+ VRAM (RTX 4090, A100)
- RAM: 32GB
- Training time: 1-2 hours
- Best for experimentation

**Google Colab**: Free T4 GPU (16GB VRAM) works well for this project

## Common Challenges & Solutions

### Challenge 1: Out of Memory (OOM)
**Problem**: GPU runs out of memory during training
**Solution**: 
- Use QLoRA (4-bit quantization)
- Reduce batch size
- Increase gradient accumulation steps
- Use smaller model (Phi-3 instead of Llama-3)

### Challenge 2: Slow Training
**Problem**: Training takes too long
**Solution**:
- Use GPU instead of CPU
- Reduce dataset size for testing
- Use smaller model
- Enable fp16/bf16 training

### Challenge 3: No Improvement
**Problem**: Fine-tuned model not better than base
**Solution**:
- Check data quality and formatting
- Increase training epochs
- Adjust learning rate
- Ensure sufficient dataset size (500+ examples)

### Challenge 4: Model Not Loading
**Problem**: Cannot load fine-tuned model
**Solution**:
- Verify LoRA weights saved correctly
- Check model and tokenizer compatibility
- Use correct loading method (PeftModel.from_pretrained)

## Next Steps

After completing this project:

1. **Add to Portfolio**: Document on GitHub with training results
2. **Write Blog Post**: "Fine-Tuning LLMs with LoRA: A Practical Guide"
3. **Extend Features**: Try different tasks, compare LoRA vs full fine-tuning
4. **Build Project C**: Continue with Multi-Agent System
5. **Production Use**: Deploy fine-tuned model with FastAPI

## Resources

- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Fine-Tuning Guide](https://huggingface.co/blog/fine-tune-llms)

## Questions?

If you get stuck:
1. Review the tech-spec.md for detailed configuration
2. Check implementation-plan.md for step-by-step timeline
3. Search Hugging Face forums for specific errors
4. Review the 100 Days bootcamp materials on fine-tuning
5. Check GPU memory usage with `nvidia-smi`
