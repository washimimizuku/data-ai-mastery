# Project F: LLM Inference Optimization

## Objective

Build a comprehensive LLM inference optimization toolkit that demonstrates quantization techniques (4-bit, 8-bit, GPTQ, AWQ), multiple inference engines (vLLM, llama.cpp, ONNX), and performance benchmarking to achieve 2-10x speedup with minimal quality loss.

**What You'll Build**: A complete inference optimization pipeline that takes a base LLM, applies various quantization methods, benchmarks performance across different engines, and provides data-driven recommendations for production deployment.

**What You'll Learn**: Model quantization techniques, inference engine optimization, memory management, latency vs quality trade-offs, GGUF format conversion, PagedAttention, KV cache optimization, and production deployment strategies.

## Time Estimate

**2 days (16 hours)** - Following the implementation plan

### Day 1 (8 hours)
- **Hour 1**: Setup & environment (install dependencies, download models, verify GPU)
- **Hours 2-3**: Quantization techniques (4-bit, 8-bit, GPTQ, AWQ implementations)
- **Hours 4-5**: llama.cpp integration (GGUF conversion, CPU inference, benchmarking)
- **Hours 6-7**: vLLM setup (PagedAttention, batch inference, throughput testing)
- **Hour 8**: ONNX conversion (export model, optimize, benchmark)

### Day 2 (8 hours)
- **Hours 1-2**: Comprehensive benchmarking (latency, throughput, memory, quality)
- **Hours 3-4**: Quality evaluation (perplexity, accuracy on test set, output comparison)
- **Hours 5-6**: Optimization analysis (trade-off charts, recommendations, cost analysis)
- **Hour 7**: Production deployment guide (Docker, API serving, monitoring)
- **Hour 8**: Documentation & polish (README, benchmark report, usage examples)

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 71-92
  - Days 71-75: LLM fundamentals and architecture
  - Days 76-80: Model optimization basics
  - Days 81-85: Inference and serving
  - Days 86-92: Production deployment
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 40-53
  - Days 40-45: Advanced optimization techniques
  - Days 46-53: Inference engines and serving

### Technical Requirements
- Python 3.11+ installed
- CUDA-capable GPU (8GB+ VRAM recommended) or CPU with 16GB+ RAM
- 50GB+ free disk space for models and quantized versions
- Understanding of transformer architecture
- Basic knowledge of quantization concepts

### Tools Needed
- Python with transformers, bitsandbytes, auto-gptq
- vLLM for GPU inference
- llama.cpp for CPU inference
- ONNX Runtime for cross-platform
- Docker for deployment
- Git for version control

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install torch transformers accelerate

# Install quantization libraries
pip install bitsandbytes auto-gptq autoawq

# Install inference engines
pip install vllm llama-cpp-python onnxruntime-gpu

# Install utilities
pip install datasets evaluate pandas matplotlib

# Verify GPU (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Download Base Model
```python
# download_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"

print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Save locally
model.save_pretrained("./models/mistral-7b-base")
tokenizer.save_pretrained("./models/mistral-7b-base")
print("✓ Model downloaded and saved")
```

### Step 4: Implement 4-bit Quantization
```python
# src/quantization/bitsandbytes_quant.py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

class BitsAndBytesQuantizer:
    """4-bit and 8-bit quantization using bitsandbytes"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
    
    def quantize_4bit(self):
        """Load model with 4-bit quantization"""
        print("Loading model with 4-bit quantization...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        print("✓ 4-bit quantization complete")
        return model
    
    def quantize_8bit(self):
        """Load model with 8-bit quantization"""
        print("Loading model with 8-bit quantization...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_8bit=True,
            device_map="auto"
        )
        
        print("✓ 8-bit quantization complete")
        return model
    
    def get_model_size(self, model):
        """Calculate model size in GB"""
        param_size = sum(p.nelement() * p.element_size() 
                        for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() 
                         for b in model.buffers())
        size_gb = (param_size + buffer_size) / (1024**3)
        return size_gb

# Usage
if __name__ == "__main__":
    quantizer = BitsAndBytesQuantizer("./models/mistral-7b-base")
    
    # 4-bit quantization
    model_4bit = quantizer.quantize_4bit()
    size_4bit = quantizer.get_model_size(model_4bit)
    print(f"4-bit model size: {size_4bit:.2f} GB")
    
    # 8-bit quantization
    model_8bit = quantizer.quantize_8bit()
    size_8bit = quantizer.get_model_size(model_8bit)
    print(f"8-bit model size: {size_8bit:.2f} GB")
```

### Step 5: Implement GPTQ Quantization
```python
# src/quantization/gptq_quant.py
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import torch

class GPTQQuantizer:
    """GPTQ quantization for aggressive compression"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def quantize(self, bits: int = 4, group_size: int = 128):
        """Quantize model using GPTQ"""
        print(f"Quantizing with GPTQ ({bits}-bit, group_size={group_size})...")
        
        # Quantization config
        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=False,
        )
        
        # Load and quantize
        model = AutoGPTQForCausalLM.from_pretrained(
            self.model_path,
            quantize_config=quantize_config
        )
        
        # Prepare calibration data
        examples = self._get_calibration_data()
        
        # Quantize
        model.quantize(examples)
        
        # Save quantized model
        save_path = f"./models/mistral-7b-gptq-{bits}bit"
        model.save_quantized(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"✓ GPTQ quantization complete, saved to {save_path}")
        return model
    
    def _get_calibration_data(self, n_samples: int = 128):
        """Get calibration dataset for quantization"""
        from datasets import load_dataset
        
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        examples = []
        
        for i in range(n_samples):
            text = dataset[i]["text"]
            if len(text) > 50:  # Skip short texts
                examples.append(text)
        
        return examples[:n_samples]

# Usage
if __name__ == "__main__":
    quantizer = GPTQQuantizer("./models/mistral-7b-base")
    
    # 4-bit GPTQ
    model_gptq = quantizer.quantize(bits=4, group_size=128)
```

### Step 6: Set Up vLLM for Fast Inference

```python
# src/engines/vllm_engine.py
from vllm import LLM, SamplingParams
import time

class vLLMEngine:
    """vLLM inference engine with PagedAttention"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        print(f"Loading model with vLLM...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9,
            max_model_len=2048
        )
        print("✓ vLLM model loaded")
    
    def generate(self, prompts: list[str], max_tokens: int = 100):
        """Generate responses for multiple prompts"""
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens
        )
        
        start = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        elapsed = time.time() - start
        
        results = []
        for output in outputs:
            results.append({
                'prompt': output.prompt,
                'generated_text': output.outputs[0].text,
                'tokens': len(output.outputs[0].token_ids)
            })
        
        return results, elapsed
    
    def benchmark_throughput(self, prompts: list[str]):
        """Benchmark throughput (tokens/second)"""
        results, elapsed = self.generate(prompts)
        
        total_tokens = sum(r['tokens'] for r in results)
        throughput = total_tokens / elapsed
        
        print(f"Throughput: {throughput:.2f} tokens/second")
        print(f"Latency: {elapsed:.2f} seconds for {len(prompts)} prompts")
        
        return throughput, elapsed

# Usage
if __name__ == "__main__":
    engine = vLLMEngine("./models/mistral-7b-base")
    
    prompts = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to sort a list:",
        "What are the benefits of exercise?"
    ]
    
    results, elapsed = engine.generate(prompts)
    for r in results:
        print(f"\nPrompt: {r['prompt']}")
        print(f"Response: {r['generated_text']}")
```

### Step 7: Set Up llama.cpp for CPU Inference
```python
# src/engines/llamacpp_engine.py
from llama_cpp import Llama
import time

class LlamaCppEngine:
    """llama.cpp engine for CPU inference"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 8):
        print(f"Loading GGUF model with llama.cpp...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0  # CPU only
        )
        print("✓ llama.cpp model loaded")
    
    def generate(self, prompt: str, max_tokens: int = 100):
        """Generate response for single prompt"""
        start = time.time()
        
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        
        elapsed = time.time() - start
        
        return {
            'prompt': prompt,
            'generated_text': output['choices'][0]['text'],
            'tokens': output['usage']['completion_tokens'],
            'time': elapsed
        }
    
    def benchmark(self, prompts: list[str]):
        """Benchmark CPU inference"""
        results = []
        total_time = 0
        total_tokens = 0
        
        for prompt in prompts:
            result = self.generate(prompt)
            results.append(result)
            total_time += result['time']
            total_tokens += result['tokens']
        
        avg_tokens_per_sec = total_tokens / total_time
        print(f"CPU Throughput: {avg_tokens_per_sec:.2f} tokens/second")
        
        return results, avg_tokens_per_sec

# Convert model to GGUF first
# python convert.py ./models/mistral-7b-base --outtype q4_0 --outfile mistral-7b-q4.gguf
```

## Key Features to Implement

### 1. Quantization Techniques
- **4-bit NF4**: Normal Float 4-bit (bitsandbytes)
- **8-bit**: Linear quantization (bitsandbytes)
- **GPTQ**: Group-wise quantization with calibration
- **AWQ**: Activation-aware weight quantization
- **GGUF**: Quantized formats for llama.cpp (Q4_0, Q5_0, Q8_0)

### 2. Inference Engines
- **vLLM**: GPU-optimized with PagedAttention
- **llama.cpp**: CPU-optimized with GGUF format
- **ONNX Runtime**: Cross-platform optimization
- **Transformers**: Baseline reference

### 3. Performance Benchmarking
- **Latency**: Time to first token, per-token latency
- **Throughput**: Tokens per second, requests per second
- **Memory**: VRAM/RAM usage, peak memory
- **Batch Performance**: Throughput scaling with batch size

### 4. Quality Evaluation
- **Perplexity**: Model confidence on test set
- **Accuracy**: Task-specific metrics
- **Output Comparison**: Side-by-side quality check
- **Human Evaluation**: Subjective quality assessment

### 5. Optimization Analysis
- **Trade-off Charts**: Latency vs quality, memory vs speed
- **Cost Analysis**: Inference cost per 1M tokens
- **Recommendations**: Best configuration for use case
- **Deployment Guide**: Production setup instructions

## Comprehensive Benchmarking Suite

```python
# src/benchmark/comprehensive_benchmark.py
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import psutil
import torch

@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    engine: str
    quantization: str
    latency_first_token: float
    latency_per_token: float
    throughput: float
    memory_gb: float
    perplexity: float
    model_size_gb: float

class ComprehensiveBenchmark:
    """Run comprehensive benchmarks across all configurations"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.test_prompts = self._load_test_prompts()
    
    def run_all_benchmarks(self):
        """Run benchmarks for all configurations"""
        configs = [
            ("transformers", "fp16", "./models/mistral-7b-base"),
            ("transformers", "4bit", "./models/mistral-7b-base"),
            ("transformers", "8bit", "./models/mistral-7b-base"),
            ("gptq", "4bit", "./models/mistral-7b-gptq-4bit"),
            ("vllm", "fp16", "./models/mistral-7b-base"),
            ("llamacpp", "q4_0", "./models/mistral-7b-q4.gguf"),
            ("llamacpp", "q8_0", "./models/mistral-7b-q8.gguf"),
        ]
        
        for engine, quant, model_path in configs:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {engine} with {quant}")
            print(f"{'='*60}")
            
            result = self.benchmark_config(engine, quant, model_path)
            self.results.append(result)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def benchmark_config(self, engine: str, quant: str, model_path: str):
        """Benchmark single configuration"""
        # Load model
        model, tokenizer = self._load_model(engine, quant, model_path)
        
        # Measure memory
        memory_gb = self._measure_memory()
        model_size_gb = self._get_model_size(model)
        
        # Benchmark latency
        latency_first, latency_per = self._benchmark_latency(model, tokenizer)
        
        # Benchmark throughput
        throughput = self._benchmark_throughput(model, tokenizer)
        
        # Evaluate quality
        perplexity = self._evaluate_perplexity(model, tokenizer)
        
        return BenchmarkResult(
            engine=engine,
            quantization=quant,
            latency_first_token=latency_first,
            latency_per_token=latency_per,
            throughput=throughput,
            memory_gb=memory_gb,
            perplexity=perplexity,
            model_size_gb=model_size_gb
        )
    
    def _benchmark_latency(self, model, tokenizer):
        """Measure first token and per-token latency"""
        prompt = "Explain machine learning in simple terms:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Warm-up
        _ = model.generate(**inputs, max_new_tokens=10)
        
        # Measure first token
        import time
        start = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=1)
        first_token_latency = time.perf_counter() - start
        
        # Measure per-token (100 tokens)
        start = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=100)
        total_time = time.perf_counter() - start
        per_token_latency = total_time / 100
        
        return first_token_latency, per_token_latency
    
    def _benchmark_throughput(self, model, tokenizer):
        """Measure throughput (tokens/second)"""
        prompts = self.test_prompts[:10]
        
        total_tokens = 0
        total_time = 0
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start = time.perf_counter()
            outputs = model.generate(**inputs, max_new_tokens=100)
            elapsed = time.perf_counter() - start
            
            total_tokens += 100
            total_time += elapsed
        
        throughput = total_tokens / total_time
        return throughput
    
    def _evaluate_perplexity(self, model, tokenizer):
        """Calculate perplexity on test set"""
        from datasets import load_dataset
        import torch
        
        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        encodings = tokenizer("\n\n".join(test_data["text"][:100]), 
                            return_tensors="pt")
        
        max_length = 512
        stride = 256
        seq_len = encodings.input_ids.size(1)
        
        nlls = []
        prev_end_loc = 0
        
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            
            if end_loc == seq_len:
                break
        
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        df = pd.DataFrame([vars(r) for r in self.results])
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv("benchmark_results.csv", index=False)
        
        # Generate visualizations
        self._plot_results(df)
        
        # Generate recommendations
        self._generate_recommendations(df)
    
    def _plot_results(self, df):
        """Create visualization charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Throughput comparison
        ax = axes[0, 0]
        df.plot(x='engine', y='throughput', kind='bar', ax=ax, legend=False)
        ax.set_title('Throughput (tokens/second)')
        ax.set_ylabel('Tokens/second')
        
        # Memory usage
        ax = axes[0, 1]
        df.plot(x='engine', y='memory_gb', kind='bar', ax=ax, legend=False, color='orange')
        ax.set_title('Memory Usage (GB)')
        ax.set_ylabel('GB')
        
        # Latency comparison
        ax = axes[1, 0]
        df.plot(x='engine', y='latency_per_token', kind='bar', ax=ax, legend=False, color='green')
        ax.set_title('Per-Token Latency (seconds)')
        ax.set_ylabel('Seconds')
        
        # Quality (perplexity)
        ax = axes[1, 1]
        df.plot(x='engine', y='perplexity', kind='bar', ax=ax, legend=False, color='red')
        ax.set_title('Perplexity (lower is better)')
        ax.set_ylabel('Perplexity')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300)
        print("\n✓ Charts saved to benchmark_results.png")
    
    def _generate_recommendations(self, df):
        """Generate deployment recommendations"""
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        # Best for speed
        fastest = df.loc[df['throughput'].idxmax()]
        print(f"\n🚀 FASTEST: {fastest['engine']} ({fastest['quantization']})")
        print(f"   Throughput: {fastest['throughput']:.2f} tokens/sec")
        print(f"   Memory: {fastest['memory_gb']:.2f} GB")
        
        # Best for memory
        smallest = df.loc[df['memory_gb'].idxmin()]
        print(f"\n💾 MOST MEMORY EFFICIENT: {smallest['engine']} ({smallest['quantization']})")
        print(f"   Memory: {smallest['memory_gb']:.2f} GB")
        print(f"   Throughput: {smallest['throughput']:.2f} tokens/sec")
        
        # Best quality
        best_quality = df.loc[df['perplexity'].idxmin()]
        print(f"\n⭐ BEST QUALITY: {best_quality['engine']} ({best_quality['quantization']})")
        print(f"   Perplexity: {best_quality['perplexity']:.2f}")
        print(f"   Throughput: {best_quality['throughput']:.2f} tokens/sec")
        
        # Balanced recommendation
        df['score'] = (
            (df['throughput'] / df['throughput'].max()) * 0.4 +
            (1 - df['memory_gb'] / df['memory_gb'].max()) * 0.3 +
            (1 - df['perplexity'] / df['perplexity'].max()) * 0.3
        )
        balanced = df.loc[df['score'].idxmax()]
        print(f"\n⚖️  BALANCED: {balanced['engine']} ({balanced['quantization']})")
        print(f"   Throughput: {balanced['throughput']:.2f} tokens/sec")
        print(f"   Memory: {balanced['memory_gb']:.2f} GB")
        print(f"   Perplexity: {balanced['perplexity']:.2f}")

# Usage
if __name__ == "__main__":
    benchmark = ComprehensiveBenchmark()
    benchmark.run_all_benchmarks()
    benchmark.generate_report()
```

## Success Criteria

By the end of this project, you should have:

- [ ] Base model downloaded and tested (Mistral-7B or Llama-3-8B)
- [ ] 4-bit quantization working (bitsandbytes)
- [ ] 8-bit quantization working (bitsandbytes)
- [ ] GPTQ quantization implemented and tested
- [ ] AWQ quantization implemented (optional)
- [ ] GGUF conversion for llama.cpp
- [ ] vLLM engine set up and benchmarked
- [ ] llama.cpp engine set up and benchmarked
- [ ] ONNX conversion and benchmarking (optional)
- [ ] Comprehensive benchmark suite running
- [ ] Performance metrics collected (latency, throughput, memory)
- [ ] Quality evaluation (perplexity, accuracy)
- [ ] Visualization charts generated
- [ ] Recommendations report created
- [ ] Production deployment guide written
- [ ] GitHub repository with all code and results

## Learning Outcomes

After completing this project, you'll be able to:

- Understand different quantization techniques and their trade-offs
- Implement 4-bit, 8-bit, GPTQ, and AWQ quantization
- Use multiple inference engines (vLLM, llama.cpp, ONNX)
- Benchmark LLM performance accurately
- Measure and optimize memory usage
- Evaluate model quality after quantization
- Make data-driven deployment decisions
- Deploy optimized models to production
- Calculate inference costs and ROI
- Explain PagedAttention and KV cache optimization

## Expected Performance Improvements

Based on typical results with Mistral-7B:

**Model Size Reduction**:
- FP16 (baseline): 14 GB
- 8-bit: 7 GB (2x smaller)
- 4-bit: 3.5 GB (4x smaller)
- GPTQ 4-bit: 3.2 GB (4.4x smaller)

**Inference Speed** (on A100 GPU):
- Transformers FP16: 25 tokens/sec
- Transformers 4-bit: 35 tokens/sec (1.4x faster)
- vLLM FP16: 120 tokens/sec (4.8x faster)
- vLLM + quantization: 180 tokens/sec (7.2x faster)

**CPU Performance** (llama.cpp on 16-core CPU):
- Q4_0: 15-20 tokens/sec
- Q8_0: 10-15 tokens/sec

**Quality Impact**:
- FP16: Perplexity 5.2 (baseline)
- 8-bit: Perplexity 5.3 (+2% degradation)
- 4-bit: Perplexity 5.6 (+8% degradation)
- GPTQ 4-bit: Perplexity 5.5 (+6% degradation)

## Project Structure

```
project-f-inference-optimization/
├── src/
│   ├── quantization/
│   │   ├── bitsandbytes_quant.py
│   │   ├── gptq_quant.py
│   │   ├── awq_quant.py
│   │   └── gguf_convert.py
│   ├── engines/
│   │   ├── vllm_engine.py
│   │   ├── llamacpp_engine.py
│   │   ├── onnx_engine.py
│   │   └── transformers_engine.py
│   ├── benchmark/
│   │   ├── comprehensive_benchmark.py
│   │   ├── latency_benchmark.py
│   │   ├── throughput_benchmark.py
│   │   └── memory_benchmark.py
│   ├── evaluation/
│   │   ├── perplexity.py
│   │   ├── accuracy.py
│   │   └── quality_comparison.py
│   └── utils/
│       ├── model_loader.py
│       ├── data_loader.py
│       └── visualization.py
├── models/
│   ├── mistral-7b-base/
│   ├── mistral-7b-gptq-4bit/
│   ├── mistral-7b-q4.gguf
│   └── mistral-7b-q8.gguf
├── notebooks/
│   ├── 01_quantization_comparison.ipynb
│   ├── 02_engine_benchmarks.ipynb
│   ├── 03_quality_evaluation.ipynb
│   └── 04_deployment_guide.ipynb
├── results/
│   ├── benchmark_results.csv
│   ├── benchmark_results.png
│   └── recommendations.md
├── docker/
│   ├── Dockerfile.vllm
│   ├── Dockerfile.llamacpp
│   └── docker-compose.yml
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges & Solutions

### Challenge 1: CUDA Out of Memory
**Problem**: GPU runs out of memory during quantization or inference
**Solution**: Use smaller batch sizes, enable CPU offloading, or use more aggressive quantization
```python
# Enable CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,
    device_map="auto",  # Automatically offload to CPU
    max_memory={0: "6GB", "cpu": "30GB"}
)
```

### Challenge 2: Slow CPU Inference
**Problem**: llama.cpp inference is too slow on CPU
**Solution**: Use smaller quantization (Q4_0), increase threads, or use Metal/CUDA acceleration
```python
# Optimize llama.cpp for CPU
llm = Llama(
    model_path="model.gguf",
    n_threads=os.cpu_count(),  # Use all CPU cores
    n_batch=512,  # Larger batch for throughput
    use_mlock=True  # Lock model in RAM
)
```

### Challenge 3: Quality Degradation
**Problem**: Quantized model produces poor quality outputs
**Solution**: Use less aggressive quantization, calibrate with representative data, or try AWQ
```python
# Use 8-bit instead of 4-bit for better quality
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,  # Better quality than 4-bit
    device_map="auto"
)
```

### Challenge 4: Inconsistent Benchmarks
**Problem**: Benchmark results vary significantly between runs
**Solution**: Run warm-up iterations, use median of multiple runs, clear cache between tests
```python
def benchmark_with_warmup(model, prompt, iterations=5):
    # Warm-up
    for _ in range(3):
        model.generate(prompt, max_new_tokens=10)
    
    # Actual benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.generate(prompt, max_new_tokens=100)
        times.append(time.perf_counter() - start)
    
    return statistics.median(times)
```

## Advanced Optimization Techniques

### 1. Flash Attention
```python
# Enable Flash Attention 2 for faster inference
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```

### 2. Speculative Decoding
```python
# Use smaller draft model for faster generation
from transformers import AutoModelForCausalLM

target_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
draft_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-3B")

# Speculative decoding can be 2-3x faster
```

### 3. Continuous Batching (vLLM)
```python
# vLLM automatically does continuous batching
# Just send requests as they come
from vllm import LLM

llm = LLM(model="mistralai/Mistral-7B-v0.1")

# Requests are batched dynamically
outputs = llm.generate(prompts, sampling_params)
```

### 4. KV Cache Quantization
```python
# Quantize KV cache to save memory
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    cache_implementation="quantized"  # Quantize KV cache
)
```

## Troubleshooting

### Installation Issues

**Issue**: bitsandbytes installation fails
```bash
# Solution: Install from source or use pre-built wheels
pip install bitsandbytes --prefer-binary
# Or for CUDA 11.8
pip install bitsandbytes-cuda118
```

**Issue**: vLLM installation fails
```bash
# Solution: Install with specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

**Issue**: llama-cpp-python doesn't use GPU
```bash
# Solution: Install with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Runtime Issues

**Issue**: "Quantization not supported for this model"
```python
# Solution: Check model architecture compatibility
# Not all models support all quantization methods
# Try different quantization library or method
```

**Issue**: vLLM "CUDA out of memory"
```python
# Solution: Reduce gpu_memory_utilization
llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.7,  # Reduce from 0.9
    max_model_len=1024  # Reduce context length
)
```

**Issue**: Perplexity calculation fails
```python
# Solution: Use smaller test set or batch processing
# Process in chunks to avoid OOM
for i in range(0, len(test_data), batch_size):
    batch = test_data[i:i+batch_size]
    perplexity = calculate_perplexity(model, batch)
```

## Production Deployment

### Docker Deployment (vLLM)
```dockerfile
# Dockerfile.vllm
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN pip install vllm

COPY models/ /models/

CMD ["python", "-m", "vllm.entrypoints.api_server", \
     "--model", "/models/mistral-7b-base", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

### API Server
```python
# serve.py
from fastapi import FastAPI
from vllm import LLM, SamplingParams
from pydantic import BaseModel

app = FastAPI()
llm = LLM(model="./models/mistral-7b-base")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
async def generate(request: GenerateRequest):
    sampling_params = SamplingParams(max_tokens=request.max_tokens)
    outputs = llm.generate([request.prompt], sampling_params)
    return {"generated_text": outputs[0].outputs[0].text}

# Run: uvicorn serve:app --host 0.0.0.0 --port 8000
```

### Monitoring
```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram

request_count = Counter('inference_requests_total', 'Total inference requests')
latency = Histogram('inference_latency_seconds', 'Inference latency')

@app.post("/generate")
@latency.time()
async def generate(request: GenerateRequest):
    request_count.inc()
    # ... inference code
```

## Cost Analysis

### Inference Cost Comparison (per 1M tokens)

**Cloud GPU (A100)**:
- FP16 baseline: $2.50
- 4-bit quantized: $1.20 (52% savings)
- vLLM optimized: $0.60 (76% savings)

**Cloud CPU (32-core)**:
- llama.cpp Q4_0: $0.80
- llama.cpp Q8_0: $1.20

**Recommendations**:
- High volume: Use vLLM with quantization on GPU
- Low volume: Use llama.cpp on CPU
- Balanced: 4-bit quantization with vLLM

## Resources

### Documentation
- [vLLM Documentation](https://docs.vllm.ai/) - Fast inference engine
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU inference
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - Quantization library
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) - GPTQ quantization
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) - AWQ quantization

### Tutorials
- [Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization) - HuggingFace
- [vLLM Tutorial](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) - Quick start
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - Specification

### Papers
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)

## Questions?

If you get stuck:
1. Review the `tech-spec.md` for detailed architecture
2. Check engine-specific documentation (vLLM, llama.cpp)
3. Search HuggingFace forums for quantization issues
4. Review the 100 Days bootcamp materials on optimization
5. Start with smaller models (3B) before 7B models
6. Test on CPU first if GPU issues occur

## Related Projects

After completing this project, consider:
- **Project G**: Prompt Engineering - Optimize prompts for quantized models
- **Project H**: Embeddings Search - Optimize embedding model inference
- **Project B**: LLM Fine-Tuning - Fine-tune then optimize
- Build a production inference service with monitoring and auto-scaling
