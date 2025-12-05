# Tech Spec: LLM Inference Optimization

## Engines to Benchmark
1. **llama.cpp** - CPU optimized, GGUF format
2. **vLLM** - GPU optimized, PagedAttention
3. **ONNX Runtime** - Cross-platform optimization
4. **Transformers** - Baseline

## Quantization
- 4-bit (GPTQ, GGUF Q4)
- 8-bit (bitsandbytes)
- FP16, FP32 (baseline)

## Metrics
- Tokens/second
- Latency (first token, per token)
- Memory usage (GB)
- Quality (perplexity)

## Optional Rust
Custom inference kernels for specific operations, compare with Python implementations.

## Implementation
```python
# llama.cpp
from llama_cpp import Llama
model = Llama(model_path="model.gguf")

# vLLM
from vllm import LLM
model = LLM(model="meta-llama/Llama-3-8B")

# ONNX
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
```
