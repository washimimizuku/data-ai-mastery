# Design Document: LLM Fine-Tuning & Inference Optimization

## Overview

This system provides an end-to-end pipeline for fine-tuning large language models using parameter-efficient techniques (LoRA/QLoRA) and serving them with optimized inference. The architecture separates concerns into three main components: a fine-tuning pipeline for model training, an evaluation framework for quality and performance measurement, and an inference service for production deployment.

The system targets 7B+ parameter models and achieves sub-2-second inference latency while keeping training costs under $50. It uses established libraries (Hugging Face Transformers, PEFT, vLLM) rather than implementing low-level optimizations from scratch.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Fine-Tuning Pipeline                      │
│                                                                   │
│  Dataset → Preprocessing → LoRA/QLoRA → Training → Checkpoints  │
│              │                            │                       │
│              └────────────────────────────┴──────────────────────┤
│                                                                   │
│                      Experiment Tracking (MLflow)                │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Evaluation Framework                        │
│                                                                   │
│  Model Loader → Perplexity → Task Metrics → Benchmarking        │
│                                                                   │
│  Cost Analysis → Performance Reports                             │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Inference Service                          │
│                                                                   │
│  FastAPI ──→ vLLM Engine ──→ Model (Quantized + KV Cache)      │
│     │                                                             │
│     ├──→ /generate (streaming)                                   │
│     └──→ /generate/batch                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Dataset Preparation Module

**Responsibilities:**
- Load datasets from various formats (JSONL, CSV, Hugging Face datasets)
- Validate data structure and required fields
- Format instruction-response pairs into training text
- Split data into train/validation sets

**Key Interfaces:**

```python
class DatasetPreprocessor:
    def load_dataset(self, file_path: str, format: str) -> Dataset:
        """Load dataset from file"""
        
    def validate_example(self, example: dict) -> bool:
        """Check if example has required fields"""
        
    def format_instruction(self, example: dict) -> dict:
        """Format example into instruction-response text"""
        
    def prepare_dataset(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Prepare and split dataset into train/val"""
```

### 2. LoRA Fine-Tuning Module

**Responsibilities:**
- Configure LoRA parameters (rank, alpha, target modules)
- Apply LoRA adapters to base model
- Handle QLoRA quantization
- Manage training loop with gradient accumulation
- Save and load adapter weights

**Key Interfaces:**

```python
class LoRATrainer:
    def __init__(self, base_model_name: str, lora_config: LoraConfig):
        """Initialize with base model and LoRA configuration"""
        
    def prepare_model(self, use_quantization: bool = False) -> PeftModel:
        """Load base model and apply LoRA adapters"""
        
    def train(self, train_dataset: Dataset, eval_dataset: Dataset, 
              training_args: TrainingArguments) -> None:
        """Execute training loop"""
        
    def save_adapter(self, output_path: str) -> None:
        """Save LoRA adapter weights"""
        
    def load_adapter(self, adapter_path: str) -> PeftModel:
        """Load trained adapter"""
```

### 3. Experiment Tracking Module

**Responsibilities:**
- Log hyperparameters and configuration
- Record training metrics (loss, learning rate)
- Track model checkpoints
- Store evaluation results

**Key Interfaces:**

```python
class ExperimentTracker:
    def start_run(self, experiment_name: str, params: dict) -> str:
        """Start new experiment run, return run_id"""
        
    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log metrics at specific training step"""
        
    def log_model(self, model_path: str, run_id: str) -> None:
        """Associate model checkpoint with run"""
        
    def end_run(self, final_metrics: dict) -> None:
        """Complete run and log final results"""
```

### 4. Evaluation Module

**Responsibilities:**
- Calculate perplexity on test data
- Measure task-specific accuracy
- Compare base vs fine-tuned models
- Generate evaluation reports

**Key Interfaces:**

```python
class ModelEvaluator:
    def calculate_perplexity(self, model: PreTrainedModel, 
                            tokenizer: PreTrainedTokenizer, 
                            text: str) -> float:
        """Compute perplexity on given text"""
        
    def evaluate_task(self, model: PreTrainedModel, 
                     test_cases: list[dict]) -> dict:
        """Run task-specific evaluation"""
        
    def compare_models(self, base_model: PreTrainedModel, 
                      finetuned_model: PreTrainedModel, 
                      test_data: Dataset) -> dict:
        """Compare two models on same test data"""
        
    def generate_report(self, results: dict) -> str:
        """Create formatted evaluation report"""
```

### 5. Performance Benchmarking Module

**Responsibilities:**
- Measure inference latency (p50, p95, p99)
- Calculate throughput (requests/second)
- Monitor GPU utilization and memory
- Generate performance reports

**Key Interfaces:**

```python
class PerformanceBenchmark:
    def benchmark_latency(self, model: LLM, prompts: list[str], 
                         num_runs: int = 100) -> dict:
        """Measure latency percentiles"""
        
    def benchmark_throughput(self, model: LLM, prompts: list[str], 
                            duration: int = 60) -> float:
        """Measure requests per second"""
        
    def monitor_resources(self) -> dict:
        """Get GPU utilization and memory usage"""
        
    def generate_benchmark_report(self, results: dict) -> str:
        """Create formatted benchmark report"""
```

### 6. Cost Analysis Module

**Responsibilities:**
- Calculate training costs based on instance type and duration
- Calculate inference costs based on throughput and pricing
- Compare costs across configurations
- Generate cost reports

**Key Interfaces:**

```python
class CostAnalyzer:
    def calculate_training_cost(self, instance_type: str, 
                               hours: float) -> float:
        """Compute training cost"""
        
    def calculate_inference_cost(self, instance_type: str, 
                                throughput: float) -> float:
        """Compute cost per 1K requests"""
        
    def compare_configurations(self, configs: list[dict]) -> dict:
        """Compare costs across different setups"""
        
    def generate_cost_report(self, analysis: dict) -> str:
        """Create formatted cost report"""
```

### 7. vLLM Inference Engine

**Responsibilities:**
- Load models with quantization
- Configure KV cache and memory settings
- Handle single and batch inference
- Support streaming generation

**Key Interfaces:**

```python
class VLLMEngine:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.9):
        """Initialize vLLM with model and configuration"""
        
    def generate(self, prompts: list[str], 
                sampling_params: SamplingParams) -> list[str]:
        """Generate completions for batch of prompts"""
        
    async def generate_stream(self, prompt: str, 
                             sampling_params: SamplingParams) -> AsyncIterator[str]:
        """Stream tokens as they are generated"""
```

### 8. FastAPI Inference Service

**Responsibilities:**
- Expose HTTP endpoints for inference
- Handle streaming responses with SSE
- Support batch inference
- Manage error responses

**Key Interfaces:**

```python
# FastAPI endpoints
@app.post("/generate")
async def generate(request: GenerateRequest) -> StreamingResponse:
    """Stream generated tokens"""

@app.post("/generate/batch")
async def generate_batch(request: BatchGenerateRequest) -> BatchGenerateResponse:
    """Generate completions for multiple prompts"""

@app.get("/health")
async def health() -> dict:
    """Health check endpoint"""
```

## Data Models

### Training Configuration

```python
@dataclass
class LoRAConfig:
    r: int = 16  # Rank
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    fp16: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    use_quantization: bool = False
```

### Inference Configuration

```python
@dataclass
class InferenceConfig:
    model_path: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
```

### API Request/Response Models

```python
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = True

class BatchGenerateRequest(BaseModel):
    prompts: list[str]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class BatchGenerateResponse(BaseModel):
    results: list[str]
    latency_ms: float
```

### Evaluation Results

```python
@dataclass
class EvaluationResults:
    perplexity: float
    task_accuracy: float
    num_examples: int
    avg_generation_length: float

@dataclass
class BenchmarkResults:
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_rps: float
    gpu_utilization_percent: float
    memory_used_gb: float

@dataclass
class CostAnalysis:
    training_cost_usd: float
    inference_cost_per_1k_requests: float
    instance_type: str
    training_hours: float
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Dataset parsing extracts all pairs

*For any* valid dataset file, parsing should extract instruction-response pairs that match the structure defined in the file.
**Validates: Requirements 1.1**

### Property 2: Formatted output contains both sections

*For any* instruction-response pair, the formatted text should contain both the instruction section and the response section.
**Validates: Requirements 1.2**

### Property 3: Validation identifies required fields

*For any* dataset example, validation should return true if and only if all required fields are present.
**Validates: Requirements 1.3**

### Property 4: Invalid data errors include identifiers

*For any* invalid dataset example, the error message should contain the example's identifier.
**Validates: Requirements 1.4**

### Property 5: LoRA configuration sets all parameters

*For any* valid LoRA configuration values, initialization should result in a config object with rank, alpha, target modules, and dropout all set to the specified values.
**Validates: Requirements 2.1**

### Property 6: LoRA creates adapters for target modules

*For any* set of target modules, applying LoRA should create adapter layers for exactly those modules and no others.
**Validates: Requirements 2.2**

### Property 7: LoRA training preserves base weights (Invariant)

*For any* training run with LoRA, the base model weights before training should equal the base model weights after training, while adapter weights should differ.
**Validates: Requirements 2.3**

### Property 8: LoRA saves only adapter weights

*For any* completed LoRA training run, the saved files should contain adapter weights but not base model weights.
**Validates: Requirements 2.4**

### Property 9: QLoRA quantizes to 4-bit

*For any* model loaded with QLoRA, the base model weights should be stored in 4-bit precision.
**Validates: Requirements 3.1**

### Property 10: Gradient accumulation delays updates

*For any* gradient accumulation setting N, weight updates should occur only after every N training steps.
**Validates: Requirements 3.3**

### Property 11: QLoRA saves config and weights

*For any* saved QLoRA model, the saved files should contain both quantization configuration and adapter weights.
**Validates: Requirements 3.4**

### Property 12: Training logs hyperparameters

*For any* training run with hyperparameters H, the experiment tracker should contain all parameters from H after the run starts.
**Validates: Requirements 4.1**

### Property 13: Metrics logged at intervals

*For any* training run with logging interval N, metrics should be recorded at steps that are multiples of N.
**Validates: Requirements 4.2**

### Property 14: Checkpoints linked to runs

*For any* saved checkpoint, the checkpoint should be associated with the correct experiment run ID.
**Validates: Requirements 4.3**

### Property 15: Final metrics logged

*For any* completed training run, the experiment tracker should contain final evaluation metrics.
**Validates: Requirements 4.4**

### Property 16: Perplexity calculation correctness

*For any* sequence of tokens with known log-likelihoods, the calculated perplexity should equal exp(mean(-log_likelihood)).
**Validates: Requirements 5.1**

### Property 17: Model comparison uses identical data

*For any* model comparison, both models should receive exactly the same test inputs.
**Validates: Requirements 5.2**

### Property 18: All test cases evaluated

*For any* set of test cases, the evaluation should validate outputs for every test case in the set.
**Validates: Requirements 5.3**

### Property 19: Evaluation report completeness

*For any* evaluation run, the generated report should contain all computed metrics (perplexity, task accuracy, etc.).
**Validates: Requirements 5.4**

### Property 20: vLLM configuration applied

*For any* vLLM initialization with specified GPU memory utilization and tensor parallelism settings, the loaded model should use those exact settings.
**Validates: Requirements 6.1**

### Property 21: Quantization reduces precision

*For any* model with quantization applied at N bits, the model weights should be stored in N-bit precision.
**Validates: Requirements 6.2**

### Property 22: Batch processing uses single forward pass

*For any* batch request with multiple prompts, the inference engine should execute exactly one forward pass for all prompts.
**Validates: Requirements 6.4**

### Property 23: Streaming emits tokens incrementally

*For any* streaming generation request, tokens should arrive in multiple separate events rather than all at once.
**Validates: Requirements 7.2**

### Property 24: Batch results match prompts

*For any* batch request with N prompts, the response should contain exactly N results.
**Validates: Requirements 7.3**

### Property 25: Errors return error status

*For any* request that triggers an error, the HTTP response should have an error status code (4xx or 5xx) and include a descriptive message.
**Validates: Requirements 7.4**

### Property 26: Streaming uses SSE format

*For any* streaming response, each token event should follow Server-Sent Events format (prefixed with "data: " and followed by double newline).
**Validates: Requirements 8.2**

### Property 27: Streaming sends completion signal

*For any* successfully completed stream, the final event should be a completion signal.
**Validates: Requirements 8.3**

### Property 28: Streaming errors close stream

*For any* error during streaming, an error event should be sent and the stream should close.
**Validates: Requirements 8.4**

### Property 29: Benchmark measures all percentiles

*For any* benchmark run, the results should include p50, p95, and p99 latency measurements.
**Validates: Requirements 9.1**

### Property 30: Throughput calculation correctness

*For any* benchmark with known request count and duration, throughput should equal requests divided by duration.
**Validates: Requirements 9.2**

### Property 31: Resource monitoring completeness

*For any* resource monitoring call, the results should include both GPU utilization and memory usage.
**Validates: Requirements 9.3**

### Property 32: Benchmark report completeness

*For any* completed benchmark, the performance report should contain all measurements (latency percentiles, throughput, resource usage).
**Validates: Requirements 9.4**

### Property 33: Training cost calculation correctness

*For any* training run with known instance hourly rate and duration, training cost should equal rate multiplied by duration.
**Validates: Requirements 10.1**

### Property 34: Inference cost calculation correctness

*For any* inference configuration with known instance rate and throughput, cost per request should be correctly computed from rate divided by throughput.
**Validates: Requirements 10.2**

### Property 35: Configuration comparison shows all configs

*For any* set of configurations, the comparison output should include cost metrics for every configuration in the set.
**Validates: Requirements 10.3**

### Property 36: Cost report completeness

*For any* cost analysis, the generated report should include both training costs and inference costs with resource type breakdown.
**Validates: Requirements 10.4**

## Error Handling

### Dataset Processing Errors
- **Invalid file format**: Return clear error message indicating expected format
- **Missing required fields**: Report which fields are missing and in which example
- **Malformed data**: Include line number and parsing error details

### Training Errors
- **Out of memory**: Suggest reducing batch size or using gradient accumulation
- **Model loading failure**: Verify model name and check network connectivity
- **Checkpoint corruption**: Fall back to previous checkpoint if available
- **Divergent loss**: Log warning and optionally stop training

### Inference Errors
- **Model not loaded**: Return 503 Service Unavailable with retry-after header
- **Invalid request parameters**: Return 400 Bad Request with parameter validation details
- **Generation timeout**: Return 504 Gateway Timeout with partial results if available
- **GPU out of memory**: Return 503 and suggest reducing batch size or max tokens

### Evaluation Errors
- **Test data unavailable**: Fail fast with clear error message
- **Metric calculation failure**: Log error and continue with other metrics
- **Model comparison mismatch**: Verify both models are compatible

## Testing Strategy

This system requires both unit testing and property-based testing to ensure correctness:

### Unit Testing Approach

Unit tests will verify specific examples and integration points:

- **Dataset preprocessing**: Test parsing of sample JSONL files with known structure
- **LoRA configuration**: Test initialization with specific parameter values
- **API endpoints**: Test HTTP request/response handling with example payloads
- **Error conditions**: Test specific error scenarios (missing files, invalid configs)
- **Integration points**: Test that components work together (e.g., trainer + experiment tracker)

Unit tests provide concrete examples that demonstrate correct behavior and catch specific bugs.

### Property-Based Testing Approach

Property-based tests will verify universal properties across many randomly generated inputs:

**Testing Framework**: We will use **Hypothesis** for Python, which is the standard property-based testing library for Python projects.

**Configuration**: Each property-based test will run a minimum of 100 iterations to ensure thorough coverage of the input space.

**Test Tagging**: Each property-based test will include a comment explicitly referencing the correctness property from this design document using the format: `# Feature: llm-fine-tuning-inference, Property {number}: {property_text}`

**Property Implementation**: Each correctness property listed above will be implemented as a single property-based test.

**Key Properties to Test**:

1. **Dataset operations** (Properties 1-4): Generate random valid/invalid datasets and verify parsing, formatting, and validation
2. **LoRA invariants** (Properties 5-8): Generate random configurations and verify adapter creation and weight preservation
3. **Quantization** (Properties 9, 11, 21): Verify bit precision and saved model structure
4. **Training mechanics** (Properties 10, 12-15): Verify gradient accumulation timing and logging behavior
5. **Evaluation calculations** (Properties 16-19): Generate random inputs and verify mathematical correctness
6. **Inference behavior** (Properties 22-28): Verify batching, streaming, and error handling
7. **Benchmarking** (Properties 29-32): Verify completeness of measurements and calculation correctness
8. **Cost calculations** (Properties 33-36): Generate random rates/durations and verify formula correctness

**Smart Generators**: We will write intelligent generators that constrain inputs to valid ranges:
- Dataset generators will create structurally valid JSONL with random content
- Configuration generators will use valid parameter ranges (e.g., rank > 0, dropout in [0,1])
- Prompt generators will create strings within token limits
- Cost generators will use realistic instance types and pricing

**Complementary Coverage**: Unit tests catch specific bugs and verify examples, while property tests verify general correctness across the input space. Together they provide comprehensive validation.

## Performance Considerations

### Training Optimization
- Use gradient accumulation to simulate larger batch sizes
- Enable mixed precision (fp16) to reduce memory usage
- Use gradient checkpointing for very large models
- Monitor GPU utilization and adjust batch size accordingly

### Inference Optimization
- vLLM provides PagedAttention for efficient KV cache management
- Continuous batching to maximize GPU utilization
- Quantization (4-bit, 8-bit) to reduce memory footprint
- Tensor parallelism for models that don't fit on single GPU

### Expected Performance
- Training: 4-6 hours for 7B model on single A100 GPU
- Inference latency: 500-800ms for 512 token generation
- Throughput: 1-2 requests/second on single GPU
- Memory: ~16GB for 7B model with 4-bit quantization

## Deployment Considerations

### Infrastructure Options
1. **Modal**: Serverless GPU deployment with auto-scaling
2. **AWS SageMaker**: Managed training and inference endpoints
3. **RunPod**: Cost-effective GPU rental for development
4. **Self-hosted**: Docker containers on GPU instances

### Scaling Strategy
- Start with single GPU for development and testing
- Use tensor parallelism for larger models (70B+)
- Implement request queuing for burst traffic
- Add horizontal scaling with load balancer for production

### Monitoring
- Track inference latency and throughput
- Monitor GPU utilization and memory
- Log error rates and types
- Alert on performance degradation

### Cost Optimization
- Use spot instances for training when possible
- Implement auto-scaling to match demand
- Use quantization to reduce inference costs
- Cache frequent requests if applicable
