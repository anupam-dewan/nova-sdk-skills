# Nova Forge SDK - Capabilities Matrix

This document provides a comprehensive overview of the Nova Forge SDK capabilities, organized by module and functionality.

---

## Core Modules

### 1. Dataset Module
**Purpose**: Data loading, transformation, and validation

| Capability | Description | Supported Formats |
|-----------|-------------|-------------------|
| **Load Data** | Load datasets from local or S3 | JSONL, JSON, CSV |
| **Transform** | Convert to training format | Converse, CPT, DPO, RFT |
| **Validate** | Check format and content | All methods |
| **Split** | Train/val/test splitting | All formats |
| **Save** | Upload to S3 or save locally | JSONL, JSON, CSV |
| **Preview** | Display sample rows | All formats |

**Supported Loaders:**
- `JSONLDatasetLoader` (recommended)
- `JSONDatasetLoader`
- `CSVDatasetLoader`

---

### 2. Model Module (NovaModelCustomizer)
**Purpose**: Main orchestration and training management

| Capability | Method | Description |
|-----------|--------|-------------|
| **Training** | `train()` | Launch training jobs |
| **Evaluation** | `evaluate()` | Run model evaluation |
| **Deployment** | `deploy()` | Deploy to Bedrock/SageMaker |
| **Batch Inference** | `batch_inference()` | Process batches |
| **Invoke Inference** | `invoke_inference()` | Single inference requests |
| **Get Logs** | `get_logs()` | Retrieve CloudWatch logs |
| **Data Mixing Config** | `get_data_mixing_config()` | View data mixing settings |
| **Data Mixing Config** | `set_data_mixing_config()` | Configure data mixing |

---

### 3. Runtime Manager Module
**Purpose**: Infrastructure management

#### SMTJRuntimeManager (SageMaker Training Jobs)
| Capability | Description |
|-----------|-------------|
| **Execute** | Start training/eval jobs |
| **Cleanup** | Stop running jobs |
| **Instance Config** | Configure GPU instances |
| **Auto-scaling** | Automatic resource management |

**Supported Instance Types:**
- ml.p4d.24xlarge
- ml.p5.48xlarge
- ml.trn1.32xlarge
- See `docs/instance_type_spec.md` for full list

#### SMHPRuntimeManager (SageMaker HyperPod)
| Capability | Description |
|-----------|-------------|
| **Execute** | Start jobs on HyperPod cluster |
| **Cleanup** | Stop cluster jobs |
| **Cluster Config** | Configure cluster settings |
| **Namespace** | Kubernetes namespace management |

---

### 4. Monitor Module
**Purpose**: Job monitoring and experiment tracking

#### CloudWatchLogMonitor
| Capability | Description |
|-----------|-------------|
| **Show Logs** | Display job logs |
| **Get Logs** | Retrieve logs as list |
| **From Job Result** | Create from training result |
| **From Job ID** | Create from job identifier |

#### MLflowMonitor
| Capability | Description |
|-----------|-------------|
| **Track Experiments** | Log metrics and parameters |
| **Presigned URL** | Generate MLflow UI access |
| **Experiment Management** | Organize runs and experiments |
| **Run Tracking** | Track individual training runs |

---

### 5. RFT Multiturn Module
**Purpose**: Reinforcement fine-tuning infrastructure for multi-turn tasks

| Capability | Description | Platforms |
|-----------|-------------|-----------|
| **Setup** | Deploy SAM stack | LOCAL, EC2, ECS |
| **Start Training Env** | Start training environment | All platforms |
| **Start Eval Env** | Start evaluation environment | All platforms |
| **Get Logs** | Retrieve environment logs | All platforms |
| **Kill Task** | Stop running task | All platforms |
| **Cleanup** | Clean up resources | All platforms |
| **Check Queues** | Monitor message counts | All platforms |
| **Flush Queues** | Purge queue messages | All platforms |

**Supported Platforms:**
- `LOCAL` - Local development
- `EC2` - Amazon EC2 instances
- `ECS` - Amazon ECS Fargate

**Built-in Environments:**
- `VFEnvId.WORDLE` - Wordle game
- `VFEnvId.TERMINAL_BENCH` - Terminal benchmark
- Custom environments via `CustomEnvironment` class

---

## Training Methods Support Matrix

| Method | SMTJ | SMHP | Models | Use Case | Difficulty |
|--------|------|------|--------|----------|------------|
| **SFT_LORA** | ✅ | ✅ | All | Supervised fine-tuning | ⭐⭐ Easy |
| **SFT_FULL** | ✅ | ✅ | All | Full-rank SFT | ⭐⭐ Easy |
| **CPT** | ❌ | ✅ | All | Continued pre-training | ⭐⭐⭐ Medium |
| **DPO_LORA** | ✅ | ✅ | Nova 1.0 | Preference optimization | ⭐⭐ Easy |
| **DPO_FULL** | ✅ | ✅ | Nova 1.0 | Full-rank DPO | ⭐⭐ Easy |
| **RFT_LORA** | ✅ | ✅ | Nova 2.0 | Reinforcement FT | ⭐⭐⭐ Medium |
| **RFT_FULL** | ✅ | ✅ | Nova 2.0 | Full RFT | ⭐⭐⭐ Medium |
| **RFT_MULTITURN_LORA** | ❌ | ✅ | Nova 2.0 | Multi-turn RFT | ⭐⭐⭐⭐ Hard |
| **RFT_MULTITURN_FULL** | ❌ | ✅ | Nova 2.0 | Full multi-turn RFT | ⭐⭐⭐⭐ Hard |
| **EVALUATION** | ✅ | ✅ | All | Model evaluation | ⭐ Very Easy |

---

## Model Support

| Model | Model ID | Context Length | Training Methods | Best For |
|-------|----------|----------------|------------------|----------|
| **NOVA_MICRO** | amazon.nova-micro-v1:0:128k | 128k | All | Fast iteration, development |
| **NOVA_LITE** | amazon.nova-lite-v1:0:300k | 300k | All | General purpose (V1) |
| **NOVA_LITE_2** | amazon.nova-2-lite-v1:0:256k | 256k | All | **Recommended** general purpose |
| **NOVA_PRO** | amazon.nova-pro-v1:0:300k | 300k | All | High quality, complex tasks |

---

## Data Format Requirements

### SFT (Supervised Fine-Tuning)
**Input Format:**
```json
{
    "question": "What is AWS?",
    "answer": "AWS is Amazon's cloud computing platform."
}
```

**After Transformation (Converse):**
```json
{
    "messages": [
        {"role": "user", "content": "What is AWS?"},
        {"role": "assistant", "content": "AWS is Amazon's cloud computing platform."}
    ]
}
```

### CPT (Continued Pre-Training)
**Required Format:**
```json
{
    "text": "Domain-specific text for pre-training..."
}
```

### DPO (Direct Preference Optimization)
**Required Format:**
```json
{
    "messages": [
        {"role": "user", "content": [{"text": "Question here"}]},
        {
            "role": "assistant",
            "candidates": [
                {
                    "content": [{"text": "Preferred response"}],
                    "preferenceLabel": "preferred"
                },
                {
                    "content": [{"text": "Non-preferred response"}],
                    "preferenceLabel": "non-preferred"
                }
            ]
        }
    ]
}
```

### RFT (Reinforcement Fine-Tuning)
**Required Format:**
```json
{
    "id": "sample_1",
    "messages": [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"}
    ],
    "reference_answer": "Ground truth answer (optional)"
}
```

---

## Evaluation Capabilities

### Public Benchmarks

| Benchmark | Task ID | Description | Measures |
|-----------|---------|-------------|----------|
| **MMLU** | `EvaluationTask.MMLU` | Multitask Language Understanding | General knowledge (57 subjects) |
| **HellaSwag** | `EvaluationTask.HELLASWAG` | Commonsense reasoning | Situation understanding |
| **ARC-Challenge** | `EvaluationTask.ARC_CHALLENGE` | AI2 Reasoning Challenge | Science questions |
| **ARC-Easy** | `EvaluationTask.ARC_EASY` | AI2 Reasoning (easier) | Basic science |
| **TruthfulQA** | `EvaluationTask.TRUTHFULQA` | Truthfulness | Factual accuracy |
| **GSM8K** | `EvaluationTask.GSM8K` | Grade School Math | Math reasoning |
| **HumanEval** | `EvaluationTask.HUMANEVAL` | Code generation | Python programming |

### Custom Evaluation

| Type | Task ID | Description |
|------|---------|-------------|
| **BYOD** | `EvaluationTask.GEN_QA` | Bring Your Own Data |
| **BYOM** | `EvaluationTask.GEN_QA` + Lambda | Bring Your Own Metrics |
| **LLM Judge** | `EvaluationTask.LLM_JUDGE` | LLM-as-Judge evaluation |

### MMLU Subtasks
- `medical` - Medical subjects
- `stem` - Science, Technology, Engineering, Math
- `humanities` - Arts, history, philosophy
- `social_sciences` - Psychology, sociology, economics

---

## Deployment Options

### Amazon Bedrock

| Capability | Platform | Description |
|-----------|----------|-------------|
| **Bedrock On-Demand** | `DeployPlatform.BEDROCK_OD` | Pay-per-use inference |
| **Bedrock Provisioned** | `DeployPlatform.BEDROCK_PT` | Reserved throughput |
| **Custom Model Import** | Both | Import trained checkpoints |
| **Inference API** | Both | Bedrock API integration |

**Configuration Options:**
- `unit_count` - Number of units (for PT)
- `endpoint_name` - Custom model name
- Automatic model import from checkpoints

### Amazon SageMaker

| Capability | Platform | Description |
|-----------|----------|-------------|
| **Real-time Endpoint** | `DeployPlatform.SAGEMAKER` | Online inference |
| **Endpoint Config** | Custom | Instance type, count, autoscaling |
| **Environment Variables** | Custom | Context length, concurrency |
| **Endpoint Updates** | In-place | Update running endpoints |

**Configuration Options:**
- `instance_type` - GPU instance type
- `unit_count` - Number of instances
- `sagemaker_environment_variables` - Custom env vars
  - `CONTEXT_LENGTH`
  - `MAX_CONCURRENCY`
  - Custom variables

---

## Hyperparameter Configuration

### Common Training Hyperparameters

| Parameter | Type | Default | Description | Typical Range |
|-----------|------|---------|-------------|---------------|
| `lr` | float | - | Learning rate | 1e-7 to 1e-5 |
| `max_epochs` | int | - | Training epochs | 1-10 |
| `max_steps` | int | - | Training steps | 100-10000 |
| `save_steps` | int | - | Checkpoint frequency | 50-500 |
| `warmup_steps` | int | - | LR warmup steps | 10-200 |
| `global_batch_size` | int | - | Total batch size | 16-256 |
| `max_length` | int | - | Max sequence length | 2048-32768 |
| `loraplus_lr_ratio` | float | 16.0 | LoRA+ LR ratio | 10.0-20.0 |

### Evaluation Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens` | int | 1024 | Max tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.9 | Nucleus sampling |
| `top_k` | int | 50 | Top-k sampling |

---

## Advanced Features

### Iterative Training
**Capability**: Continue training from previous checkpoints
**Use Case**: Multi-stage fine-tuning, curriculum learning

```python
# Stage 1
stage1_result = customizer1.train(job_name="stage1")

# Stage 2 - continue from Stage 1
customizer2 = NovaModelCustomizer(
    model_path=stage1_result.model_artifacts.checkpoint_s3_path,
    # ... other config
)
stage2_result = customizer2.train(job_name="stage2")
```

### Dry Run Mode
**Capability**: Validate configuration without starting jobs
**Use Case**: Test recipes, validate inputs, catch errors early

```python
result = customizer.train(job_name="test", dry_run=True)
# Only validates, doesn't start training
```

### Data Mixing (Forge Only)
**Capability**: Blend custom data with Nova's curated datasets
**Platform**: SMHP only
**Methods**: CPT, SFT_LORA, SFT_FULL

```python
customizer = NovaModelCustomizer(
    data_mixing_enabled=True,
    # ... other config
)

customizer.set_data_mixing_config({
    "customer_data_percent": 50,
    "nova_code_percent": 30,
    "nova_general_percent": 70
})
```

### Job Caching
**Capability**: Cache and reuse completed job results
**Use Case**: Avoid redundant training jobs

```python
customizer = NovaModelCustomizer(
    enable_job_caching=True,
    # ... other config
)
```

### Validation Configuration
**Capability**: Control pre-training validations

```python
customizer = NovaModelCustomizer(
    validation_config={
        "iam": True,          # Validate IAM permissions
        "infra": True,        # Validate infrastructure
        "rft_lambda": True    # Validate RFT Lambda (for RFT only)
    },
    # ... other config
)
```

---

## Instance Type Support

### SMTJ (SageMaker Training Jobs)

| Instance Type | GPUs | Memory | Use Case |
|--------------|------|---------|----------|
| ml.p4d.24xlarge | 8x A100 | 320 GB | Medium datasets |
| ml.p5.48xlarge | 8x H100 | 640 GB | **Recommended** |
| ml.trn1.32xlarge | 16x Trainium | 512 GB | Cost-effective |

### SMHP (SageMaker HyperPod)

Same instance types as SMTJ, plus:
- Multi-node training support
- Persistent clusters
- Better for large-scale training

**Full specifications**: See `docs/instance_type_spec.md` in SDK

---

## Monitoring & Logging

### CloudWatch Logs

| Capability | Description |
|-----------|-------------|
| **Real-time Logs** | View logs during training |
| **Historical Logs** | Access past job logs |
| **Filter Logs** | Search and filter log entries |
| **Tail Logs** | Follow latest log output |

### MLflow Tracking

| Capability | Description |
|-----------|-------------|
| **Metrics Logging** | Track training metrics |
| **Parameter Tracking** | Log hyperparameters |
| **Artifact Storage** | Store model artifacts |
| **Experiment Management** | Organize related runs |
| **UI Access** | Generate presigned URLs |

---

## Supported AWS Regions

**Primary Regions** (full support):
- us-east-1 (N. Virginia)
- us-west-2 (Oregon)
- eu-west-1 (Ireland)

**Check Current SDK for Latest Region Support**

---

## Error Handling & Validation

### Pre-Training Validations

| Validation | Description | Can Disable |
|-----------|-------------|-------------|
| **IAM Permissions** | Check role permissions | Yes |
| **Infrastructure** | Verify instance availability | Yes |
| **Data Format** | Validate dataset structure | No |
| **RFT Lambda** | Test Lambda function | Yes (RFT only) |

### Common Error Types

| Error | Category | Typical Cause |
|-------|----------|---------------|
| `AccessDenied` | IAM | Missing permissions |
| `ResourceNotFound` | Infrastructure | Invalid cluster/instance |
| `ValidationError` | Data | Invalid dataset format |
| `OutOfMemory` | Runtime | Batch size too large |
| `Divergence` | Training | Learning rate too high |

---

## Limitations & Constraints

### Platform Limitations

| Limitation | SMTJ | SMHP |
|-----------|------|------|
| CPT Support | ❌ | ✅ |
| RFT Multiturn | ❌ | ✅ |
| Multi-job | ❌ | ✅ |
| Persistent Cluster | ❌ | ✅ |

### Data Limitations

| Constraint | Value | Notes |
|-----------|-------|-------|
| Min training samples | 50 | Recommended minimum |
| Max sequence length | 300k | Depends on model |
| Supported formats | JSONL, JSON, CSV | JSONL recommended |

### Instance Limitations

- Instance availability subject to AWS quotas
- Some regions have limited instance types
- Contact AWS support for quota increases

---

## Performance Characteristics

### Training Speed (Approximate)

| Model | Instance | Dataset | Time |
|-------|----------|---------|------|
| Nova Lite 2 (LoRA) | 4x ml.p5.48xlarge | 1K samples | ~30-60 min |
| Nova Lite 2 (LoRA) | 4x ml.p5.48xlarge | 10K samples | ~1-2 hours |
| Nova Lite 2 (Full) | 4x ml.p5.48xlarge | 1K samples | ~1-2 hours |
| Nova Pro (LoRA) | 8x ml.p5.48xlarge | 10K samples | ~2-4 hours |

### Memory Requirements

| Method | Model | Typical GPU Memory | Instance Recommendation |
|--------|-------|-------------------|------------------------|
| SFT_LORA | Nova Lite 2 | ~40 GB | ml.p5.48xlarge |
| SFT_FULL | Nova Lite 2 | ~80 GB | ml.p5.48xlarge |
| SFT_LORA | Nova Pro | ~60 GB | ml.p5.48xlarge |
| SFT_FULL | Nova Pro | ~120 GB | 2x ml.p5.48xlarge |

---

## Security & Compliance

### Data Security

| Feature | Support | Description |
|---------|---------|-------------|
| **Encryption at Rest** | ✅ | S3 server-side encryption |
| **Encryption in Transit** | ✅ | TLS for all API calls |
| **VPC Support** | ✅ | SMHP clusters in VPC |
| **IAM Integration** | ✅ | Fine-grained permissions |

### Compliance

- AWS compliance programs inherited
- Data stays in your AWS account
- No data sharing with AWS (except standard telemetry)

---

## SDK Version Information

**Current SDK Version**: Check with `pip show amzn-amzn-nova-forge`

**Python Requirements**: Python 3.12+

**Dependencies**:
- boto3
- sagemaker==2.254.1 (automatically installed)
- Additional dependencies as needed

---

## Future Roadmap (Subject to Change)

### Planned Features
- Additional training methods
- More evaluation benchmarks
- Enhanced monitoring
- Performance optimizations

### Forge Features (Phase 2)
- Data mixing (currently available)
- Advanced optimization techniques
- Multi-modal support (future)

---

*Last Updated: 2026-03-14*
*For the latest capabilities, always refer to the official SDK documentation at `docs/spec.md`*
