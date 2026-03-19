# Journey 3: SFT Training (Supervised Fine-Tuning)

## Overview
**Purpose**: Fine-tune Nova models with supervised question-answer pairs
**Time**: 2-4 hours (including training time)
**Prerequisites**: Journey 1 (Setup), Journey 2 (Data Prep)
**Outputs**: Trained model checkpoint ready for evaluation/deployment

---

## What You'll Learn

- Configure SFT training jobs (LoRA vs Full-rank)
- Choose between SMTJ and SMHP platforms
- Set hyperparameters for optimal performance
- Start and monitor training jobs
- Handle training errors and optimize performance
- Export and save model checkpoints

---

## What is SFT?

**Supervised Fine-Tuning (SFT)** is the most common training method for customizing Nova models:

- **Input**: Labeled question-answer pairs
- **Purpose**: Adapt model to specific tasks or domains
- **Use Cases**: Q&A systems, instruction following, task-specific behavior
- **Training Time**: 1-3 hours (depends on dataset size and instance type)

### LoRA vs Full-Rank

| Feature | LoRA (Low-Rank Adaptation) | Full-Rank |
|---------|---------------------------|-----------|
| **Speed** | Faster (fewer parameters) | Slower (all parameters) |
| **Memory** | Lower memory usage | Higher memory usage |
| **Quality** | Good for most tasks | Best quality possible |
| **Cost** | Lower compute cost | Higher compute cost |
| **Recommendation** | **Start here** | Use if LoRA insufficient |

---

## Step 1: Choose Platform (SMTJ vs SMHP)

### Option A: SMTJ (SageMaker Training Jobs) - Recommended for Beginners

**Pros:**
- Simpler setup
- Auto-scaling
- Pay per job
- Good for experimentation

**Cons:**
- No CPT support
- Single job at a time

```python
from amzn_nova_forge import *

# Configure SMTJ runtime
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",  # GPU instance
    instance_count=4,                 # Number of instances
    # execution_role="arn:aws:iam::123:role/MyRole"  # Optional
)

print("✅ SMTJ Runtime configured")
print(f"   Instance Type: {runtime.instance_type}")
print(f"   Instance Count: {runtime.instance_count}")
```

### Option B: SMHP (SageMaker HyperPod) - For Production

**Pros:**
- All methods supported (including CPT)
- Persistent clusters
- Multi-job support
- Better for production

**Cons:**
- More complex setup
- Requires cluster management

```python
# Configure SMHP runtime
runtime = SMHPRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=4,
    cluster_name="my-hyperpod-cluster",  # Your cluster name
    namespace="kubeflow"                  # Kubernetes namespace
)

print("✅ SMHP Runtime configured")
```

**Instance Type Recommendations:**

| Dataset Size | Recommended Instance | Estimated Time |
|--------------|---------------------|----------------|
| < 1K samples | ml.p4d.24xlarge (4 nodes) | 30-60 min |
| 1K - 10K samples | ml.p5.48xlarge (4 nodes) | 1-2 hours |
| 10K - 100K samples | ml.p5.48xlarge (8 nodes) | 2-4 hours |
| > 100K samples | ml.p5.48xlarge (16+ nodes) | 4-8 hours |

---

## Step 2: Configure MLflow Monitoring (Optional but Recommended)

Track experiments with MLflow:

```python
# Create MLflow monitor
mlflow_monitor = MLflowMonitor(
    tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/my-server",
    experiment_name="nova-sft-experiments",
    run_name="sft-lora-run-1"
)

# Generate presigned URL to access MLflow UI
mlflow_url = mlflow_monitor.get_presigned_url()
print(f"📊 MLflow UI: {mlflow_url}")
```

---

## Step 3: Initialize NovaModelCustomizer

```python
from amzn_nova_forge import *

# Load environment variables
S3_BUCKET = "your-bucket-name"
S3_PREFIX = "nova-training"
train_data_s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/sft/train.jsonl"
output_s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/sft/output/"

# Create customizer
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,           # Choose your Nova model
    method=TrainingMethod.SFT_LORA,    # LoRA for efficiency
    infra=runtime,                     # Runtime from Step 1
    data_s3_path=train_data_s3_path,  # Training data
    output_s3_path=output_s3_path,    # Output location
    mlflow_monitor=mlflow_monitor,     # Optional monitoring
    validation_config={                 # Enable validations
        "iam": True,
        "infra": True
    }
)

print("✅ NovaModelCustomizer initialized")
print(f"   Model: {customizer.model}")
print(f"   Method: {customizer.method}")
print(f"   Data: {train_data_s3_path}")
```

---

## Step 4: Configure Training Hyperparameters

### 4.1 Common Hyperparameters

```python
# Basic hyperparameter configuration
training_config = {
    # Learning rate (most important!)
    "lr": 5e-6,  # 5e-6 is good starting point for LoRA
                 # 1e-6 for Full-rank

    # Training duration
    "max_epochs": 3,        # Number of passes through data
    "max_steps": 1000,      # Alternative to epochs
    "save_steps": 100,      # Save checkpoint every N steps

    # Optimization
    "warmup_steps": 100,    # Learning rate warmup
    "global_batch_size": 64,  # Total batch size across GPUs

    # Sequence length
    "max_length": 8192,     # Maximum tokens per sample
                            # Nova Lite 2: up to 256k
                            # Use 8192 for efficiency

    # LoRA-specific (if using LoRA)
    "loraplus_lr_ratio": 16.0,  # LoRA+ learning rate ratio
}

print("✅ Training configuration:")
for key, value in training_config.items():
    print(f"   {key}: {value}")
```

### 4.2 Hyperparameter Tuning Guidelines

**Learning Rate (`lr`):**
- Too high: Training unstable, loss explodes
- Too low: Training too slow, underfitting
- **Recommended**: Start with `5e-6` for LoRA, `1e-6` for Full-rank

**Global Batch Size (`global_batch_size`):**
- Larger batch: More stable, but slower per step
- Smaller batch: Faster per step, more noisy
- **Recommended**: 64-128 for most cases

**Max Length (`max_length`):**
- Longer sequences: More context, higher memory
- Shorter sequences: Less memory, faster
- **Recommended**: 8192 for efficiency, 32k for long context

**Warmup Steps (`warmup_steps`):**
- Gradually increases learning rate from 0 to `lr`
- Prevents early training instability
- **Recommended**: 10% of total steps (e.g., 100 for 1000 steps)

---

## Step 5: Start Training

### 5.1 Run Training Job

```python
# Start training
training_result = customizer.train(
    job_name="sft-nova-lite2-lora-v1",  # Unique job name
    overrides=training_config,           # Hyperparameters
    validation_data_s3_path=val_data_s3_path  # Optional validation set
)

print("\n🚀 Training job started!")
print(f"   Job ID: {training_result.job_id}")
print(f"   Started: {training_result.started_time}")
print(f"   Checkpoint: {training_result.model_artifacts.checkpoint_s3_path}")
print(f"   Output: {training_result.model_artifacts.output_s3_path}")

# Save job info for later
job_id = training_result.job_id
checkpoint_path = training_result.model_artifacts.checkpoint_s3_path
```

### 5.2 Dry Run (Test Without Training)

Test your configuration without starting a job:

```python
# Dry run mode
training_result = customizer.train(
    job_name="sft-dry-run-test",
    overrides=training_config,
    dry_run=True  # Only validates, doesn't start job
)

print("✅ Dry run completed - configuration is valid!")
```

---

## Step 6: Monitor Training Progress

### 6.1 View Real-Time Logs

```python
# View recent logs (while training is running)
print("📋 Training Logs:")
print("=" * 80)
customizer.get_logs(
    limit=50,              # Number of log lines
    start_from_head=False  # False = most recent logs
)
```

### 6.2 Monitor Training Metrics

Training logs show key metrics:

```
Step 100/1000 | Loss: 2.345 | LR: 5.0e-6 | Tokens/sec: 1234
Step 200/1000 | Loss: 1.876 | LR: 5.0e-6 | Tokens/sec: 1256
Step 300/1000 | Loss: 1.543 | LR: 5.0e-6 | Tokens/sec: 1248
```

**What to watch:**
- **Loss**: Should decrease over time
- **Learning Rate (LR)**: Should match your config
- **Tokens/sec**: Throughput (higher is better)

### 6.3 Check Training Status

```python
import boto3

# Check job status via SageMaker
sagemaker = boto3.client('sagemaker')
status = sagemaker.describe_training_job(TrainingJobName=job_id)

print(f"📊 Job Status: {status['TrainingJobStatus']}")
print(f"   Secondary Status: {status['SecondaryStatus']}")
# Status values: InProgress, Completed, Failed, Stopping, Stopped
```

### 6.4 View Logs After Completion

```python
from amzn_nova_forge import Platform

# After training completes
monitor = CloudWatchLogMonitor.from_job_id(
    job_id=job_id,
    platform=Platform.SMTJ  # or Platform.SMHP
)

# Show full logs
monitor.show_logs(limit=100, start_from_head=True)
```

---

## Step 7: Handle Training Issues

### 7.1 Out of Memory (OOM) Errors

**Symptoms:**
```
CUDA out of memory. Tried to allocate XXX MiB
```

**Solutions:**

```python
# Option 1: Reduce batch size
training_config = {
    "global_batch_size": 32,  # Reduced from 64
    "lr": 5e-6
}

# Option 2: Reduce max_length
training_config = {
    "max_length": 4096,  # Reduced from 8192
    "global_batch_size": 64
}

# Option 3: Use gradient checkpointing (automatic in SDK)
# Option 4: Increase instance count
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=8  # Increased from 4
)
```

### 7.2 Training Divergence (Loss Explodes)

**Symptoms:**
```
Step 50 | Loss: 2.3
Step 100 | Loss: 5.7
Step 150 | Loss: NaN
```

**Solutions:**

```python
# Reduce learning rate
training_config = {
    "lr": 1e-6,  # Reduced from 5e-6
    "warmup_steps": 200  # Increased warmup
}

# Or reduce global batch size
training_config = {
    "lr": 5e-6,
    "global_batch_size": 32  # Smaller batches
}
```

### 7.3 Slow Training

**Solutions:**

```python
# Option 1: Increase instance count
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=8  # More parallelism
)

# Option 2: Reduce max_length
training_config = {
    "max_length": 4096  # Faster per step
}

# Option 3: Use LoRA instead of Full-rank
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,  # Faster than SFT_FULL
    # ...
)
```

---

## Step 8: Save and Export Checkpoint

### 8.1 Access Checkpoint Path

```python
# Checkpoint is automatically saved during training
checkpoint_s3_path = training_result.model_artifacts.checkpoint_s3_path
print(f"📁 Checkpoint: {checkpoint_s3_path}")

# Download checkpoint locally (optional)
import boto3

s3 = boto3.client('s3')
bucket, key = checkpoint_s3_path.replace("s3://", "").split("/", 1)

# Download
s3.download_file(bucket, key, "local_checkpoint.tar.gz")
print("✅ Checkpoint downloaded locally")
```

### 8.2 Save Training Result for Later

```python
# Save training result to file
training_result.dump("training_result.json")
print("✅ Training result saved")

# Load later
from amzn_nova_forge import TrainingResult
loaded_result = TrainingResult.load("training_result.json")
print(f"✅ Loaded training result: {loaded_result.job_id}")
```

---

## Step 9: Iterative Training (Continue from Checkpoint)

Continue training from a previous checkpoint:

```python
# Stage 1: Initial training
stage1_customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime,
    data_s3_path="s3://bucket/stage1-data.jsonl",
    output_s3_path="s3://bucket/stage1-output/"
)

stage1_result = stage1_customizer.train(
    job_name="stage1-training",
    overrides={"max_epochs": 3}
)

# Get checkpoint from Stage 1
stage1_checkpoint = stage1_result.model_artifacts.checkpoint_s3_path

# Stage 2: Continue training from Stage 1 checkpoint
stage2_customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,  # Must match Stage 1
    infra=runtime,
    data_s3_path="s3://bucket/stage2-data.jsonl",
    output_s3_path="s3://bucket/stage2-output/",
    model_path=stage1_checkpoint  # Continue from Stage 1
)

stage2_result = stage2_customizer.train(
    job_name="stage2-training",
    overrides={"max_epochs": 2}
)

print("✅ Iterative training complete")
print(f"   Stage 1 checkpoint: {stage1_checkpoint}")
print(f"   Stage 2 checkpoint: {stage2_result.model_artifacts.checkpoint_s3_path}")
```

---

## Quick Reference

### Minimal Training Example

```python
from amzn_nova_forge import *

# Setup
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=4
)

# Create customizer
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime,
    data_s3_path="s3://bucket/train.jsonl",
    output_s3_path="s3://bucket/output/"
)

# Train
result = customizer.train(
    job_name="my-sft-training",
    overrides={"lr": 5e-6, "max_epochs": 3}
)

print(f"✅ Training started: {result.job_id}")
```

### Recommended Hyperparameters by Model

```python
# Nova Lite 2 (Most Common)
NOVA_LITE_2_CONFIG = {
    "lr": 5e-6,
    "warmup_steps": 100,
    "global_batch_size": 64,
    "max_length": 8192,
    "max_epochs": 3
}

# Nova Micro (Fast Iteration)
NOVA_MICRO_CONFIG = {
    "lr": 5e-6,
    "warmup_steps": 50,
    "global_batch_size": 128,
    "max_length": 4096,
    "max_epochs": 3
}

# Nova Pro (High Quality)
NOVA_PRO_CONFIG = {
    "lr": 1e-6,  # Lower LR for larger model
    "warmup_steps": 200,
    "global_batch_size": 32,
    "max_length": 8192,
    "max_epochs": 3
}
```

---

## Next Steps

✅ **Training complete!** Now you can:

- **Journey 7: Evaluation** - Benchmark your model's performance
- **Journey 8: Bedrock Deployment** - Deploy to Bedrock
- **Journey 9: SageMaker Deployment** - Deploy to SageMaker
- **Journey 10: Inference & Monitoring** - Run inference and monitor

---

## Resources

- [Nova SFT Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-sft.html)
- [SDK Training Spec](../amzn-nova-forge/docs/spec.md#train)
- [Hyperparameter Tuning Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-hyperparameters.html)
- [Instance Type Specifications](../amzn-nova-forge/docs/instance_type_spec.md)
