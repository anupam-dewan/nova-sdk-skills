# Journey 4: CPT Training (Continued Pre-Training)

## Overview
**Purpose**: Adapt Nova models to specific domains with unlabeled text
**Time**: 3-6 hours (including training time)
**Prerequisites**: Journey 1 (Setup), Journey 2 (Data Prep)
**Outputs**: Domain-adapted base model ready for SFT fine-tuning
**Platform**: SMHP only (CPT not supported on SMTJ)

---

## What You'll Learn

- Understand when to use CPT vs SFT
- Prepare domain-specific text corpora
- Configure CPT training jobs on SMHP
- Set optimal hyperparameters for domain adaptation
- Monitor and validate CPT training
- Chain CPT with SFT for best results

---

## What is CPT?

**Continued Pre-Training (CPT)** adapts Nova's base knowledge to specific domains:

- **Input**: Unlabeled domain-specific text (not Q&A pairs)
- **Purpose**: Teach model domain vocabulary, concepts, and patterns
- **Use Cases**: Medical, legal, finance, scientific domains
- **Training Time**: 2-5 hours (depends on corpus size)

### CPT vs SFT

| Aspect | CPT (Continued Pre-Training) | SFT (Supervised Fine-Tuning) |
|--------|----------------------------|----------------------------|
| **Input Data** | Unlabeled domain text | Labeled Q&A pairs |
| **Purpose** | Domain knowledge adaptation | Task-specific behavior |
| **When to Use** | New domain vocabulary/concepts | Task performance improvement |
| **Typical Order** | **First** (if needed) | **Second** (always) |
| **Example** | Medical textbooks, papers | Medical Q&A pairs |

### Typical Workflow

```
Raw Domain Text → CPT → Domain-Adapted Base Model → SFT → Task-Specific Model
```

**Example**: Medical AI Assistant
1. **CPT**: Train on medical textbooks (unlabeled) → Learn medical terminology
2. **SFT**: Train on medical Q&A pairs (labeled) → Learn to answer patient questions

---

## Step 1: Determine if You Need CPT

### Use CPT When:

✅ **Your domain has specialized vocabulary**
- Medical: "myocardial infarction", "ACE inhibitors"
- Legal: "tort law", "habeas corpus"
- Finance: "collateralized debt obligations"

✅ **You have large unlabeled corpora (10K+ documents)**
- Research papers
- Technical documentation
- Domain-specific books

✅ **Base Nova model lacks domain knowledge**
- Test with few-shot examples first
- If performance is poor, CPT helps

### Skip CPT When:

❌ **Your domain is general knowledge** (Nova already knows it)
❌ **You only have labeled Q&A pairs** (use SFT directly)
❌ **You have < 10K documents** (not enough for CPT benefit)

---

## Step 2: Prepare CPT Data

CPT requires **unlabeled plain text**, not Q&A pairs.

### Data Format

**Option A: Plain Text Files (Recommended)**

```text
# medical_corpus.txt (one document per line or paragraph)

The human heart is a muscular organ that pumps blood throughout the body...

Myocardial infarction, commonly known as a heart attack, occurs when...

Cardiac catheterization is a medical procedure used to diagnose...
```

**Option B: JSONL Format**

```json
{"text": "The human heart is a muscular organ that pumps blood..."}
{"text": "Myocardial infarction, commonly known as a heart attack..."}
{"text": "Cardiac catheterization is a medical procedure..."}
```

### Upload to S3

```python
import boto3

# Upload corpus to S3
s3 = boto3.client('s3')
s3.upload_file(
    'medical_corpus.txt',
    'my-training-bucket',
    'data/cpt/medical_corpus.txt'
)

print("✅ CPT corpus uploaded to S3")
```

### Validate Data Quality

```python
from amzn_nova_forge.data import Dataset

# Load and validate
dataset = Dataset.load_from_s3(
    "s3://my-training-bucket/data/cpt/medical_corpus.txt",
    data_type="cpt"  # Specify CPT format
)

print(f"✅ Loaded {len(dataset)} text segments")
print(f"   Total tokens: {dataset.total_tokens:,}")
print(f"   Avg tokens per segment: {dataset.avg_tokens:.1f}")

# Preview
print("\nSample segment:")
print(dataset[0][:200] + "...")
```

---

## Step 3: Configure SMHP Runtime (Required for CPT)

CPT **requires SMHP** - SMTJ does not support CPT.

```python
from amzn_nova_forge import *

# Configure SMHP runtime
runtime = SMHPRuntimeManager(
    instance_type="ml.p5.48xlarge",  # GPU instance for CPT
    instance_count=8,                 # More instances = faster
    cluster_name="my-hyperpod-cluster",
    namespace="kubeflow"
)

print("✅ SMHP Runtime configured for CPT")
print(f"   Instance Type: {runtime.instance_type}")
print(f"   Instance Count: {runtime.instance_count}")
```

### Instance Recommendations for CPT

| Corpus Size | Instances | Estimated Time |
|-------------|-----------|----------------|
| < 100M tokens | 4x ml.p5.48xlarge | 1-2 hours |
| 100M-500M tokens | 8x ml.p5.48xlarge | 2-4 hours |
| 500M-1B tokens | 16x ml.p5.48xlarge | 4-6 hours |

---

## Step 4: Initialize CPT Training

```python
from amzn_nova_forge import *

# Initialize customizer for CPT
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,           # Base model to adapt
    method=TrainingMethod.CPT,          # Continued Pre-Training
    infra=runtime,                      # SMHP runtime
    data_s3_path="s3://my-training-bucket/data/cpt/medical_corpus.txt",
    output_s3_path="s3://my-training-bucket/output/cpt/"
)

print("✅ CPT Customizer initialized")
print(f"   Model: {customizer.model}")
print(f"   Method: {customizer.method}")
```

---

## Step 5: Configure CPT Hyperparameters

```python
# CPT-specific hyperparameters
cpt_config = {
    # Learning rate (lower for CPT than SFT)
    "lr": 1e-5,                    # Start conservative

    # Batch size
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,  # Effective batch = 4*8 = 32

    # Training duration
    "num_train_epochs": 3,          # Typically 1-3 epochs for CPT
    "max_steps": None,              # Or use max_steps instead

    # Sequence length
    "max_seq_length": 2048,         # Context window

    # Optimization
    "warmup_ratio": 0.03,           # 3% warmup
    "weight_decay": 0.01,

    # Checkpointing
    "save_steps": 500,
    "save_total_limit": 3,          # Keep last 3 checkpoints

    # Logging
    "logging_steps": 10
}

print("✅ CPT hyperparameters configured")
```

### CPT vs SFT Hyperparameters

| Parameter | CPT (Domain Adaptation) | SFT (Task Training) |
|-----------|------------------------|---------------------|
| **Learning Rate** | 1e-5 to 5e-5 (lower) | 5e-6 to 5e-5 (higher) |
| **Epochs** | 1-3 (more data) | 3-5 (less data) |
| **Batch Size** | Larger (more stable) | Smaller (fine control) |

---

## Step 6: Start CPT Training

```python
# Start CPT training job
result = customizer.train(
    job_name="medical-cpt-v1",
    overrides=cpt_config,
    dry_run=False  # Set to True to validate config first
)

print("✅ CPT Training started!")
print(f"   Job ID: {result.job_id}")
print(f"   Job Name: {result.job_name}")
print(f"   Status: {result.status}")
print(f"   Model S3 Path: {result.model_s3_path}")
```

---

## Step 7: Monitor CPT Training

### View Training Logs

```python
# Monitor training progress
print("📋 CPT Training Logs:")
print("=" * 80)
customizer.get_logs(limit=50, start_from_head=False)
```

### Expected Log Output

```
Epoch 1/3 | Step 100/3000 | Loss: 2.456 | LR: 8.5e-6 | Perplexity: 11.65
Epoch 1/3 | Step 200/3000 | Loss: 2.234 | LR: 1.0e-5 | Perplexity: 9.34
Epoch 1/3 | Step 300/3000 | Loss: 2.087 | LR: 1.0e-5 | Perplexity: 8.06
```

**What to watch:**
- **Loss**: Should steadily decrease
- **Perplexity**: Lower is better (measures prediction quality)
- **LR**: Should match your config

---

## Step 8: Validate CPT Results

After CPT completes, test domain knowledge improvement:

```python
# Compare base model vs CPT model on domain text
from amzn_nova_forge.inference import compare_models

test_prompts = [
    "What is myocardial infarction?",
    "Explain ACE inhibitors",
    "Define cardiac catheterization"
]

comparison = compare_models(
    base_model=Model.NOVA_LITE_2,
    cpt_model=result.model_s3_path,
    prompts=test_prompts
)

for prompt, base_resp, cpt_resp in comparison:
    print(f"\nPrompt: {prompt}")
    print(f"Base Model: {base_resp[:100]}...")
    print(f"CPT Model: {cpt_resp[:100]}...")
```

---

## Step 9: Chain CPT with SFT

After CPT, use the domain-adapted model for SFT:

```python
# Use CPT output as base for SFT
sft_customizer = NovaModelCustomizer(
    model=result.model_s3_path,  # CPT checkpoint (not base model!)
    method=TrainingMethod.SFT_LORA,
    infra=runtime,
    data_s3_path="s3://my-training-bucket/data/sft/medical_qa.jsonl",
    output_s3_path="s3://my-training-bucket/output/sft/"
)

# Continue with SFT training
sft_result = sft_customizer.train(
    job_name="medical-sft-after-cpt",
    overrides={"lr": 5e-6, "num_train_epochs": 3}
)

print("✅ SFT training started on CPT-adapted model!")
```

---

## Common Issues & Solutions

### Issue 1: "CPT not supported on SMTJ"

**Error:**
```
ValidationError: CPT training requires SMHP runtime
```

**Solution:** Use `SMHPRuntimeManager` instead of `SMTJRuntimeManager`:
```python
runtime = SMHPRuntimeManager(
    cluster_name="my-hyperpod-cluster",
    instance_type="ml.p5.48xlarge",
    instance_count=8
)
```

### Issue 2: Training Loss Not Decreasing

**Symptom:** Loss stays flat or increases

**Solutions:**
```python
# 1. Lower learning rate
cpt_config["lr"] = 5e-6  # Was 1e-5

# 2. Increase warmup
cpt_config["warmup_ratio"] = 0.05  # Was 0.03

# 3. Check data quality
dataset.validate()  # Look for corrupted segments
```

### Issue 3: Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# 1. Reduce batch size
cpt_config["per_device_train_batch_size"] = 2  # Was 4

# 2. Reduce sequence length
cpt_config["max_seq_length"] = 1024  # Was 2048

# 3. Use more instances (distribute load)
runtime = SMHPRuntimeManager(instance_count=16)  # Was 8
```

### Issue 4: Corpus Too Small

**Warning:** CPT works best with 100M+ tokens

**Solutions:**
- Gather more domain text (research papers, books, docs)
- If < 100M tokens, consider skipping CPT and doing SFT only
- Try Data Mixing (Forge feature) to combine with Nova's curated data

---

## Quick Reference

### Minimal CPT Example

```python
from amzn_nova_forge import *

# 1. Configure SMHP (required)
runtime = SMHPRuntimeManager(
    cluster_name="my-cluster",
    instance_type="ml.p5.48xlarge",
    instance_count=8
)

# 2. Initialize CPT
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.CPT,
    infra=runtime,
    data_s3_path="s3://bucket/cpt/corpus.txt",
    output_s3_path="s3://bucket/output/cpt/"
)

# 3. Train
result = customizer.train(
    job_name="domain-cpt",
    overrides={"lr": 1e-5, "num_train_epochs": 2}
)

print(f"CPT complete: {result.model_s3_path}")
```

### CPT Best Practices

✅ **Do:**
- Use 100M+ tokens for meaningful adaptation
- Lower learning rate than SFT (1e-5 to 5e-5)
- Train for 1-3 epochs (more data = fewer epochs)
- Chain with SFT for best task performance
- Monitor perplexity on domain-specific eval set

❌ **Don't:**
- Use SMTJ (CPT requires SMHP)
- Use labeled Q&A data (that's for SFT)
- Train for too many epochs (overfitting risk)
- Skip validation (test domain knowledge improvement)

---

## Next Steps

**After CPT:**
1. **Validate domain knowledge** - Test on domain prompts
2. **Proceed to SFT** - Journey 3 using CPT checkpoint
3. **Evaluate combined model** - Journey 7
4. **Deploy** - Journey 8 (Bedrock) or Journey 9 (SageMaker)

**Related Journeys:**
- **Journey 2**: Data Preparation (prepare CPT corpus)
- **Journey 3**: SFT Training (chain after CPT)
- **Journey 7**: Evaluation (benchmark domain performance)

---

## Resources

- **CPT vs SFT Guide**: `reference/training-methods.md`
- **SMHP Setup**: Journey 1 (Setup & Prerequisites)
- **Data Mixing** (Forge): Advanced CPT with curated data
- **AWS SageMaker HyperPod**: https://docs.aws.amazon.com/sagemaker/latest/dg/hyperpod.html

---

**💡 Pro Tip**: Test if you need CPT first! Try SFT with your base model. If performance is poor due to domain vocabulary, then add CPT. CPT + SFT is powerful but takes longer than SFT alone.
