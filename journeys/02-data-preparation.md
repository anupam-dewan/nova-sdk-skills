# Journey 2: Data Preparation

## Overview
**Purpose**: Load, transform, validate, and prepare datasets for Nova model training
**Time**: 15-30 minutes
**Prerequisites**: Journey 1 (Setup)
**Outputs**: Validated training data uploaded to S3

---

## What You'll Learn

- Understand dataset format requirements
- Load data from JSONL, JSON, and CSV files
- Transform data to required formats (SFT, CPT, DPO, RFT)
- Validate dataset structure and content
- Split data into train/validation/test sets
- Upload datasets to S3

---

## Dataset Format Overview

Different training methods require different data formats:

| Training Method | Required Fields | Format | Example Use Case |
|----------------|-----------------|--------|------------------|
| **SFT** (Supervised Fine-Tuning) | `question`, `answer` | Converse format | Q&A, instruction following |
| **CPT** (Continued Pre-Training) | `text` | Plain text | Domain adaptation, unlabeled corpus |
| **DPO** (Direct Preference Optimization) | `messages`, `candidates` | Preference pairs | Alignment, safety |
| **RFT** (Reinforcement Fine-Tuning) | `id`, `messages`, `reference_answer` | Converse + rewards | Code generation, math |

---

## Step 1: Understanding Data Loaders

The SDK provides three data loader classes:

```python
from amzn_nova_customization_sdk import *

# For JSONL files (recommended)
loader = JSONLDatasetLoader(
    question="question",  # Column name for questions
    answer="answer"       # Column name for answers
)

# For JSON files
loader = JSONDatasetLoader(
    question="question",
    answer="answer"
)

# For CSV files
loader = CSVDatasetLoader(
    question="question",
    answer="answer"
)
```

---

## Step 2: Prepare Data for SFT Training

### 2.1 Create Sample SFT Data

```python
import json

# Sample question-answer pairs
sft_data = [
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
    },
    {
        "question": "Explain what AWS is.",
        "answer": "AWS (Amazon Web Services) is a comprehensive cloud computing platform that provides on-demand computing resources and services."
    },
    {
        "question": "What is Python used for?",
        "answer": "Python is a versatile programming language used for web development, data analysis, artificial intelligence, scientific computing, and automation."
    },
    {
        "question": "Define API.",
        "answer": "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other."
    },
    {
        "question": "What is cloud computing?",
        "answer": "Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, and analytics over the Internet."
    }
]

# Expand dataset (for demo purposes, duplicate 20 times for ~100 samples)
sft_data = sft_data * 20

# Save to JSONL file
with open("sft_training_data.jsonl", "w") as f:
    for item in sft_data:
        f.write(json.dumps(item) + "\n")

print(f"✅ Created {len(sft_data)} training samples")
```

### 2.2 Load and Transform SFT Data

```python
from amzn_nova_customization_sdk import *

# Initialize loader with column mappings
loader = JSONLDatasetLoader(
    question="question",  # Your data's question column
    answer="answer"       # Your data's answer column
)

# Load the data
loader.load("sft_training_data.jsonl")

# Preview raw data
print("📊 Raw Data Preview:")
loader.show(n=3)

# Transform to Converse format
loader.transform(
    method=TrainingMethod.SFT_LORA,  # or SFT_FULL
    model=Model.NOVA_LITE_2
)

print("\n✅ Data transformed to Converse format")
print("📊 Transformed Data Preview:")
loader.show(n=3)
```

**Converse Format Output:**
```json
{
    "messages": [
        {
            "role": "user",
            "content": "What is machine learning?"
        },
        {
            "role": "assistant",
            "content": "Machine learning is a subset of artificial intelligence..."
        }
    ]
}
```

### 2.3 Validate SFT Data

```python
# Validate dataset format and content
loader.validate(
    method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2
)
# Output: "✅ Validation completed successfully"
```

### 2.4 Split and Save SFT Data

```python
# Split into train/validation/test sets
train_loader, val_loader, test_loader = loader.split_data(
    train_ratio=0.7,   # 70% training
    val_ratio=0.2,     # 20% validation
    test_ratio=0.1     # 10% test
)

print(f"📊 Dataset Split:")
print(f"   Training samples: {len(train_loader.dataset)}")
print(f"   Validation samples: {len(val_loader.dataset)}")
print(f"   Test samples: {len(test_loader.dataset)}")

# Upload to S3
S3_BUCKET = "your-bucket-name"  # Replace with your bucket
S3_PREFIX = "nova-training"

train_s3_path = train_loader.save_data(f"s3://{S3_BUCKET}/{S3_PREFIX}/sft/train.jsonl")
val_s3_path = val_loader.save_data(f"s3://{S3_BUCKET}/{S3_PREFIX}/sft/val.jsonl")
test_s3_path = test_loader.save_data(f"s3://{S3_BUCKET}/{S3_PREFIX}/sft/test.jsonl")

print(f"\n✅ Data uploaded to S3:")
print(f"   Training: {train_s3_path}")
print(f"   Validation: {val_s3_path}")
print(f"   Test: {test_s3_path}")
```

---

## Step 3: Prepare Data for CPT Training

### 3.1 Create Sample CPT Data

CPT requires **plain text** data (no question-answer structure):

```python
import json

# Sample domain-specific text
cpt_data = [
    {"text": "Machine learning is a method of data analysis that automates analytical model building."},
    {"text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers."},
    {"text": "Natural language processing enables computers to understand, interpret, and generate human language."},
    {"text": "Computer vision allows machines to extract information from digital images and videos."},
    {"text": "Reinforcement learning involves training models through trial and error with rewards and penalties."}
] * 20  # Expand for demo

# Save to JSONL
with open("cpt_training_data.jsonl", "w") as f:
    for item in cpt_data:
        f.write(json.dumps(item) + "\n")

print(f"✅ Created {len(cpt_data)} CPT samples")
```

### 3.2 Load and Save CPT Data

```python
# Initialize loader for CPT (text-only)
loader = JSONLDatasetLoader(
    text="text"  # Column name for text data
)

# Load data
loader.load("cpt_training_data.jsonl")

# Preview
print("📊 CPT Data Preview:")
loader.show(n=3)

# No transformation needed for CPT - use raw text
# But you can still split if needed
train_loader, val_loader, _ = loader.split_data(
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.0  # CPT typically doesn't need test set
)

# Upload to S3
train_s3_path = train_loader.save_data(f"s3://{S3_BUCKET}/{S3_PREFIX}/cpt/train.jsonl")
val_s3_path = val_loader.save_data(f"s3://{S3_BUCKET}/{S3_PREFIX}/cpt/val.jsonl")

print(f"\n✅ CPT data uploaded to S3:")
print(f"   Training: {train_s3_path}")
print(f"   Validation: {val_s3_path}")
```

---

## Step 4: Prepare Data for DPO Training

### 4.1 Create Sample DPO Data

DPO requires **preference pairs** (preferred vs non-preferred responses):

```python
import json

# Sample preference data
dpo_data = [
    {
        "messages": [
            {
                "role": "user",
                "content": [{
                    "text": "What is AWS?"
                }]
            },
            {
                "role": "assistant",
                "candidates": [
                    {
                        "content": [{
                            "text": "AWS is Amazon's comprehensive cloud computing platform that provides scalable infrastructure, storage, databases, and machine learning services."
                        }],
                        "preferenceLabel": "preferred"
                    },
                    {
                        "content": [{
                            "text": "AWS is a thing that does cloud stuff."
                        }],
                        "preferenceLabel": "non-preferred"
                    }
                ]
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": [{
                    "text": "Explain machine learning."
                }]
            },
            {
                "role": "assistant",
                "candidates": [
                    {
                        "content": [{
                            "text": "Machine learning is a subset of AI that enables systems to learn from data and improve their performance without explicit programming."
                        }],
                        "preferenceLabel": "preferred"
                    },
                    {
                        "content": [{
                            "text": "It's when computers learn stuff."
                        }],
                        "preferenceLabel": "non-preferred"
                    }
                ]
            }
        ]
    }
] * 50  # Expand for demo

# Save to JSONL
with open("dpo_training_data.jsonl", "w") as f:
    for item in dpo_data:
        f.write(json.dumps(item) + "\n")

print(f"✅ Created {len(dpo_data)} DPO preference pairs")
```

### 4.2 Load and Save DPO Data

```python
# DPO loader doesn't need column mappings (uses standard format)
loader = JSONLDatasetLoader()

# Load data
loader.load("dpo_training_data.jsonl")

# Preview
print("📊 DPO Data Preview:")
loader.show(n=2)

# Split and save
train_loader, val_loader, _ = loader.split_data(
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.0
)

# Upload to S3
train_s3_path = train_loader.save_data(f"s3://{S3_BUCKET}/{S3_PREFIX}/dpo/train.jsonl")
val_s3_path = val_loader.save_data(f"s3://{S3_BUCKET}/{S3_PREFIX}/dpo/val.jsonl")

print(f"\n✅ DPO data uploaded to S3:")
print(f"   Training: {train_s3_path}")
print(f"   Validation: {val_s3_path}")
```

---

## Step 5: Prepare Data for RFT Training

### 5.1 Create Sample RFT Data

RFT requires **id**, **messages**, and optionally **reference_answer**:

```python
import json

# Sample RFT data with reference answers
rft_data = [
    {
        "id": f"sample_{i}",
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function to calculate factorial."
            },
            {
                "role": "assistant",
                "content": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)"
            }
        ],
        "reference_answer": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)"
    }
    for i in range(100)
]

# Save to JSONL
with open("rft_training_data.jsonl", "w") as f:
    for item in rft_data:
        f.write(json.dumps(item) + "\n")

print(f"✅ Created {len(rft_data)} RFT samples")
```

### 5.2 Load and Save RFT Data

```python
# RFT loader
loader = JSONLDatasetLoader()

# Load data
loader.load("rft_training_data.jsonl")

# Preview
print("📊 RFT Data Preview:")
loader.show(n=3)

# Split and save
train_loader, val_loader, _ = loader.split_data(
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.0
)

# Upload to S3
train_s3_path = train_loader.save_data(f"s3://{S3_BUCKET}/{S3_PREFIX}/rft/train.jsonl")
val_s3_path = val_loader.save_data(f"s3://{S3_BUCKET}/{S3_PREFIX}/rft/val.jsonl")

print(f"\n✅ RFT data uploaded to S3:")
print(f"   Training: {train_s3_path}")
print(f"   Validation: {val_s3_path}")
```

---

## Step 6: Data Quality Best Practices

### 6.1 Check Dataset Size

```python
# Minimum recommended sizes
MIN_TRAIN_SAMPLES = {
    TrainingMethod.SFT_LORA: 50,      # At least 50 samples
    TrainingMethod.SFT_FULL: 100,     # At least 100 samples
    TrainingMethod.CPT: 1000,         # At least 1000 text samples
    TrainingMethod.DPO_LORA: 50,      # At least 50 preference pairs
    TrainingMethod.RFT_LORA: 50,      # At least 50 samples
}

# Check your dataset
dataset_size = len(train_loader.dataset)
method = TrainingMethod.SFT_LORA  # Your chosen method

if dataset_size < MIN_TRAIN_SAMPLES.get(method, 50):
    print(f"⚠️  Warning: Dataset has {dataset_size} samples")
    print(f"   Recommended minimum: {MIN_TRAIN_SAMPLES[method]} samples")
else:
    print(f"✅ Dataset size OK: {dataset_size} samples")
```

### 6.2 Check Data Quality

```python
def check_data_quality(loader, method):
    """Check common data quality issues"""

    dataset = loader.dataset
    issues = []

    # Check for empty fields
    for i, sample in enumerate(dataset):
        if method == TrainingMethod.SFT_LORA:
            messages = sample.get("messages", [])
            if not messages:
                issues.append(f"Sample {i}: Empty messages")
            for msg in messages:
                if not msg.get("content", "").strip():
                    issues.append(f"Sample {i}: Empty message content")

        elif method == TrainingMethod.CPT:
            text = sample.get("text", "")
            if not text.strip():
                issues.append(f"Sample {i}: Empty text")
            if len(text) < 10:
                issues.append(f"Sample {i}: Text too short ({len(text)} chars)")

    # Report issues
    if issues:
        print(f"⚠️  Found {len(issues)} data quality issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"   - {issue}")
    else:
        print("✅ No data quality issues found")

    return len(issues) == 0

# Check your data
check_data_quality(train_loader, TrainingMethod.SFT_LORA)
```

### 6.3 Check Token Lengths

```python
def estimate_token_length(text, chars_per_token=4):
    """Rough estimate of token count"""
    return len(text) // chars_per_token

def check_sequence_lengths(loader, max_length=8192):
    """Check if sequences fit within model's context window"""

    long_sequences = []

    for i, sample in enumerate(loader.dataset):
        messages = sample.get("messages", [])
        total_text = " ".join([
            msg.get("content", "")
            for msg in messages
        ])

        estimated_tokens = estimate_token_length(total_text)

        if estimated_tokens > max_length:
            long_sequences.append((i, estimated_tokens))

    if long_sequences:
        print(f"⚠️  Found {len(long_sequences)} sequences exceeding {max_length} tokens:")
        for idx, tokens in long_sequences[:5]:
            print(f"   Sample {idx}: ~{tokens} tokens")
        print(f"   Consider truncating or splitting long sequences")
    else:
        print(f"✅ All sequences fit within {max_length} token limit")

# Check sequence lengths
check_sequence_lengths(train_loader, max_length=8192)
```

---

## Step 7: Load Existing Data from S3

If you already have data in S3:

```python
from amzn_nova_customization_sdk import *

# Load from S3
loader = JSONLDatasetLoader(
    question="question",
    answer="answer"
)

# Specify S3 path
s3_path = "s3://your-bucket/path/to/data.jsonl"
loader.load(s3_path)

# Preview
loader.show(n=5)

# Transform and validate
loader.transform(method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
loader.validate(method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
```

---

## Common Issues & Troubleshooting

### Issue: "KeyError: 'question'" when loading data

**Solution**: Your data's column names don't match the loader configuration.

```python
# Check your data structure first
import json
with open("data.jsonl") as f:
    sample = json.loads(f.readline())
    print(sample.keys())  # See actual column names

# Then configure loader with correct names
loader = JSONLDatasetLoader(
    question="query",  # Use your actual column name
    answer="response"  # Use your actual column name
)
```

### Issue: "Validation failed" errors

**Solution**: Check data format requirements:

```python
# For SFT: messages must have 'role' and 'content'
# For CPT: must have 'text' field
# For DPO: must have 'messages' with 'candidates'
# For RFT: must have 'id', 'messages', optionally 'reference_answer'
```

### Issue: S3 upload fails with "Access Denied"

**Solution**: Check S3 bucket permissions:

```bash
# Verify bucket exists and you have access
aws s3 ls s3://your-bucket/

# Test write permission
echo "test" | aws s3 cp - s3://your-bucket/test.txt
```

---

## Quick Reference

### Dataset Format Cheat Sheet

```python
# SFT Format (after transformation)
{
    "messages": [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"}
    ]
}

# CPT Format
{
    "text": "Your domain-specific text here..."
}

# DPO Format
{
    "messages": [
        {"role": "user", "content": [{"text": "question"}]},
        {
            "role": "assistant",
            "candidates": [
                {"content": [{"text": "preferred"}], "preferenceLabel": "preferred"},
                {"content": [{"text": "rejected"}], "preferenceLabel": "non-preferred"}
            ]
        }
    ]
}

# RFT Format
{
    "id": "sample_1",
    "messages": [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"}
    ],
    "reference_answer": "ground_truth"
}
```

---

## Next Steps

✅ **Data prepared!** You're ready to proceed to training:

- **Journey 3: SFT Training** - Most common method
- **Journey 4: CPT Training** - Domain adaptation
- **Journey 5: DPO Training** - Preference alignment
- **Journey 6: RFT Training** - Reward-based optimization

---

## Resources

- [Nova Data Format Specification](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-data-format.html)
- [Converse API Format](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
- [SDK spec.md Dataset Module](../nova-customization-sdk/docs/spec.md#dataset-loaders)
