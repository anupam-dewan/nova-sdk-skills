# Journey 7: Evaluation

## Overview
**Purpose**: Benchmark and validate Nova model performance
**Time**: 30 minutes - 2 hours (depends on evaluation task)
**Prerequisites**: Journey 1 (Setup), optionally Journey 3/4/5/6 (for custom models)
**Outputs**: Evaluation metrics, performance reports

---

## What You'll Learn

- Run public benchmark evaluations (MMLU, HellaSwag, etc.)
- Evaluate with your own data (BYOD)
- Use custom metrics (BYOM - Bring Your Own Metrics)
- Run LLM-as-Judge evaluations
- Compare base vs custom models
- Interpret evaluation results

---

## Types of Evaluation

| Evaluation Type | Purpose | When to Use |
|----------------|---------|-------------|
| **Public Benchmarks** | Standard metrics (MMLU, etc.) | Check general capabilities, compare to baselines |
| **BYOD (Bring Your Own Data)** | Task-specific evaluation | Validate domain performance |
| **BYOM (Bring Your Own Metrics)** | Custom scoring logic | Complex evaluation criteria |
| **LLM-as-Judge** | Quality assessment | Subjective quality, style, alignment |

---

## Step 1: Configure Evaluation Infrastructure

```python
from amzn_nova_forge import *

# Configure runtime (can use fewer resources than training)
eval_runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=1  # Evaluation uses fewer instances
)

# Or for SMHP
# eval_runtime = SMHPRuntimeManager(
#     instance_type="ml.p5.48xlarge",
#     instance_count=1,
#     cluster_name="my-cluster",
#     namespace="kubeflow"
# )

print("✅ Evaluation runtime configured")
```

---

## Step 2: Initialize Evaluator

```python
S3_BUCKET = "your-bucket-name"
S3_PREFIX = "nova-evaluation"

# Create evaluator
evaluator = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,           # Model to evaluate
    method=TrainingMethod.EVALUATION,  # Use EVALUATION method
    infra=eval_runtime,
    data_s3_path=f"s3://{S3_BUCKET}/{S3_PREFIX}/eval-data.jsonl",  # Not used for benchmarks
    output_s3_path=f"s3://{S3_BUCKET}/{S3_PREFIX}/eval-output/"
)

print("✅ Evaluator initialized")
```

---

## Step 3: Public Benchmark Evaluation

### 3.1 Available Benchmarks

Common public benchmarks supported:

| Benchmark | Description | Measures |
|-----------|-------------|----------|
| **MMLU** | Massive Multitask Language Understanding | General knowledge (57 subjects) |
| **HellaSwag** | Commonsense reasoning | Situation understanding |
| **ARC-Challenge** | AI2 Reasoning Challenge | Science questions |
| **TruthfulQA** | Truthfulness | Factual accuracy |
| **GSM8K** | Grade School Math | Math reasoning |
| **HumanEval** | Code generation | Python programming |

### 3.2 Run MMLU Evaluation

```python
# Evaluate on MMLU benchmark
mmlu_result = evaluator.evaluate(
    job_name="eval-nova-lite2-mmlu",
    eval_task=EvaluationTask.MMLU,
    overrides={
        "max_new_tokens": 2048,
        "temperature": 0,  # Deterministic for benchmarks
        "top_p": 1.0
    }
)

print("\n🚀 MMLU Evaluation started!")
print(f"   Job ID: {mmlu_result.job_id}")
print(f"   Output: {mmlu_result.eval_output_path}")

# Save result
mmlu_job_id = mmlu_result.job_id
```

### 3.3 Run MMLU with Subtask

Evaluate on specific MMLU subtasks:

```python
# Evaluate only on medical subjects
mmlu_medical_result = evaluator.evaluate(
    job_name="eval-mmlu-medical",
    eval_task=EvaluationTask.MMLU,
    subtask="medical",  # Available subtasks: medical, stem, humanities, social_sciences
    overrides={"max_new_tokens": 2048}
)

print(f"✅ MMLU Medical evaluation started: {mmlu_medical_result.job_id}")
```

---

## Step 4: Evaluate Custom Models

### 4.1 Evaluate from Training Result

```python
# Load training result from previous training (Journey 3)
from amzn_nova_forge import TrainingResult

training_result = TrainingResult.load("training_result.json")

# Evaluate trained model
custom_eval_result = evaluator.evaluate(
    job_name="eval-custom-sft-model",
    eval_task=EvaluationTask.MMLU,
    job_result=training_result,  # Automatically extracts checkpoint
    overrides={"max_new_tokens": 2048}
)

print(f"✅ Custom model evaluation started: {custom_eval_result.job_id}")
```

### 4.2 Evaluate from Checkpoint Path

```python
# Evaluate specific checkpoint
checkpoint_path = "s3://bucket/output/checkpoint/"

custom_eval_result = evaluator.evaluate(
    job_name="eval-checkpoint",
    eval_task=EvaluationTask.MMLU,
    model_path=checkpoint_path,  # Direct checkpoint path
    overrides={"max_new_tokens": 2048}
)
```

---

## Step 5: BYOD (Bring Your Own Data) Evaluation

### 5.1 Prepare Evaluation Dataset

```python
import json

# Create evaluation dataset in GEN_QA format
eval_data = [
    {
        "id": "q1",
        "question": "What is AWS Lambda?",
        "expected_answer": "AWS Lambda is a serverless compute service that runs code in response to events."
    },
    {
        "id": "q2",
        "question": "Explain S3 bucket policies.",
        "expected_answer": "S3 bucket policies are JSON-based access policies that control permissions for S3 buckets."
    },
    {
        "id": "q3",
        "question": "What is Amazon EC2?",
        "expected_answer": "Amazon EC2 is a web service that provides resizable compute capacity in the cloud."
    }
] * 20  # Expand dataset

# Save to JSONL
with open("custom_eval_data.jsonl", "w") as f:
    for item in eval_data:
        f.write(json.dumps(item) + "\n")

# Upload to S3
from amzn_nova_forge import JSONLDatasetLoader

loader = JSONLDatasetLoader()
loader.load("custom_eval_data.jsonl")
eval_s3_path = loader.save_data(f"s3://{S3_BUCKET}/{S3_PREFIX}/custom-eval.jsonl")

print(f"✅ Evaluation data uploaded: {eval_s3_path}")
```

### 5.2 Run BYOD Evaluation

```python
# Evaluate with your own data
byod_result = evaluator.evaluate(
    job_name="eval-byod-custom-data",
    eval_task=EvaluationTask.GEN_QA,  # Generic Q&A evaluation
    data_s3_path=eval_s3_path,
    model_path=checkpoint_path,  # Optional: evaluate custom model
    overrides={
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    }
)

print(f"✅ BYOD evaluation started: {byod_result.job_id}")
```

---

## Step 6: BYOM (Bring Your Own Metrics) Evaluation

### 6.1 Create Evaluation Lambda Function

Create a Lambda function to compute custom metrics:

```python
# save as eval_lambda.py
import json
from typing import List, Dict

def lambda_handler(event, context):
    """
    Custom evaluation metrics

    Input: List of samples with model responses
    Output: Scores and metrics for each sample
    """
    results = []

    for sample in event:
        sample_id = sample.get("id")
        question = sample.get("question", "")
        expected = sample.get("expected_answer", "")
        response = sample.get("response", "")

        # Custom scoring logic
        # Example: Check if expected keywords are in response
        expected_keywords = expected.lower().split()
        response_lower = response.lower()

        keyword_matches = sum(1 for kw in expected_keywords if kw in response_lower)
        keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0

        # Length check
        response_length = len(response.split())
        length_ok = 10 <= response_length <= 100

        # Aggregate score
        aggregate_score = (keyword_score * 0.7) + (0.3 if length_ok else 0)

        results.append({
            "id": sample_id,
            "aggregate_reward_score": aggregate_score,
            "metrics": {
                "keyword_match_score": keyword_score,
                "response_length": response_length,
                "length_appropriate": length_ok
            }
        })

    return results
```

### 6.2 Deploy Lambda and Run BYOM Evaluation

```bash
# Deploy Lambda (simplified - use AWS SAM or Console for full deployment)
aws lambda create-function \
    --function-name NovaEvalMetrics \
    --runtime python3.12 \
    --role arn:aws:iam::123456789012:role/lambda-execution-role \
    --handler eval_lambda.lambda_handler \
    --zip-file fileb://eval_lambda.zip
```

```python
# Run BYOM evaluation
byom_result = evaluator.evaluate(
    job_name="eval-byom-custom-metrics",
    eval_task=EvaluationTask.GEN_QA,
    data_s3_path=eval_s3_path,
    model_path=checkpoint_path,
    processor={
        "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:NovaEvalMetrics"
    },
    overrides={"max_new_tokens": 2048}
)

print(f"✅ BYOM evaluation started: {byom_result.job_id}")
```

---

## Step 7: LLM-as-Judge Evaluation

### 7.1 Prepare LLM-as-Judge Dataset

```python
import json

# Create evaluation dataset with judge prompts
judge_data = [
    {
        "id": "sample_1",
        "question": "Explain quantum computing.",
        "response": "Quantum computing uses quantum bits that can be in superposition...",
        "judge_criteria": "Rate the clarity, accuracy, and completeness of the explanation on a scale of 1-10."
    },
    {
        "id": "sample_2",
        "question": "What is blockchain?",
        "response": "Blockchain is a distributed ledger technology...",
        "judge_criteria": "Evaluate if the explanation is accessible to non-technical audiences."
    }
] * 10

# Save to JSONL
with open("llm_judge_data.jsonl", "w") as f:
    for item in judge_data:
        f.write(json.dumps(item) + "\n")

# Upload
loader = JSONLDatasetLoader()
loader.load("llm_judge_data.jsonl")
judge_s3_path = loader.save_data(f"s3://{S3_BUCKET}/{S3_PREFIX}/judge-eval.jsonl")
```

### 7.2 Run LLM-as-Judge Evaluation

```python
# Evaluate using LLM judge
judge_result = evaluator.evaluate(
    job_name="eval-llm-judge",
    eval_task=EvaluationTask.LLM_JUDGE,
    data_s3_path=judge_s3_path,
    model_path=checkpoint_path,
    overrides={"max_new_tokens": 2048}
)

print(f"✅ LLM-as-Judge evaluation started: {judge_result.job_id}")
```

---

## Step 8: Monitor Evaluation Progress

```python
# View evaluation logs (while running)
evaluator.get_logs(limit=50, start_from_head=False)

# After completion
from amzn_nova_forge import CloudWatchLogMonitor, Platform

monitor = CloudWatchLogMonitor.from_job_id(
    job_id=byod_result.job_id,
    platform=Platform.SMTJ
)

monitor.show_logs(limit=100, start_from_head=True)
```

---

## Step 9: Analyze Evaluation Results

### 9.1 Download Results from S3

```python
import boto3
import json

# Download evaluation results
s3 = boto3.client('s3')
output_path = byod_result.eval_output_path  # e.g., s3://bucket/path/

# Parse S3 path
bucket = output_path.replace("s3://", "").split("/")[0]
prefix = "/".join(output_path.replace("s3://", "").split("/")[1:])

# List result files
response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
for obj in response.get('Contents', []):
    print(f"   {obj['Key']}")

# Download results
results_key = f"{prefix}results.json"
s3.download_file(bucket, results_key, "eval_results.json")

# Load and analyze
with open("eval_results.json") as f:
    results = json.load(f)

print("📊 Evaluation Results:")
print(json.dumps(results, indent=2))
```

### 9.2 Interpret MMLU Results

```python
# MMLU results structure
{
    "accuracy": 0.72,  # Overall accuracy (72%)
    "subject_scores": {
        "abstract_algebra": 0.68,
        "anatomy": 0.75,
        "astronomy": 0.71,
        # ... more subjects
    },
    "num_samples": 14042,
    "model": "custom-nova-lite2"
}

# Compare with base model
base_accuracy = 0.65
custom_accuracy = 0.72
improvement = ((custom_accuracy - base_accuracy) / base_accuracy) * 100

print(f"📈 Improvement: {improvement:.1f}% over base model")
```

### 9.3 Compare Multiple Models

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load multiple evaluation results
results = {
    "Base Model": {"mmlu": 0.65, "hellaswag": 0.71, "gsm8k": 0.43},
    "SFT Model": {"mmlu": 0.72, "hellaswag": 0.75, "gsm8k": 0.58},
    "DPO Model": {"mmlu": 0.70, "hellaswag": 0.78, "gsm8k": 0.56}
}

# Create comparison DataFrame
df = pd.DataFrame(results).T

# Plot
df.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.legend(title="Benchmarks")
plt.tight_layout()
plt.savefig("eval_comparison.png")
plt.show()

print("✅ Comparison chart saved: eval_comparison.png")
```

---

## Step 10: Dry Run Evaluation

Test configuration without running:

```python
# Dry run mode
eval_result = evaluator.evaluate(
    job_name="eval-dry-run",
    eval_task=EvaluationTask.MMLU,
    dry_run=True,  # Only validates, doesn't start
    overrides={"max_new_tokens": 2048}
)

print("✅ Dry run completed - configuration is valid!")
```

---

## Common Evaluation Patterns

### Pattern 1: Quick Quality Check

```python
# Fast evaluation on small subset
quick_eval = evaluator.evaluate(
    job_name="quick-eval",
    eval_task=EvaluationTask.MMLU,
    subtask="stem",  # Smaller subset
    overrides={"max_new_tokens": 1024}
)
```

### Pattern 2: Comprehensive Benchmark Suite

```python
# Run multiple benchmarks
benchmarks = [
    EvaluationTask.MMLU,
    EvaluationTask.HELLASWAG,
    EvaluationTask.ARC_CHALLENGE,
    EvaluationTask.GSM8K
]

results = {}
for benchmark in benchmarks:
    result = evaluator.evaluate(
        job_name=f"eval-{benchmark.value}",
        eval_task=benchmark,
        model_path=checkpoint_path,
        overrides={"max_new_tokens": 2048}
    )
    results[benchmark.value] = result.job_id
    print(f"✅ {benchmark.value} started: {result.job_id}")
```

### Pattern 3: Compare Before/After Training

```python
# Evaluate base model
base_result = evaluator.evaluate(
    job_name="eval-base-model",
    eval_task=EvaluationTask.MMLU
    # No model_path = uses base model
)

# Evaluate trained model
trained_result = evaluator.evaluate(
    job_name="eval-trained-model",
    eval_task=EvaluationTask.MMLU,
    model_path=training_result.model_artifacts.checkpoint_s3_path
)

print("✅ Base and trained model evaluations started")
print(f"   Base: {base_result.job_id}")
print(f"   Trained: {trained_result.job_id}")
```

---

## Quick Reference

### Evaluation Tasks

```python
from amzn_nova_forge import EvaluationTask

# Available tasks
EvaluationTask.MMLU
EvaluationTask.HELLASWAG
EvaluationTask.ARC_CHALLENGE
EvaluationTask.ARC_EASY
EvaluationTask.TRUTHFULQA
EvaluationTask.GSM8K
EvaluationTask.HUMANEVAL
EvaluationTask.GEN_QA          # BYOD
EvaluationTask.LLM_JUDGE       # LLM-as-Judge
```

### Minimal Evaluation Example

```python
from amzn_nova_forge import *

# Setup
eval_runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=1
)

# Create evaluator
evaluator = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.EVALUATION,
    infra=eval_runtime,
    output_s3_path="s3://bucket/eval-output/"
)

# Run evaluation
result = evaluator.evaluate(
    job_name="mmlu-eval",
    eval_task=EvaluationTask.MMLU,
    overrides={"max_new_tokens": 2048}
)

print(f"✅ Evaluation started: {result.job_id}")
```

---

## Next Steps

✅ **Evaluation complete!** Now you can:

- **Journey 8: Bedrock Deployment** - Deploy to Amazon Bedrock
- **Journey 9: SageMaker Deployment** - Deploy to SageMaker
- **Iterate on Training** - Use insights to improve your model

---

## Resources

- [Nova Evaluation Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html)
- [Benchmark Descriptions](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-benchmark)
- [SDK Evaluation Spec](../amzn-nova-forge/docs/spec.md#evaluate)
- [Custom Metrics Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-byom.html)
