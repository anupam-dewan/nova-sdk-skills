# Journey 6a: RFT Single-Turn (Reinforcement Fine-Tuning)

## Overview
**Purpose**: Train Nova models with reward functions for objective optimization
**Time**: 3-5 hours (including training and reward computation)
**Prerequisites**: Journey 1 (Setup), Journey 2 (Data), Journey 3 (SFT recommended)
**Outputs**: Reward-optimized model for specific objectives
**Platform**: SMTJ or SMHP

---

## What You'll Learn

- Understand when RFT improves over SFT
- Design and implement reward functions
- Deploy reward functions as Lambda
- Configure RFT training jobs
- Optimize hyperparameters for reward-based learning
- Evaluate reward-driven improvements

---

## What is RFT?

**Reinforcement Fine-Tuning (RFT)** optimizes models using custom reward signals:

- **Input**: Prompts + Reward function (Lambda)
- **Purpose**: Optimize for specific objectives (accuracy, format, style, etc.)
- **Use Cases**: Code generation, math problems, structured outputs, factuality
- **Training Time**: 2-4 hours (depends on reward complexity)

### RFT vs SFT vs DPO

| Method | Input Data | Optimization Goal | Best For |
|--------|-----------|------------------|----------|
| **SFT** | Q&A pairs | Imitate examples | General tasks |
| **DPO** | Preference pairs (A>B) | Align with preferences | Safety, style |
| **RFT** | Prompts + Reward function | Maximize reward | Objective metrics |

### When to Use RFT

✅ **Use RFT when:**
- You have an objective function (e.g., code passes tests, math is correct)
- You want to optimize specific metrics (accuracy, format compliance)
- SFT alone doesn't achieve desired performance
- You can evaluate outputs programmatically

❌ **Use SFT instead when:**
- You just need to imitate examples
- No clear objective function exists
- Human judgment is needed (use DPO)

### Typical Workflow

```
Base Model → SFT (optional but recommended) → RFT → Reward-Optimized Model
```

**Example**: Code Generation
1. **SFT**: Learn code syntax and patterns
2. **RFT**: Optimize for test passage and correctness

---

## Step 1: Define Your Reward Function

The reward function scores model outputs (0.0 = bad, 1.0 = perfect).

### Example: Code Correctness Reward

```python
# reward_function.py

def calculate_reward(prompt: str, response: str) -> float:
    """
    Reward function for code generation.
    Returns 1.0 if code passes tests, 0.0 otherwise.
    """
    try:
        # Extract code from response
        code = extract_code_block(response)

        # Run test cases
        test_results = run_tests(code, prompt)

        # Reward based on test passage
        passed = sum(test_results)
        total = len(test_results)
        reward = passed / total

        return reward
    except Exception as e:
        return 0.0  # Failed to execute

def extract_code_block(response: str) -> str:
    """Extract code from markdown response."""
    import re
    match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
    return match.group(1) if match else response

def run_tests(code: str, prompt: str) -> list[bool]:
    """Run test cases against generated code."""
    # Parse test cases from prompt
    tests = parse_tests_from_prompt(prompt)

    results = []
    for test_input, expected_output in tests:
        try:
            # Execute code with test input
            actual_output = exec_code(code, test_input)
            results.append(actual_output == expected_output)
        except:
            results.append(False)

    return results
```

### Example: Math Accuracy Reward

```python
def calculate_reward(prompt: str, response: str) -> float:
    """Reward function for math problems."""
    try:
        # Extract answer from response
        answer = extract_answer(response)

        # Get ground truth from prompt
        ground_truth = extract_ground_truth(prompt)

        # Check if correct
        if answer == ground_truth:
            return 1.0
        else:
            return 0.0
    except:
        return 0.0

def extract_answer(response: str) -> float:
    """Extract numeric answer from response."""
    import re
    match = re.search(r'Final Answer: ([-+]?\d*\.?\d+)', response)
    if match:
        return float(match.group(1))
    return None
```

### Example: Format Compliance Reward

```python
def calculate_reward(prompt: str, response: str) -> float:
    """Reward function for JSON format compliance."""
    import json

    try:
        # Try parsing as JSON
        data = json.loads(response)

        # Check required fields
        required_fields = ["name", "age", "email"]
        has_all_fields = all(field in data for field in required_fields)

        if has_all_fields:
            return 1.0
        else:
            return 0.5  # Partial credit for valid JSON
    except:
        return 0.0  # Not valid JSON
```

---

## Step 2: Deploy Reward Function as Lambda

RFT requires reward functions deployed as AWS Lambda.

### 2.1 Package Lambda Function

```bash
# Create deployment package
mkdir reward-lambda
cd reward-lambda

# Copy reward function
cp ../reward_function.py .

# Install dependencies
pip install -r requirements.txt -t .

# Create ZIP
zip -r reward-lambda.zip .
```

### 2.2 Deploy to Lambda

```python
import boto3

lambda_client = boto3.client('lambda')

# Create Lambda function
response = lambda_client.create_function(
    FunctionName='nova-reward-code-correctness',
    Runtime='python3.11',
    Role='arn:aws:iam::YOUR-ACCOUNT:role/LambdaExecutionRole',
    Handler='reward_function.calculate_reward',
    Code={'ZipFile': open('reward-lambda.zip', 'rb').read()},
    Timeout=60,  # Reward computation timeout
    MemorySize=512
)

lambda_arn = response['FunctionArn']
print(f"✅ Lambda deployed: {lambda_arn}")
```

### 2.3 Test Lambda

```python
# Test reward function
test_payload = {
    "prompt": "Write a function that adds two numbers",
    "response": "```python\ndef add(a, b):\n    return a + b\n```"
}

result = lambda_client.invoke(
    FunctionName='nova-reward-code-correctness',
    Payload=json.dumps(test_payload)
)

reward = json.loads(result['Payload'].read())
print(f"✅ Test reward: {reward}")
```

---

## Step 3: Prepare RFT Training Data

RFT uses prompts without responses (model generates during training).

### Data Format

```json
{"prompt": "Write a Python function that reverses a string"}
{"prompt": "Create a function to check if a number is prime"}
{"prompt": "Implement binary search in Python"}
```

Upload to S3:

```python
import boto3

s3 = boto3.client('s3')
s3.upload_file(
    'rft_prompts.jsonl',
    'my-training-bucket',
    'data/rft/code_prompts.jsonl'
)

print("✅ RFT prompts uploaded")
```

---

## Step 4: Configure Runtime

```python
from amzn_nova_customization_sdk import *

# Option A: SMTJ (simpler)
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=4
)

# Option B: SMHP (production)
runtime = SMHPRuntimeManager(
    cluster_name="my-cluster",
    instance_type="ml.p5.48xlarge",
    instance_count=4
)

print("✅ Runtime configured")
```

---

## Step 5: Initialize RFT Training

```python
from amzn_nova_customization_sdk import *

# Initialize customizer for RFT
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,  # Or use SFT checkpoint
    method=TrainingMethod.RFT_SINGLETURN,
    infra=runtime,
    data_s3_path="s3://my-training-bucket/data/rft/code_prompts.jsonl",
    output_s3_path="s3://my-training-bucket/output/rft/",
    reward_lambda_arn="arn:aws:lambda:us-east-1:123:function:nova-reward-code-correctness"
)

print("✅ RFT Customizer initialized")
print(f"   Method: {customizer.method}")
print(f"   Reward Lambda: {customizer.reward_lambda_arn}")
```

---

## Step 6: Configure RFT Hyperparameters

```python
rft_config = {
    # Learning rate
    "lr": 1e-6,  # Lower than SFT (reward-based learning is sensitive)

    # Batch size
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,  # Effective batch = 32

    # Training duration
    "num_train_epochs": 3,

    # RFT-specific parameters
    "num_generations_per_prompt": 4,  # Generate N outputs per prompt
    "temperature": 0.8,                # Sampling temperature
    "top_p": 0.9,                      # Nucleus sampling

    # Reward normalization
    "reward_scaling": "standardize",   # Normalize rewards

    # Optimization
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,

    # Logging
    "logging_steps": 10,
    "save_steps": 500
}

print("✅ RFT hyperparameters configured")
```

### Key RFT Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `lr` | Learning rate (lower than SFT) | 5e-7 to 5e-6 |
| `num_generations_per_prompt` | Samples per prompt | 4-8 |
| `temperature` | Sampling randomness | 0.7-1.0 |
| `reward_scaling` | How to normalize rewards | standardize, minmax |

---

## Step 7: Start RFT Training

```python
# Start RFT training
result = customizer.train(
    job_name="code-rft-v1",
    overrides=rft_config,
    dry_run=False
)

print("✅ RFT Training started!")
print(f"   Job ID: {result.job_id}")
print(f"   Status: {result.status}")
print(f"   Lambda: {customizer.reward_lambda_arn}")
```

---

## Step 8: Monitor RFT Training

### View Training Logs

```python
# Monitor RFT progress
print("📋 RFT Training Logs:")
customizer.get_logs(limit=50)
```

### Expected Log Output

```
Epoch 1/3 | Step 100/1000 | Avg Reward: 0.34 | Loss: 1.234
Epoch 1/3 | Step 200/1000 | Avg Reward: 0.48 | Loss: 1.087
Epoch 1/3 | Step 300/1000 | Avg Reward: 0.61 | Loss: 0.943
```

**What to watch:**
- **Avg Reward**: Should increase over time (model learning to maximize reward)
- **Loss**: Policy loss (should decrease)
- **Lambda Invocations**: Check CloudWatch for reward computation

### Monitor Lambda Invocations

```python
import boto3

# Check Lambda metrics
cloudwatch = boto3.client('cloudwatch')
metrics = cloudwatch.get_metric_statistics(
    Namespace='AWS/Lambda',
    MetricName='Invocations',
    Dimensions=[{'Name': 'FunctionName', 'Value': 'nova-reward-code-correctness'}],
    StartTime=training_start_time,
    EndTime=now,
    Period=300,  # 5 minutes
    Statistics=['Sum']
)

print(f"Lambda invocations: {sum(m['Sum'] for m in metrics['Datapoints'])}")
```

---

## Step 9: Evaluate RFT Results

### Compare Pre-RFT vs Post-RFT

```python
# Test on held-out prompts
test_prompts = [
    "Write a function to find the longest common substring",
    "Implement a binary tree traversal",
    "Create a function to validate email addresses"
]

# Evaluate SFT model (before RFT)
sft_rewards = evaluate_with_reward_function(
    model=sft_checkpoint,
    prompts=test_prompts,
    reward_function=calculate_reward
)

# Evaluate RFT model (after RFT)
rft_rewards = evaluate_with_reward_function(
    model=result.model_s3_path,
    prompts=test_prompts,
    reward_function=calculate_reward
)

print(f"SFT Avg Reward: {np.mean(sft_rewards):.2f}")
print(f"RFT Avg Reward: {np.mean(rft_rewards):.2f}")
print(f"Improvement: {(np.mean(rft_rewards) - np.mean(sft_rewards)) * 100:.1f}%")
```

---

## Common Issues & Solutions

### Issue 1: Lambda Timeout

**Error:**
```
Lambda timeout: Function exceeded 60 seconds
```

**Solutions:**
```python
# Increase Lambda timeout
lambda_client.update_function_configuration(
    FunctionName='nova-reward-code-correctness',
    Timeout=300  # 5 minutes
)

# Or optimize reward function (cache, parallelize)
```

### Issue 2: Reward Not Increasing

**Symptom:** Avg reward stays flat

**Solutions:**
```python
# 1. Increase generation diversity
rft_config["num_generations_per_prompt"] = 8  # Was 4
rft_config["temperature"] = 1.0  # Was 0.8

# 2. Lower learning rate
rft_config["lr"] = 5e-7  # Was 1e-6

# 3. Check reward function
# Test manually with sample outputs
```

### Issue 3: Reward Too Easy/Hard

**Problem:** All rewards are 1.0 or 0.0

**Solution:** Add partial rewards:
```python
def calculate_reward(prompt, response):
    # Instead of binary 0.0 or 1.0
    score = 0.0

    # Partial credit for syntax (0.3)
    if is_valid_python(response):
        score += 0.3

    # Partial credit for correct approach (0.4)
    if has_correct_logic(response):
        score += 0.4

    # Full credit for passing tests (0.3)
    if passes_tests(response):
        score += 0.3

    return score
```

### Issue 4: High Lambda Costs

**Problem:** Lambda invocations expensive

**Solutions:**
- Cache reward computations for identical outputs
- Reduce `num_generations_per_prompt` (4 instead of 8)
- Use cheaper Lambda configuration (less memory)
- Batch reward computations in Lambda

---

## Quick Reference

### Minimal RFT Example

```python
from amzn_nova_customization_sdk import *

# 1. Deploy reward Lambda (once)
lambda_arn = "arn:aws:lambda:us-east-1:123:function:my-reward"

# 2. Configure runtime
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=4
)

# 3. Initialize RFT
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.RFT_SINGLETURN,
    infra=runtime,
    data_s3_path="s3://bucket/rft/prompts.jsonl",
    output_s3_path="s3://bucket/output/rft/",
    reward_lambda_arn=lambda_arn
)

# 4. Train
result = customizer.train(
    job_name="my-rft",
    overrides={"lr": 1e-6, "num_generations_per_prompt": 4}
)

print(f"RFT complete: {result.model_s3_path}")
```

### RFT Best Practices

✅ **Do:**
- Start with SFT, then apply RFT
- Design reward functions with partial credit
- Test Lambda extensively before training
- Monitor avg reward during training
- Use lower learning rates than SFT

❌ **Don't:**
- Use binary rewards only (too sparse)
- Skip Lambda testing (expensive mistakes)
- Use high learning rates (unstable)
- Train without SFT baseline (harder to optimize)

---

## Next Steps

**After RFT:**
1. **Evaluate improvements** - Journey 7 (compare SFT vs RFT)
2. **Deploy** - Journey 8 (Bedrock) or Journey 9 (SageMaker)
3. **Consider Multi-Turn RFT** - Journey 6b for conversational tasks

**Related Journeys:**
- **Journey 3**: SFT Training (recommended before RFT)
- **Journey 7**: Evaluation (benchmark reward improvements)
- **Journey 6b**: RFT Multi-Turn (for conversations)

---

## Resources

- **Lambda Best Practices**: https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html
- **RFT vs DPO Guide**: `reference/training-methods.md`
- **Reward Function Examples**: SDK samples directory

---

**💡 Pro Tip**: RFT works best when you have a clear, programmatic reward function. If you're judging based on human preferences (e.g., "this response is better"), use DPO (Journey 5) instead!
