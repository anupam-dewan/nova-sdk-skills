# Journey 6b: RFT Multi-Turn (Conversational Reinforcement Fine-Tuning)

## Overview

**Purpose**: Train Nova models for multi-turn conversations with reward optimization
**Time**: 4-8 hours (longer due to conversation complexity)
**Prerequisites**: Journey 1 (Setup), Journey 2 (Data), Journey 3 (SFT), Journey 6a (RFT Single-Turn)
**Outputs**: Conversation-optimized model with reward-based learning
**Platform**: SMHP (SMHP for production)
**Difficulty**: ⭐⭐⭐ Advanced

---

## What You'll Learn

- Understand multi-turn vs single-turn RFT
- Design conversation-aware reward functions
- Structure multi-turn training data
- Configure conversational RFT training
- Evaluate conversation quality improvements
- Handle context and turn management

---

## What is Multi-Turn RFT?

**Multi-Turn RFT** extends RFT to conversational scenarios where context matters across turns:

- **Input**: Conversation histories + Reward function
- **Purpose**: Optimize multi-turn dialogue quality
- **Use Cases**: Customer support, tutoring, interactive assistants, therapy bots
- **Training Time**: 3-6 hours (more complex than single-turn)

### Multi-Turn vs Single-Turn RFT

| Aspect | Single-Turn RFT | Multi-Turn RFT |
|--------|----------------|----------------|
| **Context** | Single prompt → response | Full conversation history |
| **Reward** | Per response | Per conversation or turn |
| **Complexity** | Simpler | More complex |
| **Use Case** | Q&A, code, math | Dialogue, support, tutoring |
| **Training Time** | 2-4 hours | 4-8 hours |

### When to Use Multi-Turn RFT

✅ **Use Multi-Turn RFT when:**

- Building conversational agents
- Context from previous turns matters
- Need to optimize dialogue flow
- Reward depends on conversation history

❌ **Use Single-Turn RFT when:**

- Each query is independent
- No conversation history needed
- Simpler and faster to train

---

## Step 1: Design Conversation Reward Function

Multi-turn rewards can evaluate per-turn or per-conversation.

### Example: Customer Support Reward

```python
# reward_function_multiturn.py

def calculate_conversation_reward(conversation: list[dict]) -> float:
    """
    Reward function for customer support conversations.

    Args:
        conversation: List of turns [{"role": "user/assistant", "content": "..."}]

    Returns:
        Reward score 0.0-1.0
    """
    score = 0.0

    # 1. Resolution quality (0.4)
    if is_issue_resolved(conversation):
        score += 0.4

    # 2. Empathy and tone (0.3)
    empathy_score = evaluate_empathy(conversation)
    score += empathy_score * 0.3

    # 3. Efficiency (0.2)
    num_turns = len([t for t in conversation if t["role"] == "assistant"])
    efficiency = max(0, 1 - (num_turns - 3) / 10)  # Optimal 3-5 turns
    score += efficiency * 0.2

    # 4. Factual accuracy (0.1)
    if all_responses_accurate(conversation):
        score += 0.1

    return score

def is_issue_resolved(conversation: list[dict]) -> bool:
    """Check if customer issue was resolved."""
    # Look for resolution indicators in final turns
    last_turns = conversation[-3:]
    resolution_keywords = ["solved", "resolved", "thank you", "that worked"]

    for turn in last_turns:
        if turn["role"] == "user":
            if any(kw in turn["content"].lower() for kw in resolution_keywords):
                return True
    return False

def evaluate_empathy(conversation: list[dict]) -> float:
    """Score empathy in assistant responses (0.0-1.0)."""
    assistant_turns = [t for t in conversation if t["role"] == "assistant"]

    empathy_indicators = [
        "understand", "sorry", "appreciate", "help", "frustrating"
    ]

    empathy_count = sum(
        any(word in turn["content"].lower() for word in empathy_indicators)
        for turn in assistant_turns
    )

    return min(1.0, empathy_count / len(assistant_turns))
```

### Example: Tutoring Reward

```python
def calculate_conversation_reward(conversation: list[dict]) -> float:
    """Reward function for educational tutoring."""
    score = 0.0

    # 1. Student understanding (0.5)
    if student_demonstrates_understanding(conversation):
        score += 0.5

    # 2. Pedagogical approach (0.3)
    # Reward asking questions, not giving answers
    pedagogy_score = evaluate_pedagogy(conversation)
    score += pedagogy_score * 0.3

    # 3. Patience and encouragement (0.2)
    if shows_encouragement(conversation):
        score += 0.2

    return score

def student_demonstrates_understanding(conversation: list[dict]) -> bool:
    """Check if student shows understanding progression."""
    # Look for improved responses over time
    student_turns = [t for t in conversation if t["role"] == "user"]

    if len(student_turns) < 2:
        return False

    # Simple heuristic: later responses are more detailed/correct
    return len(student_turns[-1]["content"]) > len(student_turns[0]["content"])

def evaluate_pedagogy(conversation: list[dict]) -> float:
    """Score use of questions vs direct answers."""
    assistant_turns = [t for t in conversation if t["role"] == "assistant"]

    questions = sum("?" in turn["content"] for turn in assistant_turns)
    return min(1.0, questions / max(1, len(assistant_turns)))
```

### Example: Per-Turn Reward

```python
def calculate_turn_reward(conversation_history: list[dict], current_turn: str) -> float:
    """
    Reward each turn based on full context.

    Args:
        conversation_history: Previous turns
        current_turn: Current assistant response to evaluate

    Returns:
        Reward for this specific turn
    """
    score = 0.0

    # 1. Contextual relevance (0.4)
    if is_contextually_relevant(conversation_history, current_turn):
        score += 0.4

    # 2. Information accuracy (0.3)
    if is_factually_correct(current_turn):
        score += 0.3

    # 3. Engagement (0.2)
    if is_engaging(current_turn):
        score += 0.2

    # 4. Appropriate length (0.1)
    if 50 < len(current_turn.split()) < 150:  # Not too short/long
        score += 0.1

    return score
```

---

## Step 2: Deploy Multi-Turn Lambda

```python
import boto3

lambda_client = boto3.client('lambda')

# Deploy conversation reward Lambda
with open('reward-lambda-multiturn.zip', 'rb') as f:
    response = lambda_client.create_function(
        FunctionName='nova-reward-conversation',
        Runtime='python3.11',
        Role='arn:aws:iam::YOUR-ACCOUNT:role/LambdaExecutionRole',
        Handler='reward_function_multiturn.calculate_conversation_reward',
        Code={'ZipFile': f.read()},
        Timeout=120,  # Longer for conversation analysis
        MemorySize=1024
    )

lambda_arn = response['FunctionArn']
print(f"✅ Multi-turn Lambda deployed: {lambda_arn}")
```

### Test Lambda with Conversation

```python
# Test with sample conversation
test_conversation = [
    {"role": "user", "content": "My order hasn't arrived yet."},
    {"role": "assistant", "content": "I understand your frustration. Let me check your order status."},
    {"role": "user", "content": "Thanks, order #12345"},
    {"role": "assistant", "content": "I see it's in transit. Expected delivery tomorrow."},
    {"role": "user", "content": "Perfect, thank you!"}
]

result = lambda_client.invoke(
    FunctionName='nova-reward-conversation',
    Payload=json.dumps({"conversation": test_conversation})
)

reward = json.loads(result['Payload'].read())
print(f"✅ Test conversation reward: {reward}")
```

---

## Step 3: Prepare Multi-Turn Training Data

Format conversations with full history per example.

### Data Format

```json
{
  "conversation": [
    {"role": "user", "content": "I need help resetting my password"},
    {"role": "assistant", "content": "I can help with that. What's your email?"},
    {"role": "user", "content": "user@example.com"},
    {"role": "assistant", "content": "I've sent a reset link to that email."}
  ]
}
```

Full example file:

```json
{"conversation": [{"role": "user", "content": "How do I upgrade?"}, {"role": "assistant", "content": "Go to Settings > Billing"}, {"role": "user", "content": "Thanks!"}]}
{"conversation": [{"role": "user", "content": "Problem with payment"}, {"role": "assistant", "content": "What error do you see?"}, {"role": "user", "content": "Card declined"}, {"role": "assistant", "content": "Let me check your billing..."}]}
```

Upload to S3:

```python
s3 = boto3.client('s3')
s3.upload_file(
    'conversations.jsonl',
    'my-training-bucket',
    'data/rft-multiturn/conversations.jsonl'
)

print("✅ Multi-turn conversations uploaded")
```

---

## Step 4: Configure Runtime

```python
from amzn_nova_forge import *

# Multi-turn RFT benefits from more memory
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=8  # More instances for conversation complexity
)

# Or SMHP for production
runtime = SMHPRuntimeManager(
    cluster_name="my-cluster",
    instance_type="ml.p5.48xlarge",
    instance_count=8
)

print("✅ Runtime configured for multi-turn RFT")
```

---

## Step 5: Initialize Multi-Turn RFT

```python
from amzn_nova_forge import *

# Initialize customizer for multi-turn RFT
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,  # Or SFT checkpoint
    method=TrainingMethod.RFT_MULTITURN,  # Multi-turn variant
    infra=runtime,
    data_s3_path="s3://my-training-bucket/data/rft-multiturn/conversations.jsonl",
    output_s3_path="s3://my-training-bucket/output/rft-multiturn/",
    reward_lambda_arn="arn:aws:lambda:us-east-1:123:function:nova-reward-conversation"
)

print("✅ Multi-Turn RFT Customizer initialized")
print(f"   Method: {customizer.method}")
print(f"   Reward Lambda: {customizer.reward_lambda_arn}")
```

---

## Step 6: Configure Multi-Turn Hyperparameters

```python
rft_multiturn_config = {
    # Learning rate (even lower for multi-turn)
    "lr": 5e-7,  # More conservative than single-turn

    # Batch size
    "per_device_train_batch_size": 1,  # Conversations are longer
    "gradient_accumulation_steps": 32,  # Effective batch = 32

    # Training duration
    "num_train_epochs": 2,  # Fewer epochs (conversations are richer)

    # Multi-turn specific
    "max_conversation_length": 10,  # Max turns per conversation
    "num_generations_per_conversation": 2,  # Samples per conversation
    "temperature": 0.9,
    "top_p": 0.95,

    # Context handling
    "max_seq_length": 4096,  # Longer for conversation history

    # Reward computation
    "reward_scaling": "standardize",
    "reward_aggregation": "mean",  # How to combine turn rewards

    # Optimization
    "warmup_ratio": 0.1,  # More warmup for stability
    "weight_decay": 0.01,

    # Logging
    "logging_steps": 5,
    "save_steps": 250
}

print("✅ Multi-turn RFT hyperparameters configured")
```

### Key Multi-Turn Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `max_conversation_length` | Max turns to consider | 8-12 |
| `num_generations_per_conversation` | Samples per conversation | 2-4 |
| `max_seq_length` | Context window | 4096-8192 |
| `reward_aggregation` | Combine turn rewards | mean, sum, last |

---

## Step 7: Start Multi-Turn RFT Training

```python
# Start multi-turn RFT training
result = customizer.train(
    job_name="support-multiturn-rft-v1",
    overrides=rft_multiturn_config,
    dry_run=False
)

print("✅ Multi-Turn RFT Training started!")
print(f"   Job ID: {result.job_id}")
print(f"   Expected duration: 4-8 hours")
```

---

## Step 8: Monitor Multi-Turn Training

```python
# Monitor training
print("📋 Multi-Turn RFT Logs:")
customizer.get_logs(limit=50)
```

### Expected Output

```yaml
Epoch 1/2 | Step 50/500 | Avg Conv Reward: 0.42 | Turns: 5.3 | Loss: 1.432
Epoch 1/2 | Step 100/500 | Avg Conv Reward: 0.56 | Turns: 5.1 | Loss: 1.287
Epoch 1/2 | Step 150/500 | Avg Conv Reward: 0.68 | Turns: 4.8 | Loss: 1.143
```

**What to watch:**

- **Avg Conv Reward**: Conversation-level reward (should increase)
- **Turns**: Average conversation length
- **Loss**: Policy loss

---

## Step 9: Evaluate Conversation Quality

### Compare Pre vs Post Training

```python
# Test conversations
test_scenarios = [
    {"opening": "I have a billing question"},
    {"opening": "How do I cancel my subscription?"},
    {"opening": "My account is locked"}
]

# Run full conversations with both models
for scenario in test_scenarios:
    print(f"\n--- Scenario: {scenario['opening']} ---")

    # Pre-RFT conversation
    pre_conversation = run_conversation(
        model=sft_checkpoint,
        opening=scenario["opening"],
        max_turns=5
    )

    # Post-RFT conversation
    post_conversation = run_conversation(
        model=result.model_s3_path,
        opening=scenario["opening"],
        max_turns=5
    )

    # Evaluate with reward function
    pre_reward = calculate_conversation_reward(pre_conversation)
    post_reward = calculate_conversation_reward(post_conversation)

    print(f"Pre-RFT Reward: {pre_reward:.2f}")
    print(f"Post-RFT Reward: {post_reward:.2f}")
    print(f"Improvement: {(post_reward - pre_reward) * 100:.1f}%")
```

---

## Common Issues & Solutions

### Issue 1: Conversation Context Truncated

**Error:**

```sh
Warning: Conversation exceeds max_seq_length, truncating early turns
```

**Solutions:**

```python
# 1. Increase sequence length
rft_multiturn_config["max_seq_length"] = 8192  # Was 4096

# 2. Limit conversation length
rft_multiturn_config["max_conversation_length"] = 8  # Was 10

# 3. Use sliding window for very long conversations
rft_multiturn_config["conversation_window"] = "sliding"
```

### Issue 2: Reward Function Too Slow

**Problem:** Lambda timeouts with long conversations

**Solutions:**

```python
# 1. Increase Lambda timeout
lambda_client.update_function_configuration(
    FunctionName='nova-reward-conversation',
    Timeout=300,  # 5 minutes
    MemorySize=2048  # More memory
)

# 2. Optimize reward function (cache, simplify)
# 3. Use per-turn rewards instead of per-conversation
```

### Issue 3: Training Instability

**Symptom:** Reward oscillates or decreases

**Solutions:**

```python
# 1. Lower learning rate
rft_multiturn_config["lr"] = 1e-7  # Was 5e-7

# 2. Increase warmup
rft_multiturn_config["warmup_ratio"] = 0.15  # Was 0.1

# 3. Reduce generation diversity
rft_multiturn_config["temperature"] = 0.7  # Was 0.9
```

### Issue 4: Model Responses Too Short/Long

**Problem:** Conversations end too quickly or drag on

**Solution:** Add length penalty to reward:

```python
def calculate_conversation_reward(conversation):
    score = base_reward(conversation)

    # Penalize too short
    num_turns = len([t for t in conversation if t["role"] == "assistant"])
    if num_turns < 3:
        score *= 0.8

    # Penalize too long
    if num_turns > 8:
        score *= 0.9

    return score
```

---

## Quick Reference

### Minimal Multi-Turn RFT Example

```python
from amzn_nova_forge import *

# 1. Deploy conversation reward Lambda
lambda_arn = "arn:aws:lambda:us-east-1:123:function:conversation-reward"

# 2. Configure runtime
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=8
)

# 3. Initialize multi-turn RFT
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.RFT_MULTITURN,
    infra=runtime,
    data_s3_path="s3://bucket/conversations.jsonl",
    output_s3_path="s3://bucket/output/",
    reward_lambda_arn=lambda_arn
)

# 4. Train
result = customizer.train(
    job_name="my-multiturn-rft",
    overrides={
        "lr": 5e-7,
        "max_conversation_length": 10,
        "num_generations_per_conversation": 2
    }
)

print(f"Multi-turn RFT complete: {result.model_s3_path}")
```

### Multi-Turn RFT Best Practices

✅ **Do:**

- Start with SFT on conversations first
- Design rewards that consider full context
- Use lower learning rates than single-turn
- Test Lambda with long conversations
- Monitor conversation length in logs
- Evaluate with real conversation scenarios

❌ **Don't:**

- Skip SFT baseline (harder to optimize from scratch)
- Use rewards that only look at final turn
- Exceed max sequence length (truncation issues)
- Use high learning rates (unstable)
- Forget to test edge cases (very short/long conversations)

---

## Next Steps

**After Multi-Turn RFT:**

1. **Evaluate conversation quality** - Journey 7 with dialogue metrics
2. **Deploy conversational model** - Journey 8 or 9
3. **Monitor in production** - Journey 10 for real user conversations

**Related Journeys:**

- **Journey 3**: SFT Training (baseline for conversations)
- **Journey 6a**: RFT Single-Turn (simpler variant)
- **Journey 7**: Evaluation (conversation metrics)

---

## Resources

- **Conversation Design**: AWS best practices for dialogue systems
- **Lambda Optimization**: https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html
- **Dialogue Evaluation Metrics**: Reference documentation

---

**💡 Pro Tip**: Multi-turn RFT is powerful but complex. Start with single-turn RFT (Journey 6a) first to understand reward-based training, then move to multi-turn when you need conversation-level optimization!
