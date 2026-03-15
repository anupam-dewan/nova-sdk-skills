# Journey 8: Bedrock Deployment

## Overview
**Purpose**: Deploy custom Nova models to Amazon Bedrock for managed inference
**Time**: 15-30 minutes
**Prerequisites**: Journey 1 (Setup), Journey 3/4/5/6 (Trained model)
**Outputs**: Deployed Bedrock custom model with inference endpoint

---

## What You'll Learn

- Deploy trained models to Amazon Bedrock
- Choose between On-Demand (OD) and Provisioned Throughput (PT)
- Configure custom model settings
- Test inference via Bedrock API
- Monitor usage and costs
- Manage model lifecycle

---

## What is Bedrock Deployment?

**Amazon Bedrock** provides fully managed inference for Nova models:

### Benefits
- **Fully Managed**: No infrastructure to manage
- **Simple API**: Unified Bedrock API across models
- **Auto-Scaling**: Automatic scaling for On-Demand
- **Integration**: Works with Bedrock Agents, Knowledge Bases
- **Security**: Built-in encryption, VPC support

### Deployment Options

| Option | Best For | Pricing | Latency |
|--------|----------|---------|---------|
| **On-Demand (OD)** | Variable workloads | Pay per token | Standard |
| **Provisioned Throughput (PT)** | Consistent workloads | Fixed hourly rate | Lower |

---

## Step 1: Verify Prerequisites

### 1.1 Check Training Checkpoint

```python
from amzn_nova_customization_sdk import TrainingResult

# Load training result from previous training
training_result = TrainingResult.load("training_result.json")

checkpoint_path = training_result.model_artifacts.checkpoint_s3_path
print(f"📁 Checkpoint: {checkpoint_path}")
print(f"   Model: {training_result.model_type}")
print(f"   Method: {training_result.method}")
```

### 1.2 Verify IAM Permissions

Ensure your IAM role has Bedrock permissions:

```python
import boto3

bedrock = boto3.client('bedrock')

try:
    # Test Bedrock access
    bedrock.list_custom_models(maxResults=1)
    print("✅ Bedrock access verified")
except Exception as e:
    print(f"❌ Bedrock access failed: {e}")
    print("   Add bedrock:* permissions to your IAM role")
```

---

## Step 2: Deploy to Bedrock On-Demand

### 2.1 Configure Deployment

```python
from amzn_nova_customization_sdk import *

# Load customizer with trained model
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,  # Match your training method
    infra=SMTJRuntimeManager(
        instance_type="ml.p5.48xlarge",
        instance_count=1
    ),
    data_s3_path="s3://bucket/data.jsonl",  # Not used for deployment
    output_s3_path="s3://bucket/output/"
)

print("✅ Customizer initialized for deployment")
```

### 2.2 Deploy On-Demand Model

```python
# Deploy to Bedrock On-Demand
deployment_result = customizer.deploy(
    deploy_platform=DeployPlatform.BEDROCK_OD,
    endpoint_name="my-custom-nova-model-od",  # Unique model name
    job_result=training_result  # Automatically extracts checkpoint
)

print("\n🚀 Bedrock On-Demand deployment started!")
print(f"   Model Name: {deployment_result.endpoint.endpoint_name}")
print(f"   Status: {deployment_result.status}")
print(f"   Model ARN: {deployment_result.endpoint.endpoint_arn}")

# Save deployment info
model_name = deployment_result.endpoint.endpoint_name
model_arn = deployment_result.endpoint.endpoint_arn
```

### 2.3 Alternative: Deploy from Checkpoint Path

```python
# If you don't have TrainingResult, use checkpoint path directly
deployment_result = customizer.deploy(
    deploy_platform=DeployPlatform.BEDROCK_OD,
    endpoint_name="my-custom-nova-model-od",
    model_path="s3://bucket/checkpoints/my-model/"  # Direct path
)
```

---

## Step 3: Deploy to Bedrock Provisioned Throughput

### 3.1 When to Use Provisioned Throughput

**Use PT when**:
- Predictable, consistent workload
- Need guaranteed throughput
- Cost-effective for high volume
- Require lower latency

**Throughput Units**:
- Each unit provides fixed tokens/minute
- Minimum: 1 unit
- Can scale up as needed

### 3.2 Deploy Provisioned Model

```python
# Deploy to Bedrock Provisioned Throughput
pt_deployment_result = customizer.deploy(
    deploy_platform=DeployPlatform.BEDROCK_PT,
    endpoint_name="my-custom-nova-model-pt",
    unit_count=2,  # Number of throughput units
    job_result=training_result
)

print("\n🚀 Bedrock Provisioned Throughput deployment started!")
print(f"   Model Name: {pt_deployment_result.endpoint.endpoint_name}")
print(f"   Status: {pt_deployment_result.status}")
print(f"   Throughput Units: 2")
print(f"   Model ARN: {pt_deployment_result.endpoint.endpoint_arn}")

pt_model_name = pt_deployment_result.endpoint.endpoint_name
```

---

## Step 4: Monitor Deployment Status

### 4.1 Check Deployment Progress

```python
import boto3
import time

bedrock = boto3.client('bedrock')

def check_model_status(model_name):
    """Check custom model deployment status"""
    try:
        response = bedrock.get_custom_model(modelIdentifier=model_name)
        status = response.get('modelStatus', 'UNKNOWN')
        return status
    except Exception as e:
        print(f"Error checking status: {e}")
        return None

# Wait for deployment to complete
print("⏳ Waiting for deployment to complete...")
while True:
    status = check_model_status(model_name)
    print(f"   Status: {status}")

    if status == 'Ready':
        print("✅ Model deployed successfully!")
        break
    elif status in ['Failed', 'Stopping']:
        print(f"❌ Deployment failed with status: {status}")
        break

    time.sleep(30)  # Check every 30 seconds
```

### 4.2 List Custom Models

```python
# List all your custom models
response = bedrock.list_custom_models(maxResults=10)

print("📊 Your Custom Models:")
for model in response.get('modelSummaries', []):
    print(f"   - {model['modelName']}")
    print(f"     ARN: {model['modelArn']}")
    print(f"     Status: {model.get('modelStatus', 'N/A')}")
    print(f"     Created: {model.get('creationTime', 'N/A')}")
    print()
```

---

## Step 5: Test Inference

### 5.1 Invoke Model via SDK

```python
# Test inference using the customizer
inference_result = customizer.invoke_inference(
    request_body={
        "messages": [
            {"role": "user", "content": "What is machine learning?"}
        ],
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }
)

# Display result
inference_result.show()
```

### 5.2 Invoke Model via Bedrock Runtime API

```python
import boto3
import json

bedrock_runtime = boto3.client('bedrock-runtime')

# Prepare request
request = {
    "messages": [
        {"role": "user", "content": "Explain cloud computing in simple terms."}
    ],
    "inferenceConfig": {
        "maxTokens": 200,
        "temperature": 0.7,
        "topP": 0.9
    }
}

# Invoke model
response = bedrock_runtime.converse(
    modelId=model_arn,  # Use your model ARN
    messages=request['messages'],
    inferenceConfig=request['inferenceConfig']
)

# Extract response
output = response['output']['message']['content'][0]['text']
print("🤖 Model Response:")
print(output)
```

### 5.3 Streaming Inference

```python
# Streaming response for real-time output
request = {
    "messages": [
        {"role": "user", "content": "Write a short story about AI."}
    ],
    "inferenceConfig": {
        "maxTokens": 500,
        "temperature": 0.8
    }
}

# Invoke with streaming
response = bedrock_runtime.converse_stream(
    modelId=model_arn,
    messages=request['messages'],
    inferenceConfig=request['inferenceConfig']
)

# Stream output
print("🤖 Model Response (streaming):")
for event in response['stream']:
    if 'contentBlockDelta' in event:
        delta = event['contentBlockDelta']['delta']
        if 'text' in delta:
            print(delta['text'], end='', flush=True)

print("\n")
```

---

## Step 6: Batch Inference (Optional)

For processing large datasets:

```python
# Prepare batch inference data
batch_data = [
    {"messages": [{"role": "user", "content": "What is AWS?"}]},
    {"messages": [{"role": "user", "content": "Explain S3 buckets."}]},
    {"messages": [{"role": "user", "content": "What is Lambda?"}]}
] * 10  # 30 samples

# Save to JSONL
with open("batch_inference_input.jsonl", "w") as f:
    for item in batch_data:
        f.write(json.dumps(item) + "\n")

# Upload to S3
import boto3
s3 = boto3.client('s3')
S3_BUCKET = "your-bucket-name"
s3.upload_file(
    "batch_inference_input.jsonl",
    S3_BUCKET,
    "batch-inference/input.jsonl"
)

# Run batch inference
batch_result = customizer.batch_inference(
    input_s3_path=f"s3://{S3_BUCKET}/batch-inference/input.jsonl",
    output_s3_path=f"s3://{S3_BUCKET}/batch-inference/output/",
    model_path=checkpoint_path
)

print(f"✅ Batch inference started: {batch_result.job_id}")
```

---

## Step 7: Monitor Usage and Costs

### 7.1 CloudWatch Metrics

```python
import boto3
from datetime import datetime, timedelta

cloudwatch = boto3.client('cloudwatch')

# Get invocation metrics
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=1)

response = cloudwatch.get_metric_statistics(
    Namespace='AWS/Bedrock',
    MetricName='Invocations',
    Dimensions=[
        {'Name': 'ModelId', 'Value': model_arn}
    ],
    StartTime=start_time,
    EndTime=end_time,
    Period=300,  # 5 minutes
    Statistics=['Sum']
)

print("📊 Invocation Metrics (last hour):")
for datapoint in response['Datapoints']:
    print(f"   {datapoint['Timestamp']}: {datapoint['Sum']} invocations")
```

### 7.2 Cost Tracking

```python
# Estimate costs (example rates, check AWS pricing for actuals)
COST_PER_1K_INPUT_TOKENS = 0.003  # Example
COST_PER_1K_OUTPUT_TOKENS = 0.015  # Example

def estimate_cost(input_tokens, output_tokens):
    """Estimate inference cost"""
    input_cost = (input_tokens / 1000) * COST_PER_1K_INPUT_TOKENS
    output_cost = (output_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS
    return input_cost + output_cost

# Example usage
total_cost = estimate_cost(input_tokens=1000, output_tokens=500)
print(f"💰 Estimated cost: ${total_cost:.4f}")
```

---

## Step 8: Model Lifecycle Management

### 8.1 Update Model (Create New Version)

To update your model, deploy a new version:

```python
# Train new version
new_training_result = customizer.train(
    job_name="model-v2",
    overrides={"lr": 3e-6}  # Different hyperparameters
)

# Deploy new version
new_deployment = customizer.deploy(
    deploy_platform=DeployPlatform.BEDROCK_OD,
    endpoint_name="my-custom-nova-model-v2",  # Different name
    job_result=new_training_result
)

print("✅ New model version deployed")
```

### 8.2 Delete Custom Model

```python
# Delete custom model (careful!)
def delete_custom_model(model_name):
    """Delete a custom Bedrock model"""
    try:
        bedrock.delete_custom_model(modelIdentifier=model_name)
        print(f"✅ Model {model_name} deleted")
    except Exception as e:
        print(f"❌ Error deleting model: {e}")

# Only delete if you're sure!
# delete_custom_model("my-old-model")
```

### 8.3 List Provisioned Throughput

```python
# For PT deployments, check throughput status
response = bedrock.list_provisioned_model_throughputs(maxResults=10)

print("📊 Provisioned Throughput Models:")
for pt in response.get('provisionedModelSummaries', []):
    print(f"   - {pt['provisionedModelName']}")
    print(f"     Model ARN: {pt['modelArn']}")
    print(f"     Status: {pt.get('status', 'N/A')}")
    print(f"     Units: {pt.get('modelUnits', 'N/A')}")
    print()
```

---

## Step 9: Integration with Bedrock Features

### 9.1 Use with Bedrock Agents

```python
# Your custom model can be used as the foundation model for Bedrock Agents
agent_config = {
    "agentName": "my-custom-agent",
    "foundationModel": model_arn,  # Your custom model
    "instruction": "You are a helpful assistant specialized in cloud computing.",
    "agentResourceRoleArn": "arn:aws:iam::123456789012:role/BedrockAgentRole"
}

# Create agent (requires bedrock-agent client)
bedrock_agent = boto3.client('bedrock-agent')
agent_response = bedrock_agent.create_agent(**agent_config)

print(f"✅ Bedrock Agent created: {agent_response['agent']['agentId']}")
```

### 9.2 Use with Knowledge Bases

```python
# Custom models work with Bedrock Knowledge Bases
# Configure retrieval-augmented generation (RAG)
kb_config = {
    "knowledgeBaseId": "your-kb-id",
    "modelArn": model_arn,
    "retrievalConfiguration": {
        "vectorSearchConfiguration": {
            "numberOfResults": 5
        }
    }
}

print("✅ Custom model configured with Knowledge Base")
```

---

## Common Issues & Troubleshooting

### Issue: "Model import failed"

**Symptoms:**
```
ModelImportJobStatus: Failed
Reason: Invalid checkpoint format
```

**Solutions:**

1. **Verify checkpoint format**:
```python
# Ensure checkpoint is from a completed training job
print(f"Checkpoint: {training_result.model_artifacts.checkpoint_s3_path}")
print(f"Job Status: {training_result.status}")
```

2. **Check training method compatibility**:
```python
# Nova 1.0 models: SFT, DPO
# Nova 2.0 models: SFT, RFT
# Ensure your model supports Bedrock deployment
```

### Issue: "AccessDenied when creating custom model"

**Solution**: Add Bedrock permissions:

```json
{
    "Effect": "Allow",
    "Action": [
        "bedrock:CreateCustomModel",
        "bedrock:GetCustomModel",
        "bedrock:CreateProvisionedModelThroughput"
    ],
    "Resource": "*"
}
```

### Issue: "InsufficientThroughput" errors

**Solution**: For On-Demand, wait and retry. For consistent load, use Provisioned Throughput:

```python
# Switch to PT for guaranteed capacity
pt_deployment = customizer.deploy(
    deploy_platform=DeployPlatform.BEDROCK_PT,
    endpoint_name="my-model-pt",
    unit_count=2  # Increase as needed
)
```

### Issue: High latency with On-Demand

**Solution**: Use Provisioned Throughput for lower latency:

```python
# PT provides lower latency than OD
deployment = customizer.deploy(
    deploy_platform=DeployPlatform.BEDROCK_PT,
    endpoint_name="my-low-latency-model",
    unit_count=1
)
```

---

## Quick Reference

### Minimal Deployment Example

```python
from amzn_nova_customization_sdk import *

# Load training result
training_result = TrainingResult.load("training_result.json")

# Setup customizer
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=1),
    data_s3_path="s3://bucket/data.jsonl"
)

# Deploy to Bedrock
result = customizer.deploy(
    deploy_platform=DeployPlatform.BEDROCK_OD,
    endpoint_name="my-model",
    job_result=training_result
)

print(f"✅ Deployed: {result.endpoint.endpoint_name}")
```

### Test Inference

```python
# Invoke model
response = customizer.invoke_inference(
    request_body={
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)

response.show()
```

---

## Comparison: When to Use Bedrock vs SageMaker

| Factor | Bedrock | SageMaker (Journey 9) |
|--------|---------|----------------------|
| **Management** | Fully managed | Self-managed endpoints |
| **Setup Complexity** | ⭐ Simple | ⭐⭐ Moderate |
| **API** | Bedrock API | SageMaker API |
| **Scaling** | Automatic (OD) | Manual/Auto-scaling |
| **Integration** | Agents, KBs | Custom apps |
| **Cost Model** | Per-token or PT | Per-instance-hour |
| **Best For** | Managed inference | Custom infrastructure |

**Recommendation**:
- **Use Bedrock** if you want simple, managed inference
- **Use SageMaker** (Journey 9) if you need custom infrastructure control

---

## Next Steps

✅ **Bedrock deployment complete!** Now you can:

- **Journey 10: Inference & Monitoring** - Run production inference
- **Journey 9: SageMaker Deployment** - Alternative deployment option
- **Bedrock Agents** - Build agentic applications
- **Iterate** - Retrain and deploy improved models

---

## Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bedrock Custom Models](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html)
- [Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [SDK Deploy Spec](../nova-customization-sdk/docs/spec.md#deploy)
- [Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/)
