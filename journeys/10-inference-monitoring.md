# Journey 10: Inference & Monitoring

## Overview
**Purpose**: Run inference with deployed Nova models and monitor production performance
**Time**: 30 minutes setup + ongoing monitoring
**Prerequisites**: Journey 8 (Bedrock) or Journey 9 (SageMaker) - model must be deployed
**Outputs**: Production-ready inference setup with monitoring dashboards

---

## What You'll Learn

- Run inference on deployed Nova models
- Configure inference parameters (temperature, top-p, etc.)
- Monitor model performance and costs
- Set up CloudWatch dashboards and alarms
- Handle inference errors and rate limits
- Optimize inference throughput
- Track model quality metrics

---

## Step 1: Run Inference on Bedrock

If you deployed to Bedrock (Journey 8):

### Basic Inference

```python
import boto3
import json

# Initialize Bedrock Runtime client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# Your custom model ID (from Journey 8)
model_id = "arn:aws:bedrock:us-east-1:123:provisioned-model/my-nova-model"

def invoke_nova_model(prompt: str) -> str:
    """Run inference on Bedrock-deployed Nova model."""

    payload = {
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["\n\n"]
    }

    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(payload)
    )

    result = json.loads(response['body'].read())
    return result['completion']

# Test inference
prompt = "Explain photosynthesis in simple terms"
response = invoke_nova_model(prompt)
print(f"Model: {response}")
```

### Batch Inference

```python
def batch_inference(prompts: list[str]) -> list[str]:
    """Run inference on multiple prompts."""

    responses = []
    for prompt in prompts:
        try:
            response = invoke_nova_model(prompt)
            responses.append(response)
        except Exception as e:
            print(f"Error on prompt: {prompt[:50]}... | {e}")
            responses.append(None)

    return responses

# Process batch
test_prompts = [
    "What is machine learning?",
    "Explain quantum computing",
    "How does photosynthesis work?"
]

results = batch_inference(test_prompts)
for prompt, response in zip(test_prompts, results):
    print(f"\nQ: {prompt}")
    print(f"A: {response}")
```

---

## Step 2: Run Inference on SageMaker

If you deployed to SageMaker (Journey 9):

### Basic Inference

```python
import boto3
import json

# Initialize SageMaker Runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Your endpoint name (from Journey 9)
endpoint_name = "my-nova-endpoint"

def invoke_sagemaker_endpoint(prompt: str) -> str:
    """Run inference on SageMaker endpoint."""

    payload = {
        "prompt": prompt,
        "parameters": {
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )

    result = json.loads(response['Body'].read())
    return result['generated_text']

# Test
response = invoke_sagemaker_endpoint("What is AI?")
print(response)
```

### Streaming Inference

```python
def invoke_streaming(prompt: str):
    """Stream responses from SageMaker endpoint."""

    payload = {
        "prompt": prompt,
        "parameters": {
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": True
        }
    }

    response = sagemaker_runtime.invoke_endpoint_with_response_stream(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )

    # Stream tokens as they arrive
    for event in response['Body']:
        chunk = json.loads(event['PayloadPart']['Bytes'])
        if 'token' in chunk:
            print(chunk['token'], end='', flush=True)

    print()  # Newline after streaming

# Stream response
invoke_streaming("Write a short story about AI")
```

---

## Step 3: Configure Inference Parameters

### Key Parameters

| Parameter | Description | Typical Range | Effect |
|-----------|-------------|---------------|--------|
| `temperature` | Sampling randomness | 0.0-1.0 | Higher = more creative |
| `top_p` | Nucleus sampling | 0.0-1.0 | Controls diversity |
| `max_tokens` | Max output length | 1-4096 | Response length limit |
| `top_k` | Top-K sampling | 0-100 | Alternative to top_p |
| `stop_sequences` | Stop generation | List of strings | Custom stopping |

### Example Configurations

```python
# Creative writing
creative_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 1024
}

# Factual Q&A (more deterministic)
factual_config = {
    "temperature": 0.3,
    "top_p": 0.85,
    "max_tokens": 256
}

# Code generation
code_config = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 512,
    "stop_sequences": ["```"]
}

# Use appropriate config
response = invoke_nova_model(
    prompt="Write a Python function to...",
    **code_config
)
```

---

## Step 4: Monitor with CloudWatch

### 4.1 Bedrock Metrics

```python
import boto3
from datetime import datetime, timedelta

cloudwatch = boto3.client('cloudwatch')

# Get Bedrock invocation metrics
def get_bedrock_metrics(model_id: str, hours: int = 24):
    """Fetch Bedrock metrics from CloudWatch."""

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)

    # Invocation count
    invocations = cloudwatch.get_metric_statistics(
        Namespace='AWS/Bedrock',
        MetricName='Invocations',
        Dimensions=[{'Name': 'ModelId', 'Value': model_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,  # 1 hour
        Statistics=['Sum']
    )

    # Latency
    latency = cloudwatch.get_metric_statistics(
        Namespace='AWS/Bedrock',
        MetricName='InvocationLatency',
        Dimensions=[{'Name': 'ModelId', 'Value': model_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,
        Statistics=['Average', 'p99']
    )

    # Errors
    errors = cloudwatch.get_metric_statistics(
        Namespace='AWS/Bedrock',
        MetricName='Errors',
        Dimensions=[{'Name': 'ModelId', 'Value': model_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,
        Statistics=['Sum']
    )

    return {
        'invocations': invocations['Datapoints'],
        'latency': latency['Datapoints'],
        'errors': errors['Datapoints']
    }

# Get metrics
metrics = get_bedrock_metrics(model_id)
print(f"Total invocations: {sum(m['Sum'] for m in metrics['invocations'])}")
print(f"Avg latency: {sum(m['Average'] for m in metrics['latency']) / len(metrics['latency']):.2f}ms")
print(f"Total errors: {sum(m['Sum'] for m in metrics['errors'])}")
```

### 4.2 SageMaker Metrics

```python
def get_sagemaker_metrics(endpoint_name: str, hours: int = 24):
    """Fetch SageMaker endpoint metrics."""

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)

    # Invocations
    invocations = cloudwatch.get_metric_statistics(
        Namespace='AWS/SageMaker',
        MetricName='Invocations',
        Dimensions=[
            {'Name': 'EndpointName', 'Value': endpoint_name},
            {'Name': 'VariantName', 'Value': 'AllTraffic'}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,
        Statistics=['Sum']
    )

    # Model latency
    model_latency = cloudwatch.get_metric_statistics(
        Namespace='AWS/SageMaker',
        MetricName='ModelLatency',
        Dimensions=[
            {'Name': 'EndpointName', 'Value': endpoint_name},
            {'Name': 'VariantName', 'Value': 'AllTraffic'}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,
        Statistics=['Average', 'p99']
    )

    # Invocation errors
    errors = cloudwatch.get_metric_statistics(
        Namespace='AWS/SageMaker',
        MetricName='ModelInvocationErrors',
        Dimensions=[
            {'Name': 'EndpointName', 'Value': endpoint_name},
            {'Name': 'VariantName', 'Value': 'AllTraffic'}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,
        Statistics=['Sum']
    )

    return {
        'invocations': invocations['Datapoints'],
        'latency': model_latency['Datapoints'],
        'errors': errors['Datapoints']
    }

# Get SageMaker metrics
metrics = get_sagemaker_metrics(endpoint_name)
```

---

## Step 5: Create CloudWatch Dashboard

```python
def create_monitoring_dashboard(model_id_or_endpoint: str, dashboard_type: str):
    """Create CloudWatch dashboard for model monitoring."""

    if dashboard_type == 'bedrock':
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/Bedrock", "Invocations", {"stat": "Sum", "label": "Total Invocations"}]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": "us-east-1",
                        "title": "Model Invocations"
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/Bedrock", "InvocationLatency", {"stat": "Average"}],
                            ["...", {"stat": "p99"}]
                        ],
                        "period": 300,
                        "region": "us-east-1",
                        "title": "Latency"
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/Bedrock", "Errors", {"stat": "Sum"}]
                        ],
                        "period": 300,
                        "region": "us-east-1",
                        "title": "Errors"
                    }
                }
            ]
        }
    else:  # sagemaker
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "Invocations", {"stat": "Sum"}]
                        ],
                        "period": 300,
                        "title": "Endpoint Invocations"
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "ModelLatency", {"stat": "Average"}],
                            ["...", {"stat": "p99"}]
                        ],
                        "period": 300,
                        "title": "Model Latency"
                    }
                }
            ]
        }

    # Create dashboard
    cloudwatch.put_dashboard(
        DashboardName=f"Nova-{dashboard_type.title()}-Monitoring",
        DashboardBody=json.dumps(dashboard_body)
    )

    print(f"✅ Dashboard created: Nova-{dashboard_type.title()}-Monitoring")

# Create dashboard
create_monitoring_dashboard(model_id, 'bedrock')
```

---

## Step 6: Set Up CloudWatch Alarms

### High Latency Alarm

```python
def create_latency_alarm(model_id: str, threshold_ms: int = 5000):
    """Alert when latency exceeds threshold."""

    cloudwatch.put_metric_alarm(
        AlarmName=f'Nova-HighLatency-{model_id}',
        ComparisonOperator='GreaterThanThreshold',
        EvaluationPeriods=2,
        MetricName='InvocationLatency',
        Namespace='AWS/Bedrock',
        Period=300,
        Statistic='Average',
        Threshold=threshold_ms,
        ActionsEnabled=True,
        AlarmActions=[
            'arn:aws:sns:us-east-1:123:my-alerts'  # SNS topic
        ],
        AlarmDescription=f'Alert when latency > {threshold_ms}ms',
        Dimensions=[{'Name': 'ModelId', 'Value': model_id}]
    )

    print(f"✅ Latency alarm created (threshold: {threshold_ms}ms)")
```

### High Error Rate Alarm

```python
def create_error_alarm(model_id: str, error_threshold: int = 10):
    """Alert when errors exceed threshold."""

    cloudwatch.put_metric_alarm(
        AlarmName=f'Nova-HighErrors-{model_id}',
        ComparisonOperator='GreaterThanThreshold',
        EvaluationPeriods=1,
        MetricName='Errors',
        Namespace='AWS/Bedrock',
        Period=300,
        Statistic='Sum',
        Threshold=error_threshold,
        ActionsEnabled=True,
        AlarmActions=[
            'arn:aws:sns:us-east-1:123:my-alerts'
        ],
        AlarmDescription=f'Alert when errors > {error_threshold}',
        Dimensions=[{'Name': 'ModelId', 'Value': model_id}]
    )

    print(f"✅ Error alarm created (threshold: {error_threshold} errors)")
```

---

## Step 7: Track Model Quality Metrics

Monitor output quality in production.

### Log Predictions for Analysis

```python
import boto3
import json
from datetime import datetime

logs_client = boto3.client('logs')
log_group = '/aws/nova/predictions'

def log_prediction(prompt: str, response: str, metadata: dict = None):
    """Log predictions to CloudWatch Logs for analysis."""

    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'prompt': prompt,
        'response': response,
        'metadata': metadata or {}
    }

    logs_client.put_log_events(
        logGroupName=log_group,
        logStreamName='production',
        logEvents=[{
            'timestamp': int(datetime.utcnow().timestamp() * 1000),
            'message': json.dumps(log_entry)
        }]
    )

# Use in production
response = invoke_nova_model(prompt)
log_prediction(prompt, response, metadata={'user_id': '12345'})
```

### Sample and Evaluate

```python
def sample_and_evaluate(num_samples: int = 100):
    """Sample predictions and evaluate quality."""

    # Fetch recent predictions from CloudWatch Logs
    response = logs_client.filter_log_events(
        logGroupName=log_group,
        logStreamName='production',
        limit=num_samples
    )

    predictions = [json.loads(event['message']) for event in response['events']]

    # Evaluate with your metrics
    scores = []
    for pred in predictions:
        # Custom quality metric (e.g., reward function)
        score = evaluate_quality(pred['prompt'], pred['response'])
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    print(f"📊 Quality Metrics:")
    print(f"   Samples: {len(scores)}")
    print(f"   Avg Score: {avg_score:.3f}")
    print(f"   Min Score: {min(scores):.3f}")
    print(f"   Max Score: {max(scores):.3f}")

    # Alert if quality drops
    if avg_score < 0.7:
        print("⚠️  WARNING: Quality below threshold!")
```

---

## Step 8: Cost Monitoring

### Track Inference Costs

```python
def estimate_bedrock_costs(model_id: str, hours: int = 24):
    """Estimate Bedrock inference costs."""

    metrics = get_bedrock_metrics(model_id, hours)
    total_invocations = sum(m['Sum'] for m in metrics['invocations'])

    # Bedrock pricing (example, check current pricing)
    cost_per_1k_tokens = 0.003  # Input tokens
    cost_per_1k_output = 0.006  # Output tokens

    # Estimate tokens (assume avg)
    avg_input_tokens = 100
    avg_output_tokens = 200

    total_cost = (
        (total_invocations * avg_input_tokens / 1000) * cost_per_1k_tokens +
        (total_invocations * avg_output_tokens / 1000) * cost_per_1k_output
    )

    print(f"💰 Estimated Cost (last {hours}h):")
    print(f"   Invocations: {total_invocations:,}")
    print(f"   Cost: ${total_cost:.2f}")

    return total_cost

# Monitor costs
cost = estimate_bedrock_costs(model_id)
```

---

## Common Issues & Solutions

### Issue 1: Rate Limiting

**Error:**
```
ThrottlingException: Rate exceeded
```

**Solutions:**
```python
# 1. Add exponential backoff
import time
from botocore.exceptions import ClientError

def invoke_with_retry(prompt: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return invoke_nova_model(prompt)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")

# 2. Request quota increase
# AWS Console > Service Quotas > Amazon Bedrock
```

### Issue 2: High Latency

**Symptom:** Slow inference responses

**Solutions:**
```python
# 1. Reduce max_tokens
config["max_tokens"] = 256  # Was 512

# 2. Use provisioned throughput (Bedrock)
# Provision capacity for consistent performance

# 3. Scale SageMaker endpoint
sagemaker.update_endpoint_weights_and_capacities(
    EndpointName=endpoint_name,
    DesiredWeightsAndCapacities=[{
        'VariantName': 'AllTraffic',
        'DesiredInstanceCount': 4  # Scale up
    }]
)
```

### Issue 3: Memory Errors (SageMaker)

**Error:**
```
ModelError: Out of memory
```

**Solutions:**
```python
# 1. Use larger instance type
# Update endpoint with ml.p5.48xlarge instead of ml.p4d.24xlarge

# 2. Reduce batch size (if batching)
# 3. Lower max_seq_length in inference config
```

---

## Quick Reference

### Minimal Inference Setup

```python
import boto3
import json

# Bedrock
bedrock = boto3.client('bedrock-runtime')
response = bedrock.invoke_model(
    modelId="your-model-id",
    body=json.dumps({"prompt": "Hello", "max_tokens": 256})
)
print(json.loads(response['body'].read())['completion'])

# SageMaker
sagemaker = boto3.client('sagemaker-runtime')
response = sagemaker.invoke_endpoint(
    EndpointName="your-endpoint",
    Body=json.dumps({"prompt": "Hello"})
)
print(json.loads(response['Body'].read())['generated_text'])
```

### Monitoring Best Practices

✅ **Do:**
- Monitor latency, errors, and costs
- Set up CloudWatch alarms for anomalies
- Log predictions for quality analysis
- Sample and evaluate outputs regularly
- Use exponential backoff for rate limits
- Track model performance trends

❌ **Don't:**
- Ignore latency spikes (investigate causes)
- Skip error monitoring (catch issues early)
- Forget cost tracking (can get expensive)
- Overload with high QPS without provisioning
- Ignore quality degradation signals

---

## Next Steps

**Production Readiness:**
1. **Scale testing** - Load test with expected QPS
2. **Implement caching** - Cache common responses
3. **A/B testing** - Compare model versions
4. **Feedback loop** - Collect user feedback for retraining

**Related Journeys:**
- **Journey 8**: Bedrock Deployment (if using Bedrock)
- **Journey 9**: SageMaker Deployment (if using SageMaker)
- **Journey 7**: Evaluation (offline quality metrics)

---

## Resources

- **Bedrock Metrics**: https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring.html
- **SageMaker Monitoring**: https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html
- **CloudWatch Best Practices**: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Best_Practices.html

---

**💡 Pro Tip**: Set up monitoring from day one! Don't wait for production issues. CloudWatch dashboards and alarms help you catch problems early and maintain model quality over time.
