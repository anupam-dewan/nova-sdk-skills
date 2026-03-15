# Journey 9: SageMaker Deployment

## Overview
**Purpose**: Deploy custom Nova models to Amazon SageMaker for flexible, self-managed inference
**Time**: 15-30 minutes
**Prerequisites**: Journey 1 (Setup), Journey 3/4/5/6 (Trained model)
**Outputs**: SageMaker endpoint with custom infrastructure configuration

---

## What You'll Learn

- Deploy trained models to SageMaker endpoints
- Configure instance types and autoscaling
- Set environment variables (context length, concurrency)
- Test inference via SageMaker API
- Monitor endpoint metrics
- Update and manage endpoints

---

## What is SageMaker Deployment?

**Amazon SageMaker Endpoints** provide self-managed real-time inference:

### Benefits
- **Custom Infrastructure**: Choose instance types, counts
- **Autoscaling**: Configure scaling policies
- **VPC Support**: Deploy in your VPC
- **Fine-grained Control**: Environment variables, monitoring
- **Cost Optimization**: Pay only for instances you use

### When to Use SageMaker vs Bedrock

| Factor | SageMaker (This Journey) | Bedrock (Journey 8) |
|--------|-------------------------|---------------------|
| **Control** | High (custom infra) | Low (fully managed) |
| **Setup** | Moderate | Simple |
| **Scaling** | Manual/Auto policies | Automatic |
| **VPC** | Full VPC support | Limited |
| **Best For** | Custom requirements | Simple, managed inference |

---

## Step 1: Verify Prerequisites

### 1.1 Check Training Checkpoint

```python
from amzn_nova_customization_sdk import TrainingResult

# Load training result
training_result = TrainingResult.load("training_result.json")

checkpoint_path = training_result.model_artifacts.checkpoint_s3_path
print(f"📁 Checkpoint: {checkpoint_path}")
print(f"   Model: {training_result.model_type}")
print(f"   Method: {training_result.method}")
```

### 1.2 Verify IAM Permissions

```python
import boto3

sagemaker = boto3.client('sagemaker')

try:
    # Test SageMaker access
    sagemaker.list_endpoints(MaxResults=1)
    print("✅ SageMaker access verified")
except Exception as e:
    print(f"❌ SageMaker access failed: {e}")
    print("   Add sagemaker:* permissions to your IAM role")
```

---

## Step 2: Deploy to SageMaker Endpoint

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

### 2.2 Deploy with Basic Configuration

```python
# Deploy to SageMaker
deployment_result = customizer.deploy(
    deploy_platform=DeployPlatform.SAGEMAKER,
    unit_count=1,  # Number of instances
    endpoint_name="my-custom-nova-endpoint",  # Unique endpoint name
    job_result=training_result  # Automatically extracts checkpoint
)

print("\n🚀 SageMaker deployment started!")
print(f"   Endpoint Name: {deployment_result.endpoint.endpoint_name}")
print(f"   Status: {deployment_result.status}")
print(f"   Instance Count: 1")

# Save endpoint info
endpoint_name = deployment_result.endpoint.endpoint_name
```

### 2.3 Deploy with Custom Environment Variables

```python
# Deploy with custom configuration
deployment_result = customizer.deploy(
    deploy_platform=DeployPlatform.SAGEMAKER,
    unit_count=2,  # Multiple instances for higher throughput
    endpoint_name="my-custom-nova-endpoint-prod",
    sagemaker_environment_variables={
        "CONTEXT_LENGTH": "12000",  # Custom context window
        "MAX_CONCURRENCY": "32",    # Max concurrent requests
        "BATCH_SIZE": "8",          # Batch size for processing
        "GPU_MEMORY_UTILIZATION": "0.9"  # GPU memory usage
    },
    job_result=training_result
)

print("\n🚀 SageMaker deployment with custom config started!")
print(f"   Endpoint: {deployment_result.endpoint.endpoint_name}")
print(f"   Instances: 2")
print(f"   Context Length: 12000")
print(f"   Max Concurrency: 32")

endpoint_name = deployment_result.endpoint.endpoint_name
```

### 2.4 Alternative: Deploy from Checkpoint Path

```python
# If you don't have TrainingResult, use checkpoint path directly
deployment_result = customizer.deploy(
    deploy_platform=DeployPlatform.SAGEMAKER,
    unit_count=1,
    endpoint_name="my-endpoint-from-checkpoint",
    model_path="s3://bucket/checkpoints/my-model/"  # Direct S3 path
)
```

---

## Step 3: Monitor Deployment Progress

### 3.1 Check Endpoint Status

```python
import boto3
import time

sagemaker = boto3.client('sagemaker')

def check_endpoint_status(endpoint_name):
    """Check SageMaker endpoint status"""
    try:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        return status
    except Exception as e:
        print(f"Error checking status: {e}")
        return None

# Wait for deployment to complete
print("⏳ Waiting for endpoint deployment...")
while True:
    status = check_endpoint_status(endpoint_name)
    print(f"   Status: {status}")

    if status == 'InService':
        print("✅ Endpoint deployed successfully!")
        break
    elif status in ['Failed', 'RollingBack']:
        print(f"❌ Deployment failed with status: {status}")
        # Get failure reason
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        print(f"   Reason: {response.get('FailureReason', 'Unknown')}")
        break

    time.sleep(30)  # Check every 30 seconds
```

### 3.2 Get Endpoint Details

```python
# Get full endpoint details
response = sagemaker.describe_endpoint(EndpointName=endpoint_name)

print("📊 Endpoint Details:")
print(f"   Name: {response['EndpointName']}")
print(f"   ARN: {response['EndpointArn']}")
print(f"   Status: {response['EndpointStatus']}")
print(f"   Created: {response['CreationTime']}")
print(f"   Last Modified: {response['LastModifiedTime']}")

# Get endpoint configuration
config_name = response['EndpointConfigName']
config_response = sagemaker.describe_endpoint_config(EndpointConfigName=config_name)

for variant in config_response['ProductionVariants']:
    print(f"\n   Variant: {variant['VariantName']}")
    print(f"   Instance Type: {variant['InstanceType']}")
    print(f"   Instance Count: {variant['InitialInstanceCount']}")
    print(f"   Model: {variant['ModelName']}")
```

---

## Step 4: Test Inference

### 4.1 Invoke via SDK

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

### 4.2 Invoke via SageMaker Runtime API

```python
import boto3
import json

sagemaker_runtime = boto3.client('sagemaker-runtime')

# Prepare request
request = {
    "messages": [
        {
            "role": "user",
            "content": "Explain cloud computing in simple terms."
        }
    ],
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9
}

# Invoke endpoint
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(request)
)

# Parse response
result = json.loads(response['Body'].read().decode())
output = result['output']['message']['content']

print("🤖 Model Response:")
print(output)
```

### 4.3 Streaming Inference

```python
# Invoke with streaming response
request = {
    "messages": [
        {"role": "user", "content": "Write a short story about AI."}
    ],
    "max_tokens": 500,
    "temperature": 0.8,
    "stream": True  # Enable streaming
}

response = sagemaker_runtime.invoke_endpoint_with_response_stream(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(request)
)

# Stream output
print("🤖 Model Response (streaming):")
event_stream = response['Body']
for event in event_stream:
    if 'PayloadPart' in event:
        data = json.loads(event['PayloadPart']['Bytes'].decode())
        if 'token' in data:
            print(data['token']['text'], end='', flush=True)

print("\n")
```

### 4.4 Batch Inference

```python
# For multiple requests, can use batch processing
requests = [
    {"messages": [{"role": "user", "content": "What is AWS?"}], "max_tokens": 100},
    {"messages": [{"role": "user", "content": "Explain S3."}], "max_tokens": 100},
    {"messages": [{"role": "user", "content": "What is Lambda?"}], "max_tokens": 100}
]

results = []
for req in requests:
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(req)
    )
    result = json.loads(response['Body'].read().decode())
    results.append(result)

print(f"✅ Processed {len(results)} requests")
```

---

## Step 5: Configure Autoscaling

### 5.1 Enable Autoscaling

```python
import boto3

autoscaling = boto3.client('application-autoscaling')

# Register scalable target
autoscaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=5  # Scale up to 5 instances
)

print("✅ Autoscaling target registered")
print(f"   Min instances: 1")
print(f"   Max instances: 5")
```

### 5.2 Configure Scaling Policy

```python
# Target tracking scaling policy
autoscaling.put_scaling_policy(
    PolicyName=f'{endpoint_name}-scaling-policy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,  # Target 70% invocations per instance
        'CustomizedMetricSpecification': {
            'MetricName': 'SageMakerVariantInvocationsPerInstance',
            'Namespace': 'AWS/SageMaker',
            'Dimensions': [
                {'Name': 'EndpointName', 'Value': endpoint_name},
                {'Name': 'VariantName', 'Value': 'AllTraffic'}
            ],
            'Statistic': 'Average'
        },
        'ScaleInCooldown': 300,  # 5 minutes
        'ScaleOutCooldown': 60   # 1 minute
    }
)

print("✅ Scaling policy configured")
print(f"   Target: 70 invocations per instance")
```

---

## Step 6: Monitor Endpoint Performance

### 6.1 CloudWatch Metrics

```python
import boto3
from datetime import datetime, timedelta

cloudwatch = boto3.client('cloudwatch')

# Get invocation metrics
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=1)

response = cloudwatch.get_metric_statistics(
    Namespace='AWS/SageMaker',
    MetricName='Invocations',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': endpoint_name},
        {'Name': 'VariantName', 'Value': 'AllTraffic'}
    ],
    StartTime=start_time,
    EndTime=end_time,
    Period=300,  # 5 minutes
    Statistics=['Sum', 'Average']
)

print("📊 Invocation Metrics (last hour):")
for datapoint in sorted(response['Datapoints'], key=lambda x: x['Timestamp']):
    print(f"   {datapoint['Timestamp']}: {datapoint['Sum']} invocations")
```

### 6.2 Monitor Latency

```python
# Get model latency metrics
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/SageMaker',
    MetricName='ModelLatency',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': endpoint_name},
        {'Name': 'VariantName', 'Value': 'AllTraffic'}
    ],
    StartTime=start_time,
    EndTime=end_time,
    Period=300,
    Statistics=['Average', 'Maximum']
)

print("\n📊 Latency Metrics (last hour):")
for datapoint in sorted(response['Datapoints'], key=lambda x: x['Timestamp']):
    avg_ms = datapoint.get('Average', 0) / 1000  # Convert to milliseconds
    max_ms = datapoint.get('Maximum', 0) / 1000
    print(f"   {datapoint['Timestamp']}: Avg={avg_ms:.2f}ms, Max={max_ms:.2f}ms")
```

### 6.3 Monitor Instance Count (Autoscaling)

```python
# Get current instance count
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/SageMaker',
    MetricName='CPUUtilization',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': endpoint_name},
        {'Name': 'VariantName', 'Value': 'AllTraffic'}
    ],
    StartTime=start_time,
    EndTime=end_time,
    Period=300,
    Statistics=['Average']
)

print("\n📊 CPU Utilization (last hour):")
for datapoint in sorted(response['Datapoints'], key=lambda x: x['Timestamp']):
    print(f"   {datapoint['Timestamp']}: {datapoint['Average']:.2f}%")
```

---

## Step 7: Update Endpoint

### 7.1 Update Instance Count

```python
# Update endpoint with more instances
new_config_name = f"{endpoint_name}-config-v2"

# Create new endpoint configuration
sagemaker.create_endpoint_config(
    EndpointConfigName=new_config_name,
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': response['EndpointConfigName'],  # Same model
            'InstanceType': 'ml.g5.12xlarge',  # Can change instance type
            'InitialInstanceCount': 3  # Increased from 1 to 3
        }
    ]
)

# Update endpoint
sagemaker.update_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=new_config_name
)

print("✅ Endpoint update started (3 instances)")
print("   Wait for status to return to 'InService'")
```

### 7.2 Deploy New Model Version

```python
# Train new model version
new_training_result = customizer.train(
    job_name="model-v2",
    overrides={"lr": 3e-6}
)

# Deploy new version with UPDATE_IF_EXISTS mode
customizer_update = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=1),
    data_s3_path="s3://bucket/data.jsonl",
    deployment_mode=DeploymentMode.UPDATE_IF_EXISTS  # Update existing
)

# This will update the existing endpoint
updated_deployment = customizer_update.deploy(
    deploy_platform=DeployPlatform.SAGEMAKER,
    unit_count=2,
    endpoint_name=endpoint_name,  # Same endpoint name
    job_result=new_training_result
)

print("✅ Endpoint updated with new model version")
```

---

## Step 8: Cost Optimization

### 8.1 Choose Right Instance Type

```python
# Instance type recommendations by workload
INSTANCE_RECOMMENDATIONS = {
    "low_throughput": {
        "instance_type": "ml.g5.xlarge",
        "count": 1,
        "cost_per_hour": 1.006  # Approximate
    },
    "medium_throughput": {
        "instance_type": "ml.g5.12xlarge",
        "count": 2,
        "cost_per_hour": 5.672 * 2
    },
    "high_throughput": {
        "instance_type": "ml.p4d.24xlarge",
        "count": 3,
        "cost_per_hour": 32.77 * 3
    }
}

# Calculate monthly cost
def estimate_monthly_cost(cost_per_hour, utilization=1.0):
    """Estimate monthly endpoint cost"""
    hours_per_month = 730  # Average
    return cost_per_hour * hours_per_month * utilization

print("💰 Monthly Cost Estimates:")
for workload, config in INSTANCE_RECOMMENDATIONS.items():
    monthly = estimate_monthly_cost(config['cost_per_hour'])
    print(f"   {workload}: ${monthly:,.2f}/month")
```

### 8.2 Use Spot Instances (If Supported)

```python
# Deploy with spot instances for cost savings
# Note: Check if spot instances are available for your region/instance type
deployment_result = customizer.deploy(
    deploy_platform=DeployPlatform.SAGEMAKER,
    unit_count=2,
    endpoint_name="my-spot-endpoint",
    # Spot instance configuration would go here
    # (Feature availability depends on SDK version)
    job_result=training_result
)
```

### 8.3 Implement Auto-Stop for Dev Endpoints

```python
# For development endpoints, auto-delete after hours
import boto3
from datetime import datetime, timedelta

def schedule_endpoint_deletion(endpoint_name, hours_from_now=8):
    """Schedule endpoint deletion using CloudWatch Events"""
    events = boto3.client('events')
    lambda_client = boto3.client('lambda')

    # Create rule to trigger after N hours
    rule_name = f"delete-{endpoint_name}"
    schedule_time = datetime.utcnow() + timedelta(hours=hours_from_now)

    events.put_rule(
        Name=rule_name,
        ScheduleExpression=f"cron({schedule_time.minute} {schedule_time.hour} {schedule_time.day} {schedule_time.month} ? {schedule_time.year})",
        State='ENABLED'
    )

    print(f"✅ Scheduled deletion for {endpoint_name} at {schedule_time}")
    print(f"   Cancel with: aws events delete-rule --name {rule_name}")

# Example: Delete dev endpoint after 8 hours
# schedule_endpoint_deletion("my-dev-endpoint", hours_from_now=8)
```

---

## Step 9: Endpoint Lifecycle Management

### 9.1 List All Endpoints

```python
# List all your SageMaker endpoints
response = sagemaker.list_endpoints(MaxResults=50)

print("📊 Your SageMaker Endpoints:")
for endpoint in response['Endpoints']:
    print(f"   - {endpoint['EndpointName']}")
    print(f"     Status: {endpoint['EndpointStatus']}")
    print(f"     Created: {endpoint['CreationTime']}")
    print()
```

### 9.2 Delete Endpoint

```python
def delete_endpoint(endpoint_name, delete_config=True, delete_model=True):
    """
    Delete SageMaker endpoint and associated resources

    WARNING: This is destructive and cannot be undone!
    """
    try:
        # Get endpoint details before deletion
        endpoint = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        config_name = endpoint['EndpointConfigName']

        # Get config to find model name
        config = sagemaker.describe_endpoint_config(EndpointConfigName=config_name)
        model_names = [v['ModelName'] for v in config['ProductionVariants']]

        # Delete endpoint
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
        print(f"✅ Endpoint {endpoint_name} deleted")

        # Delete endpoint configuration
        if delete_config:
            sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
            print(f"✅ Endpoint config {config_name} deleted")

        # Delete models
        if delete_model:
            for model_name in model_names:
                sagemaker.delete_model(ModelName=model_name)
                print(f"✅ Model {model_name} deleted")

    except Exception as e:
        print(f"❌ Error deleting endpoint: {e}")

# Only delete if you're absolutely sure!
# delete_endpoint("my-old-endpoint")
```

---

## Common Issues & Troubleshooting

### Issue: "ResourceLimitExceeded" when creating endpoint

**Symptoms:**
```
ClientError: Instance type ml.p5.48xlarge not available
```

**Solutions:**

1. **Request quota increase**:
```bash
# Check current limits
aws service-quotas list-service-quotas \
    --service-code sagemaker \
    --query 'Quotas[?contains(QuotaName, `ml.p5`)]'

# Request increase via AWS Console or CLI
```

2. **Use different instance type**:
```python
# Try smaller instance type
deployment = customizer.deploy(
    deploy_platform=DeployPlatform.SAGEMAKER,
    unit_count=1,
    endpoint_name="my-endpoint",
    # SDK chooses instance automatically based on model
    job_result=training_result
)
```

### Issue: High latency on endpoint

**Solutions:**

1. **Increase instance count**:
```python
# Update endpoint with more instances
sagemaker.update_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName="new-config-with-more-instances"
)
```

2. **Adjust environment variables**:
```python
deployment = customizer.deploy(
    deploy_platform=DeployPlatform.SAGEMAKER,
    unit_count=2,
    endpoint_name="optimized-endpoint",
    sagemaker_environment_variables={
        "MAX_CONCURRENCY": "64",  # Increase concurrency
        "BATCH_SIZE": "16"         # Larger batches
    },
    job_result=training_result
)
```

### Issue: Endpoint deployment fails

**Solution**: Check CloudWatch logs:

```python
logs = boto3.client('logs')

log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"

try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        limit=50
    )

    print("📋 Endpoint Logs:")
    for event in response['events']:
        print(event['message'])
except Exception as e:
    print(f"❌ Error fetching logs: {e}")
```

---

## Quick Reference

### Minimal Deployment Example

```python
from amzn_nova_customization_sdk import *

# Load training result
training_result = TrainingResult.load("training_result.json")

# Setup
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=1),
    data_s3_path="s3://bucket/data.jsonl"
)

# Deploy
result = customizer.deploy(
    deploy_platform=DeployPlatform.SAGEMAKER,
    unit_count=1,
    endpoint_name="my-endpoint",
    job_result=training_result
)

print(f"✅ Deployed: {result.endpoint.endpoint_name}")
```

### Test Inference

```python
# Invoke
response = customizer.invoke_inference(
    request_body={
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)

response.show()
```

---

## Comparison: SageMaker vs Bedrock

| Feature | SageMaker (This Journey) | Bedrock (Journey 8) |
|---------|-------------------------|---------------------|
| **Infrastructure Control** | ✅ Full control | ❌ Fully managed |
| **Instance Type Choice** | ✅ Any GPU instance | ❌ Managed |
| **Autoscaling Config** | ✅ Custom policies | ✅ Automatic (OD) |
| **VPC Deployment** | ✅ Full support | ⚠️ Limited |
| **Environment Variables** | ✅ Custom configs | ❌ Not available |
| **Setup Complexity** | ⭐⭐ Moderate | ⭐ Simple |
| **Cost Model** | Per-instance-hour | Per-token or PT |
| **Best For** | Custom infrastructure | Managed inference |

**Recommendation**:
- **Use SageMaker** when you need infrastructure control, VPC deployment, or custom configurations
- **Use Bedrock** (Journey 8) for simpler, fully managed inference

---

## Next Steps

✅ **SageMaker deployment complete!** Now you can:

- **Journey 10: Inference & Monitoring** - Production inference patterns
- **Journey 8: Bedrock Deployment** - Alternative deployment option
- **Setup Monitoring** - CloudWatch dashboards, alarms
- **Optimize Costs** - Autoscaling, instance selection
- **Iterate** - Retrain and deploy improved models

---

## Resources

- [SageMaker Endpoints Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)
- [SageMaker Inference Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [Autoscaling Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)
- [SDK Deploy Spec](../nova-customization-sdk/docs/spec.md#deploy)
- [SageMaker API Reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/)
