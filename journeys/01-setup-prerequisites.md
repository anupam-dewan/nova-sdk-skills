# Journey 1: Setup & Prerequisites

## Overview
**Purpose**: Configure your environment for Nova model customization
**Time**: 20-30 minutes
**Prerequisites**: AWS account, Python 3.12+
**Outputs**: Configured environment ready for training

---

## What You'll Learn

- Install the Nova Customization SDK
- Configure AWS credentials and IAM roles
- Set up execution roles for SageMaker
- Install HyperPod CLI (for SMHP users)
- Verify your environment

---

## Step 1: Python Environment Setup

### 1.1 Check Python Version

The SDK requires **Python 3.12 or higher**.

```bash
python --version
# Should show Python 3.12.x or higher
```

If you need to upgrade Python:

```bash
# macOS (using Homebrew)
brew install python@3.12

# Ubuntu/Debian
sudo apt update
sudo apt install python3.12

# Or use pyenv for version management
pyenv install 3.12.0
pyenv global 3.12.0
```

### 1.2 Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3.12 -m venv nova-env

# Activate it
source nova-env/bin/activate  # On macOS/Linux
# nova-env\Scripts\activate   # On Windows

# Verify activation
which python
# Should point to your venv
```

---

## Step 2: Install Nova Customization SDK

```bash
# Install the SDK
pip install amzn-nova-customization-sdk

# Verify installation
python -c "from amzn_nova_customization_sdk import *; print('✅ SDK installed successfully!')"
```

**Note**: The SDK automatically installs `sagemaker==2.254.1` as a dependency.

---

## Step 3: Configure AWS Credentials

### 3.1 Option A: AWS CLI Configuration

```bash
# Install AWS CLI if not already installed
pip install awscli

# Configure credentials
aws configure
# Enter your:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Output format (json)
```

### 3.2 Option B: Environment Variables

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

### 3.3 Verify Credentials

```python
import boto3

# Test credentials
sts = boto3.client('sts')
identity = sts.get_caller_identity()
print(f"✅ Authenticated as: {identity['Arn']}")
print(f"   Account ID: {identity['Account']}")
```

---

## Step 4: Configure IAM Roles and Policies

### 4.1 Create IAM Policy for SDK Usage

The SDK requires specific IAM permissions. Create a policy with these permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ConnectToHyperPodCluster",
            "Effect": "Allow",
            "Action": [
                "eks:DescribeCluster",
                "eks:ListAddons",
                "sagemaker:DescribeCluster"
            ],
            "Resource": [
                "arn:aws:eks:<region>:<account_id>:cluster/*",
                "arn:aws:sagemaker:<region>:<account_id>:cluster/*"
            ]
        },
        {
            "Sid": "StartSageMakerTrainingJob",
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateTrainingJob",
                "sagemaker:DescribeTrainingJob"
            ],
            "Resource": "arn:aws:sagemaker:<region>:<account_id>:training-job/*"
        },
        {
            "Sid": "HandleTrainingInputAndOutput",
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:GetObject",
                "s3:ListBucket",
                "s3:PutObject",
                "s3:AbortMultipartUpload",
                "s3:ListMultipartUploadParts"
            ],
            "Resource": "arn:aws:s3:::*"
        },
        {
            "Sid": "AccessCloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:DescribeLogStreams",
                "logs:FilterLogEvents",
                "logs:GetLogEvents"
            ],
            "Resource": "arn:aws:logs:<region>:<account_id>:log-group:*"
        },
        {
            "Sid": "ImportModelToBedrock",
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateCustomModel"
            ],
            "Resource": "*"
        },
        {
            "Sid": "DeployModelInBedrock",
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateProvisionedModelThroughput",
                "bedrock:GetCustomModel",
                "bedrock:GetProvisionedModelThroughput"
            ],
            "Resource": "arn:aws:bedrock:<region>:<account_id>:custom-model/*"
        },
        {
            "Sid": "DeployAndInvokeModelInSageMaker",
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:CreateModel",
                "sagemaker:DescribeEndpoint",
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": [
                "arn:aws:sagemaker:<region>:<account_id>:endpoint/*",
                "arn:aws:sagemaker:<region>:<account_id>:endpoint-config/*",
                "arn:aws:sagemaker:<region>:<account_id>:model/*"
            ]
        },
        {
            "Sid": "InteractWithIAMRoles",
            "Effect": "Allow",
            "Action": [
                "iam:GetRole",
                "iam:PassRole"
            ],
            "Resource": "arn:aws:iam::<account_id>:role/*"
        }
    ]
}
```

**To create the policy:**

```bash
# Save the policy to a file: nova-sdk-policy.json
# Then create it:
aws iam create-policy \
    --policy-name NovaCustomizationSDKPolicy \
    --policy-document file://nova-sdk-policy.json
```

### 4.2 Create Execution Role for SageMaker

The **execution role** is the role that SageMaker assumes to execute training jobs on your behalf.

**Trust Policy** (save as `trust-policy.json`):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

**Create the role:**

```bash
# Create the execution role
aws iam create-role \
    --role-name NovaCustomizationExecutionRole \
    --assume-role-policy-document file://trust-policy.json

# Attach SageMaker full access policy
aws iam attach-role-policy \
    --role-name NovaCustomizationExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Attach S3 access policy
aws iam attach-role-policy \
    --role-name NovaCustomizationExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

**For RFT Training**, add Lambda invoke permissions:

```json
{
    "Effect": "Allow",
    "Action": "lambda:InvokeFunction",
    "Resource": "arn:aws:lambda:<region>:<account_id>:function:*"
}
```

### 4.3 Get Your Execution Role ARN

```bash
aws iam get-role --role-name NovaCustomizationExecutionRole --query 'Role.Arn' --output text
# Save this ARN - you'll need it for training jobs
```

---

## Step 5: Set Up S3 Bucket

Create an S3 bucket for training data and model outputs:

```bash
# Create bucket (bucket names must be globally unique)
export S3_BUCKET="nova-customization-$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)"
aws s3 mb s3://${S3_BUCKET}

# Verify bucket creation
aws s3 ls | grep ${S3_BUCKET}

# Save bucket name for later
echo "export S3_BUCKET=${S3_BUCKET}" >> ~/.bashrc  # or ~/.zshrc
```

---

## Step 6: Install HyperPod CLI (For SMHP Users)

**Skip this step if you're only using SMTJ (SageMaker Training Jobs).**

### 6.1 Install Helm

```bash
# macOS
brew install helm

# Linux
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
rm -f ./get_helm.sh

# Verify
helm --help
```

### 6.2 Install HyperPod CLI

**For Non-Forge Customers:**

```bash
# Clone the release_v2 branch
git clone -b release_v2 https://github.com/aws/sagemaker-hyperpod-cli.git
cd sagemaker-hyperpod-cli

# Install the CLI
pip install .

# Verify installation
hyperpod --help
```

**For Forge Customers:** (Skip for now, will cover in Phase 2)

### 6.3 Configure EKS Access (HyperPod Only)

After creating your execution role, grant it access to your HyperPod cluster's EKS cluster:

```bash
# Create access entry
aws eks create-access-entry \
    --cluster-name <your-cluster-name> \
    --principal-arn arn:aws:iam::<account_id>:role/NovaCustomizationExecutionRole

# Associate cluster admin policy
aws eks associate-access-policy \
    --cluster-name <your-cluster-name> \
    --principal-arn arn:aws:iam::<account_id>:role/NovaCustomizationExecutionRole \
    --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \
    --access-scope type=cluster
```

---

## Step 7: Verify Environment

Create a verification script to test your setup:

```python
# save as verify_setup.py
import boto3
from amzn_nova_customization_sdk import *

def verify_setup():
    print("🔍 Verifying Nova SDK Environment...\n")

    # 1. Check SDK installation
    print("✅ SDK imported successfully")

    # 2. Check AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS credentials configured")
        print(f"   Account: {identity['Account']}")
        print(f"   User: {identity['Arn']}")
    except Exception as e:
        print(f"❌ AWS credentials not configured: {e}")
        return False

    # 3. Check S3 bucket access
    try:
        s3 = boto3.client('s3')
        s3.list_buckets()
        print(f"✅ S3 access verified")
    except Exception as e:
        print(f"❌ S3 access failed: {e}")
        return False

    # 4. Check IAM role access
    try:
        iam = boto3.client('iam')
        # Try to get a role (will fail gracefully if none exists)
        iam.list_roles(MaxItems=1)
        print(f"✅ IAM access verified")
    except Exception as e:
        print(f"⚠️  IAM access limited: {e}")

    # 5. Check SageMaker access
    try:
        sm = boto3.client('sagemaker')
        sm.list_training_jobs(MaxResults=1)
        print(f"✅ SageMaker access verified")
    except Exception as e:
        print(f"❌ SageMaker access failed: {e}")
        return False

    print("\n🎉 Environment setup complete!")
    print("\n📋 Next Steps:")
    print("   1. Journey 2: Prepare your training data")
    print("   2. Journey 3: Start your first SFT training job")

    return True

if __name__ == "__main__":
    verify_setup()
```

Run the verification:

```bash
python verify_setup.py
```

---

## Common Issues & Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'amzn_nova_customization_sdk'"

**Solution**: Make sure you installed the SDK in the correct environment:
```bash
pip list | grep nova
# Should show: amzn-nova-customization-sdk
```

### Issue: "NoCredentialsError: Unable to locate credentials"

**Solution**: Configure AWS credentials:
```bash
aws configure
# Or set environment variables
```

### Issue: "AccessDenied" when creating IAM roles

**Solution**: Your user needs IAM permissions. Ask your AWS administrator to grant:
- `iam:CreateRole`
- `iam:CreatePolicy`
- `iam:AttachRolePolicy`

### Issue: Python version < 3.12

**Solution**: Use pyenv to install Python 3.12:
```bash
curl https://pyenv.run | bash
pyenv install 3.12.0
pyenv global 3.12.0
```

---

## Quick Reference

### Essential Commands

```bash
# Activate environment
source nova-env/bin/activate

# Check SDK version
pip show amzn-nova-customization-sdk

# Test AWS credentials
aws sts get-caller-identity

# List S3 buckets
aws s3 ls

# Get execution role ARN
aws iam get-role --role-name NovaCustomizationExecutionRole
```

### Key Variables to Save

```bash
# Save these to your ~/.bashrc or ~/.zshrc
export AWS_DEFAULT_REGION="us-east-1"
export S3_BUCKET="your-bucket-name"
export EXECUTION_ROLE_ARN="arn:aws:iam::123456789012:role/NovaCustomizationExecutionRole"
```

---

## Next Steps

✅ **Environment configured!** You're ready to proceed to:

- **Journey 2: Data Preparation** - Load and prepare your training datasets
- **Quick Start**: Jump to Journey 3 (SFT Training) with sample data

---

## Resources

- [AWS IAM Roles Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html)
- [SageMaker Execution Roles](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)
- [Nova SDK GitHub](https://github.com/aws/nova-customization-sdk)
- [HyperPod CLI Documentation](https://github.com/aws/sagemaker-hyperpod-cli)
