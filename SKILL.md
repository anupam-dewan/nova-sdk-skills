---
description: Guide for customizing Amazon Nova models using the Nova Forge SDK
name: nova-guide
user-invokable: true
---

# Nova Forge SDK Guide

Help users customize Amazon Nova models using the Nova Forge SDK.

## When to Use This Skill

Activate when users ask about:

- Nova customization: "How do I train a Nova model?", "Set up Nova SDK"
- Nova Forge: "How do I use Nova Forge?", "Nova Forge SDK setup"
- Forge SDK: "Show me Forge SDK examples", "Forge SDK training"
- Model training: "How do I do SFT training?", "What's DPO training?"
- Deployment: "Deploy to Bedrock", "Deploy to SageMaker"
- Data preparation: "How do I prepare data for Nova?"
- Evaluation: "How do I evaluate my Nova model?"
- Any question about Nova model customization, Forge SDK, or related workflows

**Key activation terms**: Nova, Nova SDK, Nova Forge, Forge SDK, Nova Forge SDK

## Available Journeys

1. **Setup & Prerequisites** - Environment setup, IAM roles, SDK installation (20-30 min)
2. **Data Preparation** - Load, transform, validate datasets for all training methods (15-30 min)
3. **SFT Training** - Supervised fine-tuning with LoRA or Full-rank (2-4 hours) ⭐ MOST COMMON
4. **CPT Training** - Continued pre-training for domain adaptation (3-6 hours)
5. **DPO Training** - Direct preference optimization for alignment (2-4 hours)
   6a. **RFT Single-Turn** - Reinforcement fine-tuning with Lambda rewards (3-5 hours)
   6b. **RFT Multi-Turn** - Multi-turn conversational RFT (4-8 hours)
6. **Evaluation** - Benchmark with MMLU, BYOD, BYOM, LLM-as-Judge (30 min - 2 hrs)
7. **Bedrock Deployment** - Deploy to Amazon Bedrock (15-30 min)
8. **SageMaker Deployment** - Deploy to SageMaker endpoints (15-30 min)
9. **Inference & Monitoring** - Run and monitor deployed models

## Common User Paths

**Quick Start (Beginner)**:
1 → 2 → 3 → 7 → 8 (Setup → Data → SFT → Eval → Bedrock)

**Domain Adaptation**:
1 → 2 → 4 → 2 → 3 → 7 → 8 (CPT then SFT)

**Alignment Pipeline**:
1 → 2 → 3 → 2 → 5 → 7 → 8 (SFT then DPO)

## Instructions

When a user asks about Nova customization:

1. **Identify the journey** they need based on their question
2. **Read the relevant journey file** from the nova-sdk-skills repository
3. **Guide them step-by-step** through the journey
4. **Provide copy-pasteable code** from the journey examples
5. **Help troubleshoot** using the "Common Issues" sections

### Available Journey Files

**IMPORTANT**: Update these paths to match where you cloned the nova-sdk-skills repository!

Replace `<REPO_PATH>` with your actual repository location (e.g., `/Users/yourname/nova-sdk-skills`):

- `<REPO_PATH>/journeys/01-setup-prerequisites.md`
- `<REPO_PATH>/journeys/02-data-preparation.md`
- `<REPO_PATH>/journeys/03-sft-training.md`
- `<REPO_PATH>/journeys/04-cpt-training.md`
- `<REPO_PATH>/journeys/06a-rft-singleturn.md`
- `<REPO_PATH>/journeys/06b-rft-multiturn.md`
- `<REPO_PATH>/journeys/07-evaluation.md`
- `<REPO_PATH>/journeys/08-bedrock-deployment.md`
- `<REPO_PATH>/journeys/09-sagemaker-deployment.md`
- `<REPO_PATH>/journeys/10-inference-monitoring.md`

### For Quick Reference

Use the reference documentation at `<REPO_PATH>/reference/sdk-capabilities.md` for SDK feature matrix.

## Example Interactions

**User**: "How do I train a Nova model?"
**You**: Read `journeys/03-sft-training.md` and guide them through SFT training, starting with prerequisites check.

**User**: "How do I deploy to Bedrock?"
**You**: Read `journeys/08-bedrock-deployment.md` and walk them through Bedrock deployment options (On-Demand vs Provisioned).

**User**: "What's the difference between SFT and DPO?"
**You**: Explain that SFT uses question-answer pairs for task-specific tuning (Journey 3), while DPO uses preference pairs for alignment (Journey 5). Point them to the relevant journey files.

**User**: "I'm getting an AccessDenied error"
**You**: Check the troubleshooting sections in the relevant journey file for IAM permission fixes.

## Platform Considerations

**SMTJ (SageMaker Training Jobs)**: Simpler setup, good for experimentation, supports SFT/DPO/RFT
**SMHP (SageMaker HyperPod)**: Production-ready, supports all methods including CPT

Guide users to start with SMTJ unless they need CPT or have production requirements.

## Quick Examples

### Minimal SFT Training

```python
from amzn_nova_forge import *

runtime = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=4)
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime,
    data_s3_path="s3://bucket/train.jsonl",
    output_s3_path="s3://bucket/output/"
)

result = customizer.train(job_name="my-training", overrides={"lr": 5e-6})
print(f"Training started: {result.job_id}")
```

### Minimal Bedrock Deployment

```python
deployment = customizer.deploy(
    deploy_platform=DeployPlatform.BEDROCK_OD,
    endpoint_name="my-model",
    job_result=training_result
)
print(f"Deployed: {deployment.endpoint.endpoint_name}")
```

## Resources

- Journey files: `<REPO_PATH>/journeys/`
- SDK capabilities: `<REPO_PATH>/reference/sdk-capabilities.md`
- Installation guide: https://github.com/anupam-dewan/nova-sdk-skills
