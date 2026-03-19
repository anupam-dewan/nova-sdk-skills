# Nova Guide - Claude Code Skill

A Claude Code skill that provides interactive guidance for customizing Amazon Nova models.

## Quick Setup (3 minutes)

```bash
# 1. Clone this repository
git clone https://github.com/anupam-dewan/nova-sdk-skills.git
cd nova-sdk-skills

# 2. Run the setup script
./setup.sh

# 3. Start Claude Code
claude

# 4. Verify (type in Claude Code)
/skills
# You should see "nova-guide" in the list
```

**That's it!** Start asking questions:

```sh
How do I set up the Nova Forge SDK?
Show me a quick SFT training example using Nova Forge
How do I deploy my model to Bedrock?
```

> **Tip**: Mention "Nova", "Nova Forge", or "Forge SDK" for auto-activation, or use `/nova-guide` to explicitly invoke the skill.

---

## What This Skill Does

When you ask Claude Code about Nova customization, this skill automatically activates and guides you through:

- Setting up the Nova SDK environment
- Preparing datasets for training
- Training Nova models (SFT, CPT, DPO, RFT)
- Evaluating model performance
- Deploying to Bedrock or SageMaker
- Running inference and monitoring

## Available Journeys

1. **Setup & Prerequisites** - Environment setup, IAM roles, SDK installation
2. **Data Preparation** - Load, transform, and validate datasets
3. **SFT Training** - Supervised Fine-Tuning ⭐ Most Common
4. **CPT Training** - Continued Pre-Training for domain adaptation
5. **Evaluation** - Benchmark and validate model performance
6a. **RFT Single-Turn** - Reinforcement Fine-Tuning with reward functions
6b. **RFT Multi-Turn** - Conversational RFT for dialogue systems
7. **Bedrock Deployment** - Deploy to Amazon Bedrock
8. **SageMaker Deployment** - Deploy to SageMaker endpoints
9. **Inference & Monitoring** - Run models and monitor production performance

## Example Interaction

```md
You: How do I train a Nova model?

Claude: I'll help you train a Nova model using Supervised Fine-Tuning (SFT).
        Let me read the SFT training journey...

        [Provides step-by-step guidance with code examples]
```

## Troubleshooting

**Skill not showing in `/skills`?**

```bash
# Check if installed correctly
ls -la ~/.claude/skills/nova-guide/SKILL.md

# Re-run setup if needed
cd /path/to/nova-sdk-skills
./setup.sh
```

**Can't find journey files?**

The setup script automatically configures paths. If you moved the repository, re-run `./setup.sh` from the new location.

**Skill doesn't activate?**

The skill auto-activates when you mention these terms:
- ✅ "How do I train a **Nova model** with SFT?"
- ✅ "Show me a **Nova Forge SDK** example"
- ✅ "How do I use the **Forge SDK**?"
- ❌ "How do I fine-tune a model?" (no activation keywords)

**Activation keywords**: Nova, Nova SDK, Nova Forge, Forge SDK, amzn-nova-forge

For generic questions, explicitly invoke the skill:
```
/nova-guide How do I fine-tune a model?
```

---

## Additional Details

<details>
<summary><b>Manual Installation</b></summary>

If you prefer manual setup:

```bash
# 1. Clone repository
git clone https://github.com/anupam-dewan/nova-sdk-skills.git
cd nova-sdk-skills

# 2. Create skill directory
mkdir -p ~/.claude/skills/nova-guide

# 3. Install with path configuration
REPO_PATH=$(pwd)
sed "s|<REPO_PATH>|$REPO_PATH|g" SKILL.md > ~/.claude/skills/nova-guide/SKILL.md

# 4. Verify
ls -la ~/.claude/skills/nova-guide/SKILL.md
```

</details>

<details>
<summary><b>How It Works</b></summary>

### Directory Structure

```ini
~/.claude/skills/nova-guide/SKILL.md    # Skill file (installed by setup)
/your/path/nova-sdk-skills/
  ├── journeys/                         # Journey guides (read by Claude)
  ├── reference/                        # Reference docs
  └── SKILL.md                          # Skill template
```

### Setup Script

The `setup.sh` script:

1. Creates `~/.claude/skills/nova-guide/` directory
2. Copies `SKILL.md` and configures paths automatically
3. Points to journey files in your repository

### When You Ask a Question

**Auto-activation** (mentions Nova):
1. You ask: "How do I train a Nova model?"
2. Claude detects "Nova" and auto-activates the skill
3. Reads relevant journey file
4. Provides step-by-step guidance

**Manual activation** (generic question):
1. You type: `/nova-guide How do I fine-tune a model?`
2. Skill explicitly activated
3. Same guidance as above

</details>

<details>
<summary><b>Common User Paths</b></summary>

**Quick Start (Beginner):**

```ini
Setup → Data Prep → SFT → Evaluation → Bedrock Deploy
(Journeys 1 → 2 → 3 → 4 → 5)
```

**Domain Adaptation:**

```ini
Setup → Data Prep → CPT → Data Prep → SFT → Evaluation
(Journeys 1 → 2 → [CPT] → 2 → 3 → 4)
```

</details>

<details>
<summary><b>Requirements</b></summary>

- **Claude Code CLI** - [Installation instructions](https://docs.anthropic.com/claude-code)
- **amzn-nova-forge** - For actual training (optional for learning)
- **AWS Account** - With appropriate permissions for training jobs

</details>

<details>
<summary><b>Contributing</b></summary>

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Adding new journeys
- Improving existing guides
- Testing changes

</details>

<details>
<summary><b>Repository Structure</b></summary>

```ini
nova-sdk-skills/
├── SKILL.md              # Skill template
├── setup.sh              # Automated installer
├── journeys/             # User journey guides (10 complete!)
│   ├── 01-setup-prerequisites.md
│   ├── 02-data-preparation.md
│   ├── 03-sft-training.md
│   ├── 04-cpt-training.md
│   ├── 06a-rft-singleturn.md
│   ├── 06b-rft-multiturn.md
│   ├── 07-evaluation.md
│   ├── 08-bedrock-deployment.md
│   ├── 09-sagemaker-deployment.md
│   └── 10-inference-monitoring.md
└── reference/            # Reference documentation
    └── sdk-capabilities.md
```

</details>

---

## Resources

- [Amazon Nova Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-customization.html)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [amzn-nova-forge Repository](https://github.com/aws/amzn-nova-forge)

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review journey files directly in `journeys/`
3. Open an issue in this repository

---

**Happy customizing!** 🚀
