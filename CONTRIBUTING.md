# Contributing to Nova SDK Skills

Thank you for your interest in improving the Nova Guide skill!

## Repository Structure

```
nova-sdk-skills/
├── README.md                   # Main documentation (start here!)
├── SKILL.md                    # Skill file template
├── setup.sh                    # Automated installation script
├── CONTRIBUTING.md             # This file
│
├── journeys/                   # User journey guides
│   ├── 01-setup-prerequisites.md
│   ├── 02-data-preparation.md
│   ├── 03-sft-training.md
│   ├── 07-evaluation.md
│   ├── 08-bedrock-deployment.md
│   └── 09-sagemaker-deployment.md
│
└── reference/                  # Reference documentation
    └── sdk-capabilities.md
```

## How to Contribute

### Adding a New Journey

1. **Create the journey file** in `journeys/`
   ```bash
   cp journeys/03-sft-training.md journeys/XX-your-journey.md
   ```

2. **Follow the journey template structure:**
   - Overview section
   - What You'll Learn
   - Prerequisites
   - Step-by-step guide
   - Troubleshooting
   - Quick Reference
   - Next Steps

3. **Update SKILL.md** to include the new journey:
   - Add to "Available Journeys" list
   - Add to "Available Journey Files" paths
   - Update "Common User Paths" if applicable

4. **Test the journey:**
   - Install the skill locally
   - Ask Claude to guide you through the new journey
   - Verify all code examples work
   - Check all links are valid

### Improving Existing Journeys

1. **Identify the issue:**
   - Outdated code examples
   - Missing troubleshooting steps
   - Unclear instructions
   - Broken links

2. **Make your changes:**
   - Update the specific journey file
   - Test the changes with Claude Code
   - Ensure consistency with other journeys

3. **Update related docs:**
   - Check if README needs updates
   - Update journey cross-references if needed

### Testing Your Changes

1. **Install locally:**
   ```bash
   cd nova-sdk-skills
   ./setup.sh
   ```

2. **Test with Claude Code:**
   ```bash
   claude
   # Ask relevant questions to trigger your journey
   ```

3. **Verify:**
   - Skill activates correctly
   - Journey files are readable
   - Code examples are correct
   - Links work properly

## Journey Writing Guidelines

### Code Examples

- **Always include working code** - Users should be able to copy-paste
- **Use realistic examples** - Not toy data
- **Show both minimal and full examples**
- **Include imports and setup**

Example:
```python
# ✅ Good - Complete and runnable
from amzn_nova_customization_sdk import *

runtime = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=4)
customizer = NovaModelCustomizer(model=Model.NOVA_LITE_2, ...)
result = customizer.train(job_name="my-job")

# ❌ Bad - Incomplete
customizer.train(job_name="my-job")
```

### Troubleshooting Sections

- **List common errors** users actually encounter
- **Provide specific solutions** not general advice
- **Include error messages** exactly as they appear
- **Show the fix** with code when applicable

### Writing Style

- **Use active voice** - "Run the command" not "The command should be run"
- **Be concise** - Get to the point quickly
- **Use examples** - Show don't tell
- **Anticipate questions** - Address "why" not just "how"

## Pull Request Process

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/improve-sft-journey
   ```
3. **Make your changes**
4. **Test thoroughly** (see Testing section)
5. **Commit with clear messages:**
   ```bash
   git commit -m "Add troubleshooting section for SFT OOM errors"
   ```
6. **Push and create PR:**
   ```bash
   git push origin feature/improve-sft-journey
   ```
7. **Describe your changes** in the PR description

## Journey Status

**Complete and Available:** 🎉 All 10 core journeys complete!
- ✅ Journey 1: Setup & Prerequisites
- ✅ Journey 2: Data Preparation
- ✅ Journey 3: SFT Training
- ✅ Journey 4: CPT Training
- ✅ Journey 6a: RFT Single-Turn
- ✅ Journey 6b: RFT Multi-Turn
- ✅ Journey 7: Evaluation
- ✅ Journey 8: Bedrock Deployment
- ✅ Journey 9: SageMaker Deployment
- ✅ Journey 10: Inference & Monitoring

**Future Enhancements:**
- 📝 Journey 5: DPO Training (preference-based optimization)
- 📝 Advanced: Data Mixing (Forge feature)
- 📝 Advanced: Multi-modal training

Contributions are welcome! Open an issue to discuss new journeys or improvements.

## Questions?

- **General questions**: Open a GitHub issue
- **Bug reports**: Open a GitHub issue with the "bug" label
- **Feature requests**: Open a GitHub issue with the "enhancement" label

## Code of Conduct

- Be respectful and constructive
- Help others learn
- Give credit where due
- Focus on improving the skill for all users

---

Thank you for contributing! 🙏
