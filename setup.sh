#!/bin/bash

# Nova Guide Skill - Setup Script
# This script installs the nova-guide skill into Claude Code

set -e

echo "🚀 Nova Guide Skill - Setup"
echo "======================================"
echo ""

# Get the repository path
REPO_PATH=$(cd "$(dirname "$0")" && pwd)
echo "📁 Repository location: $REPO_PATH"
echo ""

# Create Claude Code skills directory structure
SKILL_DIR="$HOME/.claude/skills/nova-guide"
echo "📂 Creating skill directory..."
mkdir -p "$SKILL_DIR"
echo "✅ Directory ready: $SKILL_DIR"
echo ""

# Copy and configure the skill file
SKILL_FILE="$SKILL_DIR/SKILL.md"
echo "📝 Installing skill file..."

# Replace <REPO_PATH> with actual path
sed "s|<REPO_PATH>|$REPO_PATH|g" SKILL.md > "$SKILL_FILE"

echo "✅ Skill installed: $SKILL_FILE"
echo ""

# Verify journey files exist
echo "🔍 Verifying journey files..."
JOURNEY_COUNT=$(find "$REPO_PATH/journeys" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
echo "✅ Found $JOURNEY_COUNT journey files"
echo ""

# Test if Claude Code is available
echo "🧪 Checking Claude Code installation..."
if command -v claude &> /dev/null; then
    echo "✅ Claude Code CLI found"
else
    echo "⚠️  Claude Code CLI not found in PATH"
    echo "   Install from: https://docs.anthropic.com/claude-code"
fi
echo ""

echo "======================================"
echo "✅ Installation complete!"
echo ""
echo "Skill installed at:"
echo "  $SKILL_FILE"
echo ""
echo "Next steps:"
echo "  1. Start Claude Code: claude"
echo "  2. Check skills: /skills (should see 'nova-guide')"
echo "  3. Try: 'How do I set up the Nova SDK?'"
echo "  4. Or: 'Show me a quick SFT training example'"
echo ""
echo "Troubleshooting:"
echo "  - View skill file: cat $SKILL_FILE"
echo "  - Check journeys: ls $REPO_PATH/journeys/"
echo "  - Review installation: cat $REPO_PATH/INSTALL.md"
echo ""
echo "Happy customizing! 🎉"
