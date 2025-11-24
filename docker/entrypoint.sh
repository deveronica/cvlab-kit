#!/bin/bash
# CVLab-Kit Docker Entrypoint
# Handles code updates and startup

set -e

echo "🚀 CVLab-Kit Docker Entrypoint"

# Auto-update if enabled (default: true)
AUTO_UPDATE=${AUTO_UPDATE:-true}

if [ "$AUTO_UPDATE" = "true" ] && [ -d ".git" ]; then
    echo "📦 Checking for code updates..."

    # Fetch latest changes
    git fetch origin 2>/dev/null || echo "⚠️  Git fetch failed (offline?)"

    # Check if updates available
    LOCAL=$(git rev-parse HEAD 2>/dev/null)
    REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "$LOCAL")

    if [ "$LOCAL" != "$REMOTE" ]; then
        echo "⬇️  Updating code from origin/main..."
        git pull origin main --ff-only 2>/dev/null || {
            echo "⚠️  Git pull failed, continuing with current version"
        }

        # Sync dependencies if uv.lock changed
        if git diff --name-only HEAD@{1} HEAD 2>/dev/null | grep -q "uv.lock"; then
            echo "📚 Dependencies changed, running uv sync..."
            uv sync --frozen || echo "⚠️  uv sync failed"
        fi

        echo "✅ Code updated to $(git rev-parse --short HEAD)"
    else
        echo "✅ Code is up to date ($(git rev-parse --short HEAD))"
    fi
else
    echo "ℹ️  Auto-update disabled or not a git repo"
fi

# Show version info
echo "📋 Version Info:"
echo "   Git: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "   Branch: $(git branch --show-current 2>/dev/null || echo 'N/A')"
echo "   Python: $(python --version 2>&1)"

echo ""
echo "🎯 Starting CVLab-Kit..."

# Execute the main command
exec "$@"
