#!/bin/bash
# CVLab-Kit Docker Entrypoint
# Handles code updates and startup

set -e

echo "🚀 CVLab-Kit Docker Entrypoint"

# Auto-update if enabled (default: true)
AUTO_UPDATE=${AUTO_UPDATE:-true}
GIT_REPO=${GIT_REPO:-https://github.com/yourusername/cvlab-kit.git}

# Initial clone if not a git repo
if [ "$AUTO_UPDATE" = "true" ] && [ ! -d ".git" ]; then
    echo "📦 Git repo not found, cloning from $GIT_REPO..."

    # Backup current files if any
    if [ "$(ls -A)" ]; then
        echo "⚠️  Directory not empty, backing up to /tmp/cvlab-kit-backup..."
        mkdir -p /tmp/cvlab-kit-backup
        cp -r . /tmp/cvlab-kit-backup/ 2>/dev/null || true
    fi

    # Clone repo
    git clone "$GIT_REPO" /tmp/cvlab-kit-clone

    # Move contents to current directory
    shopt -s dotglob
    mv /tmp/cvlab-kit-clone/* . 2>/dev/null || true
    rm -rf /tmp/cvlab-kit-clone

    echo "✅ Repository cloned successfully"
fi

if [ "$AUTO_UPDATE" = "true" ] && [ -d ".git" ]; then
    echo "📦 Checking for code updates..."

    # Fetch latest changes
    git fetch origin 2>/dev/null || echo "⚠️  Git fetch failed (offline?)"

    # Check if updates available
    LOCAL=$(git rev-parse HEAD 2>/dev/null)
    REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "$LOCAL")

    if [ "$LOCAL" != "$REMOTE" ]; then
        echo "⬇️  Updating code from origin/main..."
        if ! git pull origin main --ff-only 2>/dev/null; then
            echo "⚠️  Git pull failed, continuing with current version"
        fi

        # Sync dependencies if uv.lock changed
        if git diff --name-only HEAD@{1} HEAD 2>/dev/null | grep -q "uv.lock"; then
            echo "📚 Dependencies changed, running uv sync..."
            if ! uv sync --frozen 2>/dev/null; then
                echo "⚠️  uv sync failed, continuing with current version"
            fi
        fi
        # Rebuild frontend if frontend code changed
        if git diff --name-only HEAD@{1} HEAD 2>/dev/null | grep -q "web_helper/frontend/src"; then
            echo "🎨 Frontend changed, rebuilding..."
            cd web_helper/frontend && npm run build && cd ../.. || echo "⚠️  Frontend build failed"
        fi

        echo "✅ Code updated to $(git rev-parse --short HEAD)"
    else
        echo "✅ Code is up to date ($(git rev-parse --short HEAD))"
    fi
else
    echo "ℹ️  Auto-update disabled or not a git repo"
fi

# Ensure frontend is built (for zip deployments without git)
if [ ! -d "web_helper/frontend/dist" ] || [ ! -f "web_helper/frontend/dist/index.html" ]; then
    echo "🎨 Frontend not built, building now..."
    if [ -d "web_helper/frontend" ] && [ -f "web_helper/frontend/package.json" ]; then
        cd web_helper/frontend
        npm ci 2>/dev/null || npm install
        npm run build
        cd ../..
        echo "✅ Frontend built successfully"
    else
        echo "⚠️  Frontend source not found, skipping build"
    fi
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
