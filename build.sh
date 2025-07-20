#!/usr/bin/env bash
# exit on error
set -o errexit

echo "🔧 Starting build process..."

# Upgrade pip and install build tools
echo "📦 Upgrading pip and build tools..."
python -m pip install --upgrade pip setuptools wheel

# Use requirements.txt with deployment-friendly flags
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt --prefer-binary --only-binary=:all: --no-build-isolation || {
    echo "⚠️  Failed with strict binary-only, trying with fallback..."
    pip install -r requirements.txt --prefer-binary --no-build-isolation
}

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p static/uploads
mkdir -p instance
touch static/uploads/.gitkeep

echo "✅ Build completed successfully!"