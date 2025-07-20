#!/usr/bin/env bash
# exit on error
set -o errexit

echo "🔧 Starting build process..."

# Upgrade pip and install build tools
echo "📦 Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

# Install dependencies one by one to isolate issues
echo "📦 Installing core dependencies..."
pip install Flask==2.3.2
pip install Flask-SQLAlchemy==3.0.5
pip install Flask-Login==0.6.2
pip install Flask-Mail==0.9.1
pip install Werkzeug==2.3.6
pip install psycopg2-binary==2.9.6
pip install Authlib==1.2.1
pip install requests==2.31.0
pip install python-dotenv==1.0.0
pip install razorpay==1.3.0
pip install gunicorn==20.1.0

echo "📦 Installing ML dependencies (may take longer)..."
pip install --only-binary=all numpy==1.24.3
pip install --only-binary=all pandas==2.0.2
pip install --only-binary=all scikit-learn==1.2.2
pip install joblib==1.2.0

echo "📦 Installing remaining dependencies..."
pip install google-generativeai==0.2.2
pip install google-auth==2.22.0
pip install google-auth-oauthlib==1.0.0
pip install Pillow==9.5.0
pip install python-dateutil==2.8.2

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p static/uploads
mkdir -p instance
touch static/uploads/.gitkeep

echo "✅ Build completed successfully!"