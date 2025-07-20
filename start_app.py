#!/usr/bin/env python3
"""
Startup script for House Price Predictor with Computer Vision
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'), 
        ('sklearn', 'scikit-learn'),
        ('flask', 'Flask'),
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow'),
        ('joblib', 'joblib'),
        ('requests', 'requests')
    ]
    
    missing_packages = []
    
    for package_name, pip_name in required_packages:
        try:
            __import__(package_name)
            print(f"✅ {package_name} - OK")
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            missing_packages.append(pip_name)
    
    # Check optional packages
    optional_packages = [
        ('tensorflow', 'tensorflow'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm')
    ]
    
    print("\n🔍 Checking optional packages...")
    for package_name, pip_name in optional_packages:
        try:
            __import__(package_name)
            print(f"✅ {package_name} - OK")
        except ImportError:
            print(f"⚠️  {package_name} - Optional (some features may be limited)")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All required dependencies are available!")
    return True

def initialize_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'static/uploads',
        'instance'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}")

def start_application():
    """Start the Flask application"""
    print("\n🚀 Starting House Price Predictor...")
    print("=" * 60)
    
    try:
        # Import and run the Flask app
        from app import app
        
        print("🏠 House Price Predictor with Computer Vision")
        print("📊 Multi-Model Ensemble AI")
        print("📸 Advanced Image Analysis")
        print("🤖 Gemini AI Chat Integration")
        print("=" * 60)
        print("🌐 Server starting on http://localhost:5000")
        print("💡 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure all dependencies are installed")
        print("2. Check your .env file configuration")
        print("3. Ensure database connection is working")
        return False
    
    return True

if __name__ == "__main__":
    print("🏠 House Price Predictor - Startup Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot start application due to missing dependencies")
        sys.exit(1)
    
    # Initialize directories
    initialize_directories()
    
    # Start the application
    if not start_application():
        print("\n❌ Failed to start application")
        sys.exit(1)