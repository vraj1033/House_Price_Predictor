#!/usr/bin/env python3
"""
Database initialization script for House Price Predictor
Run this script to set up the database and create initial data
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app, db, User, Prediction
    print("✅ Successfully imported Flask app and models")
except ImportError as e:
    print(f"❌ Error importing app: {e}")
    print("Make sure you've installed all requirements: pip install -r requirements.txt")
    sys.exit(1)

def test_database_connection():
    """Test PostgreSQL database connection"""
    try:
        with app.app_context():
            # Test connection (compatible with newer SQLAlchemy versions)
            with db.engine.connect() as connection:
                connection.execute(db.text('SELECT 1'))
            print("✅ PostgreSQL database connection successful")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Check your DATABASE_URL in .env file")
        print("💡 Make sure PostgreSQL server is running and accessible")
        return False

def init_database():
    """Initialize the database with tables and sample data"""
    
    with app.app_context():
        try:
            # Test connection first
            if not test_database_connection():
                return False
            
            # Create all tables
            db.create_all()
            print("✅ PostgreSQL database tables created successfully")
            
            # Check if we already have users
            user_count = User.query.count()
            if user_count > 0:
                print(f"📊 Database already has {user_count} users")
                return True
            
            # Create a demo user for testing
            demo_user = User(
                email='demo@example.com',
                name='Demo User',
                password_hash='demo123',  # In production, this should be hashed
                created_at=datetime.utcnow()
            )
            
            db.session.add(demo_user)
            db.session.commit()
            
            print("✅ Demo user created:")
            print("   Email: demo@example.com")
            print("   Password: demo123")
            
            # Create a sample prediction for the demo user
            sample_prediction = Prediction(
                user_id=demo_user.id,
                bedrooms=3,
                bathrooms=2.0,
                sqft_living=2000,
                sqft_lot=8000,
                floors=2.0,
                waterfront=0,
                view=2,
                condition=3,
                grade=7,
                yr_built=2000,
                predicted_price=450000.0,
                created_at=datetime.utcnow()
            )
            
            db.session.add(sample_prediction)
            db.session.commit()
            
            print("✅ Sample prediction created for demo user")
            return True
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            print("💡 Make sure your PostgreSQL database is accessible")
            db.session.rollback()
            return False

def check_environment():
    """Check if environment variables are properly set"""
    
    print("\n🔍 Checking environment configuration...")
    
    required_vars = ['SECRET_KEY', 'GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value or 'your-' in value.lower() or 'change' in value.lower():
            missing_vars.append(var)
        else:
            print(f"✅ {var}: {'*' * min(len(value), 20)}...")
    
    if missing_vars:
        print(f"\n⚠️  Warning: The following environment variables need to be configured:")
        for var in missing_vars:
            print(f"   - {var}")
        
        if 'GOOGLE_CLIENT_ID' in missing_vars or 'GOOGLE_CLIENT_SECRET' in missing_vars:
            print("\n📝 To set up Google OAuth:")
            print("   1. Go to https://console.cloud.google.com/")
            print("   2. Create OAuth 2.0 credentials")
            print("   3. Add redirect URI: http://localhost:5000/auth/google/callback")
            print("   4. Update your .env file with the credentials")
        
        print(f"\n💡 Edit the .env file to configure these variables")
        return False
    
    print("✅ All environment variables are configured")
    return True

def main():
    """Main initialization function"""
    
    print("🏠 House Price Predictor - Database Initialization")
    print("=" * 50)
    
    # Check environment
    env_ok = check_environment()
    
    # Initialize database
    print("\n🗄️ Initializing database...")
    try:
        init_database()
        print("✅ Database initialization completed successfully!")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    
    # Final status
    print("\n" + "=" * 50)
    if env_ok:
        print("🎉 Setup completed successfully!")
        print("\n🚀 You can now run the application:")
        print("   python app.py")
        print("\n🌐 Then visit:")
        print("   http://localhost:5000")
    else:
        print("⚠️  Setup completed with warnings")
        print("   The app will work, but Google OAuth won't function until configured")
        print("\n🚀 You can still run the application:")
        print("   python app.py")
        print("\n🌐 Then visit:")
        print("   http://localhost:5000/demo (for demo mode)")
    
    return True

if __name__ == '__main__':
    main()