#!/usr/bin/env python3
"""
Update database to include Pro subscription table
"""
from app import app, db
from datetime import datetime

def update_database():
    """Create new tables for Pro subscription"""
    with app.app_context():
        try:
            print("🔄 Updating database for Pro Plan features...")
            
            # Create all tables (including new ProSubscription table)
            db.create_all()
            
            print("✅ Database updated successfully!")
            print("📊 New table added: ProSubscription")
            print("🎯 Pro Plan payment system is ready!")
            
        except Exception as e:
            print(f"❌ Error updating database: {e}")

if __name__ == "__main__":
    update_database()