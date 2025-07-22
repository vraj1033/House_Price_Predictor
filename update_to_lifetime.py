#!/usr/bin/env python3
"""
Script to update Pro subscriptions to lifetime access
"""

import os
import sys
from datetime import datetime, timedelta
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your app and models
from app import app, db, ProSubscription, User

def update_subscriptions_to_lifetime():
    """Update all active Pro subscriptions to lifetime"""
    with app.app_context():
        try:
            # Find all active Pro subscriptions
            active_subscriptions = ProSubscription.query.filter_by(
                is_active=True,
                payment_status='completed'
            ).all()
            
            print(f"Found {len(active_subscriptions)} active Pro subscriptions")
            
            for subscription in active_subscriptions:
                user = User.query.get(subscription.user_id)
                print(f"\nUpdating subscription for user: {user.name} ({user.email})")
                print(f"Current plan_type: {subscription.plan_type}")
                
                # Update to lifetime
                subscription.plan_type = 'lifetime'
                subscription.subscription_end = datetime.utcnow() + timedelta(days=36500)  # 100 years
                subscription.updated_at = datetime.utcnow()
                
                print(f"Updated to: {subscription.plan_type}")
                print(f"New expiry: {subscription.subscription_end}")
            
            # Commit changes
            db.session.commit()
            print(f"\n✅ Successfully updated {len(active_subscriptions)} subscriptions to lifetime!")
            
            # Verify the changes
            print("\n📋 Verification:")
            for subscription in active_subscriptions:
                user = User.query.get(subscription.user_id)
                print(f"- {user.name}: {subscription.plan_type} (expires: {subscription.subscription_end.strftime('%Y-%m-%d')})")
                
        except Exception as e:
            print(f"❌ Error updating subscriptions: {e}")
            db.session.rollback()

def show_current_subscriptions():
    """Show current Pro subscriptions"""
    with app.app_context():
        try:
            subscriptions = ProSubscription.query.filter_by(is_active=True).all()
            print(f"\n📋 Current Active Pro Subscriptions ({len(subscriptions)}):")
            print("-" * 60)
            
            for sub in subscriptions:
                user = User.query.get(sub.user_id)
                print(f"User: {user.name} ({user.email})")
                print(f"Plan: {sub.plan_type}")
                print(f"Status: {sub.payment_status}")
                print(f"Expires: {sub.subscription_end}")
                print(f"Active: {sub.is_active}")
                print("-" * 60)
                
        except Exception as e:
            print(f"❌ Error fetching subscriptions: {e}")

if __name__ == "__main__":
    print("🔄 Pro Subscription Lifetime Updater")
    print("=" * 50)
    
    # Show current subscriptions first
    show_current_subscriptions()
    
    # Ask for confirmation
    response = input("\n❓ Do you want to update all active Pro subscriptions to lifetime? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        update_subscriptions_to_lifetime()
        
        # Show updated subscriptions
        print("\n" + "=" * 50)
        show_current_subscriptions()
    else:
        print("❌ Update cancelled.")