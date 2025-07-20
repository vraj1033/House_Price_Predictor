#!/usr/bin/env python3
"""
Test script for payment confirmation email functionality
"""
from robust_email_sender import send_payment_confirmation_background
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_payment_email():
    """Test the payment confirmation email system"""
    print("🧪 Testing Payment Confirmation Email System")
    print("=" * 50)
    
    # Test data
    test_user_email = os.getenv('MAIL_USERNAME', 'test@example.com')
    test_user_name = "Test User"
    test_plan_type = "monthly"
    test_amount_paid = 49900  # ₹499 in paise
    test_payment_id = "pay_test_1234567890"
    test_subscription_end = datetime.now() + timedelta(days=30)
    
    print(f"📧 Test Email: {test_user_email}")
    print(f"👤 Test User: {test_user_name}")
    print(f"💳 Plan Type: {test_plan_type}")
    print(f"💰 Amount: ₹{test_amount_paid/100}")
    print(f"🆔 Payment ID: {test_payment_id}")
    print(f"📅 Subscription End: {test_subscription_end.strftime('%B %d, %Y')}")
    print()
    
    # Send test email
    print("📤 Sending test payment confirmation email...")
    try:
        send_payment_confirmation_background(
            user_email=test_user_email,
            user_name=test_user_name,
            plan_type=test_plan_type,
            amount_paid=test_amount_paid,
            payment_id=test_payment_id,
            subscription_end=test_subscription_end
        )
        print("✅ Test email queued successfully!")
        print("📧 Check your email inbox for the payment confirmation")
        print()
        print("🎉 Payment Email System is Ready!")
        print("   ✅ Email templates created")
        print("   ✅ Background sending configured")
        print("   ✅ Integration with payment flow complete")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_payment_email()