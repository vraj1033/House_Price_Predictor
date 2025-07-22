#!/usr/bin/env python3
"""
Razorpay Payment Gateway Integration for Pro Plan
Free to use in India with transaction-based pricing
"""
import razorpay
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

load_dotenv()

class PaymentGateway:
    """Razorpay payment gateway integration"""
    
    def __init__(self):
        # Razorpay credentials (get from https://dashboard.razorpay.com/)
        self.key_id = os.getenv('RAZORPAY_KEY_ID')
        self.key_secret = os.getenv('RAZORPAY_KEY_SECRET')
        
        if self.key_id and self.key_secret:
            self.client = razorpay.Client(auth=(self.key_id, self.key_secret))
            self.enabled = True
            print("✅ Razorpay payment gateway initialized")
        else:
            self.client = None
            self.enabled = False
            print("⚠️  Razorpay credentials not found - Payment gateway disabled")
    
    def create_order(self, amount, currency='INR', user_email=None, user_name=None):
        """
        Create a payment order
        
        Args:
            amount (int): Amount in paise (₹100 = 10000 paise)
            currency (str): Currency code (default: INR)
            user_email (str): User's email
            user_name (str): User's name
            
        Returns:
            dict: Order details or None if failed
        """
        if not self.enabled:
            return None
            
        try:
            order_data = {
                'amount': amount,  # Amount in paise
                'currency': currency,
                'receipt': f'pro_plan_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'notes': {
                    'plan': 'Pro Plan',
                    'user_email': user_email or 'unknown',
                    'user_name': user_name or 'unknown',
                    'duration': '1 month'
                }
            }
            
            order = self.client.order.create(data=order_data)
            print(f"✅ Payment order created: {order['id']}")
            return order
            
        except Exception as e:
            print(f"❌ Failed to create payment order: {e}")
            return None
    
    def verify_payment(self, payment_id, order_id, signature):
        """
        Verify payment signature
        
        Args:
            payment_id (str): Razorpay payment ID
            order_id (str): Razorpay order ID
            signature (str): Payment signature
            
        Returns:
            bool: True if payment is verified
        """
        if not self.enabled:
            return False
            
        try:
            params_dict = {
                'razorpay_order_id': order_id,
                'razorpay_payment_id': payment_id,
                'razorpay_signature': signature
            }
            
            # Verify signature
            self.client.utility.verify_payment_signature(params_dict)
            print(f"✅ Payment verified: {payment_id}")
            return True
            
        except Exception as e:
            print(f"❌ Payment verification failed: {e}")
            return False
    
    def get_payment_details(self, payment_id):
        """Get payment details from Razorpay"""
        if not self.enabled:
            return None
            
        try:
            payment = self.client.payment.fetch(payment_id)
            return payment
        except Exception as e:
            print(f"❌ Failed to fetch payment details: {e}")
            return None

# Pro Plan Configuration
PRO_PLAN_CONFIG = {
    'monthly': {
        'name': 'Pro Plan - Monthly',
        'price_inr': 499,  # ₹499 per month
        'price_paise': 49900,  # Amount in paise for Razorpay
        'duration_days': 30,
        'features': [
            'Unlimited Property Predictions',
            'Advanced AI Photo Analysis',
            'Market Trend Reports',
            'Priority Email Support',
            'Export Predictions to PDF',
            'Advanced Analytics Dashboard',
            'Property Investment Insights',
            'No Ads'
        ]
    },
    'yearly': {
        'name': 'Pro Plan - Yearly',
        'price_inr': 4999,  # ₹4999 per year (2 months free)
        'price_paise': 499900,
        'duration_days': 365,
        'features': [
            'All Monthly Pro Features',
            '2 Months Free (Best Value!)',
            'Quarterly Market Reports',
            'Phone Support',
            'Custom Property Alerts',
            'API Access (Limited)',
            'Priority Feature Requests'
        ]
    },
    'lifetime': {
        'name': 'Pro Plan - Lifetime',
        'price_inr': 99,  # ₹99 lifetime
        'price_paise': 9900,
        'duration_days': 36500,  # 100 years (effectively lifetime)
        'features': [
            'Everything in Free',
            'Data persistence across sessions',
            'AI House Predictor Cloud sync',
            'Cross-device synchronization',
            '5 device activations',
            'Lifetime updates',
            'Self-hosting option',
            'Priority support'
        ]
    }
}

def get_pro_plan_info(plan_type='monthly'):
    """Get pro plan information"""
    return PRO_PLAN_CONFIG.get(plan_type, PRO_PLAN_CONFIG['monthly'])

def calculate_pro_expiry(plan_type='monthly'):
    """Calculate pro plan expiry date"""
    plan_info = get_pro_plan_info(plan_type)
    return datetime.now() + timedelta(days=plan_info['duration_days'])

# Initialize global payment gateway
payment_gateway = PaymentGateway()

if __name__ == "__main__":
    # Test the payment gateway
    print("🧪 Testing Razorpay Payment Gateway")
    print("=" * 50)
    
    if payment_gateway.enabled:
        # Test order creation
        test_order = payment_gateway.create_order(
            amount=49900,  # ₹499
            user_email="test@example.com",
            user_name="Test User"
        )
        
        if test_order:
            print(f"✅ Test order created successfully!")
            print(f"   Order ID: {test_order['id']}")
            print(f"   Amount: ₹{test_order['amount']/100}")
            print(f"   Currency: {test_order['currency']}")
        else:
            print("❌ Failed to create test order")
    else:
        print("❌ Payment gateway not configured")
        print("💡 Add RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET to .env file")