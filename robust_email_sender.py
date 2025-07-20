#!/usr/bin/env python3
"""
Robust email sender that handles Gmail authentication issues
"""
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import threading
import time

# Load environment variables
load_dotenv()

class RobustEmailSender:
    """A robust email sender that handles Gmail authentication properly"""
    
    def __init__(self):
        self.mail_server = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
        self.mail_port = int(os.getenv('MAIL_PORT', 587))
        self.mail_username = os.getenv('MAIL_USERNAME')
        self.mail_password = os.getenv('MAIL_PASSWORD')
        self.mail_sender = os.getenv('MAIL_DEFAULT_SENDER')
        
    def send_payment_confirmation_email(self, user_email, user_name, plan_type, amount_paid, payment_id, subscription_end):
        """Send payment confirmation email for Pro Plan subscription"""
        print(f"💳 RobustEmailSender: Sending payment confirmation to {user_email}")
        
        if not all([self.mail_username, self.mail_password, self.mail_sender]):
            print("❌ Email configuration incomplete")
            return False
        
        try:
            # Plan details
            plan_name = "Pro Plan - Monthly" if plan_type == "monthly" else "Pro Plan - Yearly"
            amount_inr = amount_paid / 100  # Convert paise to rupees
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.mail_sender
            msg['To'] = user_email
            msg['Subject'] = f'🎉 Payment Successful - Welcome to {plan_name}!'
            
            # Plain text version
            text_body = f"""
Payment Successful - Welcome to Pro Plan!

Hi {user_name},

🎉 Congratulations! Your payment has been processed successfully and your Pro Plan subscription is now active!

Payment Details:
- Plan: {plan_name}
- Amount Paid: ₹{amount_inr:,.0f}
- Payment ID: {payment_id}
- Subscription Valid Until: {subscription_end.strftime('%B %d, %Y')}

🚀 Your Pro Features Are Now Active:

✅ Unlimited Property Predictions - No daily limits anymore!
✅ Advanced AI Photo Analysis - Get deeper insights from property images
✅ Market Trend Reports - Access exclusive market analysis
✅ Priority Email Support - Get faster responses to your queries
✅ Export to PDF - Download your predictions and reports
✅ Advanced Analytics Dashboard - Track your prediction history
✅ No Advertisements - Enjoy a clean, ad-free experience

🎯 What's Next?
1. Visit your dashboard: http://localhost:5000/dashboard
2. Try unlimited predictions with your new Pro status
3. Explore advanced features in the Pro section
4. Contact us anytime for priority support

Thank you for choosing House Price Predictor Pro! We're excited to help you make smarter real estate decisions.

Best regards,
The House Price Predictor Team

---
Need help? Reply to this email or contact our priority support.
Payment ID: {payment_id}
            """
            
            # HTML version
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Payment Successful - Pro Plan Active</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
                    .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }}
                    .header {{ background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 40px 30px; text-align: center; }}
                    .header h1 {{ margin: 0; font-size: 28px; font-weight: 700; }}
                    .header p {{ margin: 10px 0 0 0; opacity: 0.9; font-size: 16px; }}
                    .content {{ padding: 40px 30px; }}
                    .success-badge {{ background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 15px 25px; border-radius: 25px; display: inline-block; margin-bottom: 30px; font-weight: 600; }}
                    .payment-details {{ background: #f8f9fa; border-radius: 10px; padding: 25px; margin: 30px 0; border-left: 5px solid #28a745; }}
                    .detail-row {{ display: flex; justify-content: space-between; margin: 10px 0; padding: 8px 0; border-bottom: 1px solid #e9ecef; }}
                    .detail-label {{ font-weight: 600; color: #495057; }}
                    .detail-value {{ color: #28a745; font-weight: 600; }}
                    .features {{ background: #f8f9fa; border-radius: 10px; padding: 25px; margin: 30px 0; }}
                    .feature {{ display: flex; align-items: center; margin: 15px 0; }}
                    .feature-icon {{ width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 18px; background: linear-gradient(135deg, #28a745, #20c997); color: white; }}
                    .cta-button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-weight: 600; margin: 20px 0; transition: transform 0.3s ease; }}
                    .footer {{ background: #f8f9fa; padding: 30px; text-align: center; color: #666; font-size: 14px; }}
                    .crown {{ color: #ffd700; font-size: 24px; }}
                </style>
            </head>
            <body>
                <div style="padding: 20px;">
                    <div class="container">
                        <div class="header">
                            <h1>🎉 Payment Successful!</h1>
                            <p>Welcome to Pro Plan - You're now a premium member!</p>
                        </div>
                        
                        <div class="content">
                            <div style="text-align: center;">
                                <div class="success-badge">
                                    ✅ Payment Processed Successfully
                                </div>
                            </div>
                            
                            <p style="font-size: 18px; color: #333; line-height: 1.6;">
                                Hi <strong>{user_name}</strong>,
                            </p>
                            <p style="color: #666; line-height: 1.6;">
                                Congratulations! Your payment has been processed successfully and your <strong>{plan_name}</strong> subscription is now active!
                            </p>
                            
                            <div class="payment-details">
                                <h3 style="margin-top: 0; color: #333; display: flex; align-items: center;">
                                    <span class="crown">👑</span> Payment Details
                                </h3>
                                <div class="detail-row">
                                    <span class="detail-label">Plan:</span>
                                    <span class="detail-value">{plan_name}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">Amount Paid:</span>
                                    <span class="detail-value">₹{amount_inr:,.0f}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">Payment ID:</span>
                                    <span class="detail-value">{payment_id}</span>
                                </div>
                                <div class="detail-row" style="border-bottom: none;">
                                    <span class="detail-label">Valid Until:</span>
                                    <span class="detail-value">{subscription_end.strftime('%B %d, %Y')}</span>
                                </div>
                            </div>
                            
                            <div class="features">
                                <h3 style="margin-top: 0; color: #333;">🚀 Your Pro Features Are Now Active:</h3>
                                
                                <div class="feature">
                                    <div class="feature-icon">∞</div>
                                    <div>
                                        <strong>Unlimited Predictions</strong><br>
                                        <span style="color: #666;">No daily limits anymore!</span>
                                    </div>
                                </div>
                                
                                <div class="feature">
                                    <div class="feature-icon">📸</div>
                                    <div>
                                        <strong>Advanced AI Photo Analysis</strong><br>
                                        <span style="color: #666;">Get deeper insights from property images</span>
                                    </div>
                                </div>
                                
                                <div class="feature">
                                    <div class="feature-icon">📊</div>
                                    <div>
                                        <strong>Market Trend Reports</strong><br>
                                        <span style="color: #666;">Access exclusive market analysis</span>
                                    </div>
                                </div>
                                
                                <div class="feature">
                                    <div class="feature-icon">⚡</div>
                                    <div>
                                        <strong>Priority Support</strong><br>
                                        <span style="color: #666;">Get faster responses to your queries</span>
                                    </div>
                                </div>
                                
                                <div class="feature">
                                    <div class="feature-icon">📄</div>
                                    <div>
                                        <strong>Export to PDF</strong><br>
                                        <span style="color: #666;">Download predictions and reports</span>
                                    </div>
                                </div>
                                
                                <div class="feature">
                                    <div class="feature-icon">🚫</div>
                                    <div>
                                        <strong>Ad-Free Experience</strong><br>
                                        <span style="color: #666;">Enjoy a clean, premium interface</span>
                                    </div>
                                </div>
                            </div>
                            
                            <div style="text-align: center; margin: 40px 0;">
                                <a href="http://localhost:5000/dashboard" class="cta-button">
                                    👑 Access Your Pro Dashboard
                                </a>
                            </div>
                        </div>
                        
                        <div class="footer">
                            <p><strong>Thank you for choosing House Price Predictor Pro!</strong></p>
                            <p>We're excited to help you make smarter real estate decisions.</p>
                            <p style="margin-top: 20px; font-size: 12px; color: #999;">
                                Payment ID: {payment_id}<br>
                                Need help? Reply to this email for priority support.
                            </p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Attach both versions
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"📤 Attempt {attempt + 1}: Connecting to Gmail SMTP...")
                    
                    # Create SMTP connection
                    server = smtplib.SMTP(self.mail_server, self.mail_port)
                    server.starttls()  # Enable TLS encryption
                    
                    print(f"🔐 Attempt {attempt + 1}: Logging in...")
                    server.login(self.mail_username, self.mail_password)
                    
                    print(f"📧 Attempt {attempt + 1}: Sending payment confirmation...")
                    server.send_message(msg)
                    server.quit()
                    
                    print(f"✅ Payment confirmation email sent successfully to {user_email}")
                    return True
                    
                except smtplib.SMTPAuthenticationError as e:
                    print(f"❌ Attempt {attempt + 1}: Authentication failed: {e}")
                    if attempt == max_retries - 1:
                        print("🔧 All attempts failed. Please check your Gmail App Password setup.")
                        return False
                    time.sleep(2)  # Wait before retry
                    
                except Exception as e:
                    print(f"❌ Attempt {attempt + 1}: Email sending failed: {e}")
                    if attempt == max_retries - 1:
                        return False
                    time.sleep(2)  # Wait before retry
            
            return False
            
        except Exception as e:
            print(f"❌ Failed to prepare payment confirmation email: {e}")
            return False

    def send_welcome_email_robust(self, user_email, user_name):
        """Send welcome email using direct SMTP (more reliable than Flask-Mail)"""
        print(f"🚀 RobustEmailSender: Sending welcome email to {user_email}")
        
        if not all([self.mail_username, self.mail_password, self.mail_sender]):
            print("❌ Email configuration incomplete")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.mail_sender
            msg['To'] = user_email
            msg['Subject'] = 'Welcome to AI-Powered House Price Predictor! 🏠'
            
            # Plain text version
            text_body = f"""
Welcome to AI-Powered House Price Predictor!

Hi {user_name},

Welcome to the future of real estate analysis! We're thrilled to have you join our community of smart property investors and homebuyers.

With our advanced AI technology, you now have access to instant, accurate property valuations at your fingertips.

🚀 What You Can Do Now:

🧠 AI-Powered Predictions - Get instant property valuations using advanced machine learning
⚡ Lightning Fast Results - Receive accurate price estimates in seconds, not days
🎯 Market-Based Analysis - Predictions based on real market data and trends
🔒 Secure & Private - Your data is protected with enterprise-grade security

💡 Pro Tips for Getting Started:
- Try our Interactive Map to explore different neighborhoods
- Use the Photo Analyzer to get insights from property images
- Check your Analytics Dashboard to track your prediction history
- Chat with our AI Assistant for personalized real estate advice

Get started now: http://localhost:5000/

Need help? Our AI assistant is available 24/7!

Best regards,
The House Price Predictor Team
            """
            
            # HTML version
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Welcome to House Price Predictor</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
                    .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 30px; text-align: center; }}
                    .header h1 {{ margin: 0; font-size: 28px; font-weight: 700; }}
                    .header p {{ margin: 10px 0 0 0; opacity: 0.9; font-size: 16px; }}
                    .content {{ padding: 40px 30px; }}
                    .welcome-text {{ font-size: 18px; color: #333; line-height: 1.6; margin-bottom: 30px; }}
                    .features {{ background: #f8f9fa; border-radius: 10px; padding: 25px; margin: 30px 0; }}
                    .feature {{ display: flex; align-items: center; margin: 15px 0; }}
                    .feature-icon {{ width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 18px; }}
                    .ai-icon {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; }}
                    .speed-icon {{ background: linear-gradient(135deg, #4facfe, #00f2fe); color: white; }}
                    .accuracy-icon {{ background: linear-gradient(135deg, #43e97b, #38f9d7); color: white; }}
                    .security-icon {{ background: linear-gradient(135deg, #fa709a, #fee140); color: white; }}
                    .cta-button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-weight: 600; margin: 20px 0; transition: transform 0.3s ease; }}
                    .footer {{ background: #f8f9fa; padding: 30px; text-align: center; color: #666; font-size: 14px; }}
                </style>
            </head>
            <body>
                <div style="padding: 20px;">
                    <div class="container">
                        <div class="header">
                            <h1>🏠 Welcome to House Price Predictor!</h1>
                            <p>Your AI-Powered Real Estate Companion</p>
                        </div>
                        
                        <div class="content">
                            <div class="welcome-text">
                                <p>Hi <strong>{user_name}</strong>,</p>
                                <p>Welcome to the future of real estate analysis! We're thrilled to have you join our community of smart property investors and homebuyers.</p>
                                <p>With our advanced AI technology, you now have access to instant, accurate property valuations at your fingertips.</p>
                            </div>
                            
                            <div class="features">
                                <h3 style="margin-top: 0; color: #333;">🚀 What You Can Do Now:</h3>
                                
                                <div class="feature">
                                    <div class="feature-icon ai-icon">🧠</div>
                                    <div>
                                        <strong>AI-Powered Predictions</strong><br>
                                        <span style="color: #666;">Get instant property valuations using advanced machine learning</span>
                                    </div>
                                </div>
                                
                                <div class="feature">
                                    <div class="feature-icon speed-icon">⚡</div>
                                    <div>
                                        <strong>Lightning Fast Results</strong><br>
                                        <span style="color: #666;">Receive accurate price estimates in seconds, not days</span>
                                    </div>
                                </div>
                                
                                <div class="feature">
                                    <div class="feature-icon accuracy-icon">🎯</div>
                                    <div>
                                        <strong>Market-Based Analysis</strong><br>
                                        <span style="color: #666;">Predictions based on real market data and trends</span>
                                    </div>
                                </div>
                                
                                <div class="feature">
                                    <div class="feature-icon security-icon">🔒</div>
                                    <div>
                                        <strong>Secure & Private</strong><br>
                                        <span style="color: #666;">Your data is protected with enterprise-grade security</span>
                                    </div>
                                </div>
                            </div>
                            
                            <div style="text-align: center; margin: 40px 0;">
                                <a href="http://localhost:5000/" class="cta-button">
                                    🚀 Start Predicting Property Prices
                                </a>
                            </div>
                        </div>
                        
                        <div class="footer">
                            <p><strong>Need Help?</strong></p>
                            <p>Our AI assistant is available 24/7 to help you navigate the platform and answer your real estate questions.</p>
                            <p style="margin-top: 30px; font-size: 12px; color: #999;">
                                You're receiving this email because you signed up for House Price Predictor.<br>
                                This is an automated welcome message.
                            </p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Attach both versions
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"📤 Attempt {attempt + 1}: Connecting to Gmail SMTP...")
                    
                    # Create SMTP connection
                    server = smtplib.SMTP(self.mail_server, self.mail_port)
                    server.starttls()  # Enable TLS encryption
                    
                    print(f"🔐 Attempt {attempt + 1}: Logging in...")
                    server.login(self.mail_username, self.mail_password)
                    
                    print(f"📧 Attempt {attempt + 1}: Sending email...")
                    server.send_message(msg)
                    server.quit()
                    
                    print(f"✅ Welcome email sent successfully to {user_email}")
                    return True
                    
                except smtplib.SMTPAuthenticationError as e:
                    print(f"❌ Attempt {attempt + 1}: Authentication failed: {e}")
                    if attempt == max_retries - 1:
                        print("🔧 All attempts failed. Please check your Gmail App Password setup.")
                        return False
                    time.sleep(2)  # Wait before retry
                    
                except Exception as e:
                    print(f"❌ Attempt {attempt + 1}: Email sending failed: {e}")
                    if attempt == max_retries - 1:
                        return False
                    time.sleep(2)  # Wait before retry
            
            return False
            
        except Exception as e:
            print(f"❌ Failed to prepare welcome email: {e}")
            return False

# Global instance
robust_email_sender = RobustEmailSender()

def send_payment_confirmation_background(user_email, user_name, plan_type, amount_paid, payment_id, subscription_end):
    """Send payment confirmation email in background thread"""
    def send_async():
        try:
            robust_email_sender.send_payment_confirmation_email(
                user_email, user_name, plan_type, amount_paid, payment_id, subscription_end
            )
        except Exception as e:
            print(f"❌ Background payment email thread failed: {e}")
    
    print(f"🧵 Starting payment confirmation email thread for {user_email}")
    email_thread = threading.Thread(target=send_async)
    email_thread.daemon = True
    email_thread.start()

def send_welcome_email_background(user_email, user_name):
    """Send welcome email in background thread using robust sender"""
    def send_async():
        try:
            robust_email_sender.send_welcome_email_robust(user_email, user_name)
        except Exception as e:
            print(f"❌ Background email thread failed: {e}")
    
    print(f"🧵 Starting robust background email thread for {user_email}")
    email_thread = threading.Thread(target=send_async)
    email_thread.daemon = True
    email_thread.start()

if __name__ == "__main__":
    # Test the robust email sender
    print("🧪 Testing Robust Email Sender")
    print("=" * 40)
    
    test_email = os.getenv('MAIL_USERNAME')
    test_name = "Test User"
    
    if robust_email_sender.send_welcome_email_robust(test_email, test_name):
        print("✅ Robust email sender test passed!")
    else:
        print("❌ Robust email sender test failed!")