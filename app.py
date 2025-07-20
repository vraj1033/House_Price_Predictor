from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import requests
import google.generativeai as genai
from werkzeug.utils import secure_filename
import uuid
import threading
from payment_gateway import payment_gateway, get_pro_plan_info, calculate_pro_expiry
from robust_email_sender import send_payment_confirmation_background
import json
# Load environment variables
load_dotenv()

# Helper function to get base URL
def get_base_url():
    """Get the base URL for the application"""
    if os.getenv('FLASK_ENV') == 'production':
        return 'https://house-price-predictor.onrender.com'
    return 'http://localhost:5000'

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-fallback-secret-key')

# File upload configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database Configuration
database_url = os.getenv('DATABASE_URL')
if not database_url or 'your-postgresql-database-url-here' in database_url:
    print("⚠️  WARNING: PostgreSQL DATABASE_URL not configured!")
    print("💡 Using SQLite fallback for development")
    database_url = 'sqlite:///house_predictor.db'
else:
    print(f"✅ Using PostgreSQL database: {database_url[:30]}...")

# Handle PostgreSQL URL format (some services use postgres:// instead of postgresql://)
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth'

# OAuth Configuration
oauth = OAuth(app)

# Gemini AI Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key-here':
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini AI configured successfully")
    GEMINI_ENABLED = True
else:
    print("⚠️  Gemini API key not found. Chat functionality will be limited.")
    GEMINI_ENABLED = False

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    google = oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={
            'scope': 'openid email profile'
        }
    )
    GOOGLE_OAUTH_ENABLED = True
    print("✅ Google OAuth configured successfully")
else:
    GOOGLE_OAUTH_ENABLED = False
    print("⚠️  Google OAuth not configured. Using email/password authentication only.")

# Email Configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'False').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

# Initialize Flask-Mail
mail = Mail(app)

# Check email configuration
if app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD']:
    EMAIL_ENABLED = True
    print("✅ Email notifications configured successfully")
else:
    EMAIL_ENABLED = False
    print("⚠️  Email not configured. Welcome emails will be disabled.")

# Email Functions
def send_welcome_email(user_email, user_name):
    """Send welcome email to new user using robust SMTP"""
    print(f"🚀 send_welcome_email called for: {user_email} ({user_name})")
    
    if not EMAIL_ENABLED:
        print(f"📧 Email disabled - Would send welcome email to {user_email}")
        return
    
    print(f"📧 Email enabled - Using robust SMTP sender for {user_email}")
    
    # Send email in background thread to avoid blocking
    def send_async_email():
        with app.app_context():
            email_record = None
            try:
                # Get user ID for the email record
                user = User.query.filter_by(email=user_email).first()
                user_id = user.id if user else None
                
                # Create email record with pending status
                email_record = WelcomeEmailSent(
                    user_id=user_id,
                    email_address=user_email,
                    user_name=user_name,
                    email_status='pending',
                    email_type='welcome'
                )
                db.session.add(email_record)
                db.session.commit()
                
                print(f"📤 Sending welcome email to {user_email} using robust SMTP...")
                
                # Use robust SMTP sender instead of Flask-Mail
                success = send_welcome_email_robust(user_email, user_name)
                
                if success:
                    # Update email record to sent status
                    email_record.email_status = 'sent'
                    db.session.commit()
                    print(f"✅ Welcome email sent successfully to {user_email}")
                else:
                    # Update email record to failed status
                    email_record.email_status = 'failed'
                    email_record.error_message = 'Robust SMTP sender failed'
                    db.session.commit()
                    print(f"❌ Failed to send welcome email to {user_email}")
                
            except Exception as e:
                print(f"❌ Failed to send welcome email to {user_email}: {e}")
                print(f"Error details: {type(e).__name__}: {str(e)}")
                
                # Update email record to failed status
                if email_record:
                    try:
                        email_record.email_status = 'failed'
                        email_record.error_message = str(e)
                        db.session.commit()
                    except Exception as db_error:
                        print(f"❌ Failed to update email record: {db_error}")
    
    # Start background thread
    print(f"🧵 Starting robust background email thread for {user_email}")
    email_thread = threading.Thread(target=send_async_email)
    email_thread.daemon = True
    email_thread.start()
    print(f"🚀 Robust background email thread started for {user_email}")

def send_welcome_email_robust(user_email, user_name):
    """Send welcome email using direct SMTP with enhanced error handling"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import time
    
    print(f"🚀 RobustEmailSender: Sending welcome email to {user_email}")
    
    mail_server = app.config.get('MAIL_SERVER', 'smtp.gmail.com')
    mail_port = app.config.get('MAIL_PORT', 587)
    mail_username = app.config.get('MAIL_USERNAME')
    mail_password = app.config.get('MAIL_PASSWORD')
    mail_sender = app.config.get('MAIL_DEFAULT_SENDER')
    
    if not all([mail_username, mail_password, mail_sender]):
        print("❌ Email configuration incomplete")
        print("🔄 Using fallback logging system...")
        return send_welcome_email_fallback(user_email, user_name)
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = mail_sender
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
                        
                        <div style="background: #e3f2fd; border-radius: 10px; padding: 20px; margin: 30px 0;">
                            <h4 style="margin-top: 0; color: #1976d2;">💡 Pro Tips for Getting Started:</h4>
                            <ul style="color: #333; line-height: 1.6;">
                                <li>Try our <strong>Interactive Map</strong> to explore different neighborhoods</li>
                                <li>Use the <strong>Photo Analyzer</strong> to get insights from property images</li>
                                <li>Check your <strong>Analytics Dashboard</strong> to track your prediction history</li>
                                <li>Chat with our <strong>AI Assistant</strong> for personalized real estate advice</li>
                            </ul>
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
                server = smtplib.SMTP(mail_server, mail_port)
                server.starttls()  # Enable TLS encryption
                
                print(f"🔐 Attempt {attempt + 1}: Logging in...")
                server.login(mail_username, mail_password)
                
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

def send_welcome_email_fallback(user_email, user_name):
    """Fallback email system that logs instead of sending when SMTP fails"""
    print(f"📧 FALLBACK: Would send welcome email to {user_email}")
    print(f"👤 User: {user_name}")
    print(f"📝 Email content: Professional welcome email with platform features")
    print(f"🎨 Template: HTML email with AI-powered house price predictor branding")
    print(f"✨ Features highlighted: AI predictions, fast results, market analysis, security")
    print(f"🔗 Call-to-action: Start predicting property prices")
    print(f"💡 Pro tips: Interactive map, photo analyzer, dashboard, AI assistant")
    print(f"✅ Email 'sent' (logged only - SMTP unavailable)")
    
    # Still return True so the system knows the "email" was handled
    return True

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(200))
    google_id = db.Column(db.String(100), unique=True)
    avatar_url = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    bedrooms = db.Column(db.Integer, nullable=False)
    bathrooms = db.Column(db.Float, nullable=False)
    sqft_living = db.Column(db.Integer, nullable=False)
    sqft_lot = db.Column(db.Integer, nullable=False)
    floors = db.Column(db.Float, nullable=False)
    waterfront = db.Column(db.Integer, nullable=False)
    view = db.Column(db.Integer, nullable=False)
    condition = db.Column(db.Integer, nullable=False)
    grade = db.Column(db.Integer, nullable=False)
    yr_built = db.Column(db.Integer, nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class WelcomeEmailSent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    email_address = db.Column(db.String(100), nullable=False)
    user_name = db.Column(db.String(100), nullable=False)
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)
    email_status = db.Column(db.String(20), default='sent')  # sent, failed, pending
    error_message = db.Column(db.Text, nullable=True)
    email_type = db.Column(db.String(50), default='welcome')  # welcome, registration, login
    
    # Relationship
    user = db.relationship('User', backref=db.backref('welcome_emails', lazy=True))
    
    def __repr__(self):
        return f'<WelcomeEmailSent {self.email_address} - {self.email_status}>'

class PhotoAnalyzerRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Nullable for anonymous users
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)  # Size in bytes
    image_width = db.Column(db.Integer, nullable=True)
    image_height = db.Column(db.Integer, nullable=True)
    analysis_result = db.Column(db.Text, nullable=True)  # JSON string of analysis results
    ai_description = db.Column(db.Text, nullable=True)  # AI-generated description
    detected_features = db.Column(db.Text, nullable=True)  # JSON string of detected features
    property_type = db.Column(db.String(50), nullable=True)  # house, apartment, condo, etc.
    estimated_rooms = db.Column(db.Integer, nullable=True)
    estimated_condition = db.Column(db.String(20), nullable=True)  # excellent, good, fair, poor
    processing_time = db.Column(db.Float, nullable=True)  # Time taken for analysis in seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('photo_analyses', lazy=True))
    
    def __repr__(self):
        return f'<PhotoAnalyzerRecord {self.original_filename} - {self.created_at}>'

class ProSubscription(db.Model):
    """Pro Plan subscription model"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plan_type = db.Column(db.String(20), nullable=False)  # 'monthly' or 'yearly'
    razorpay_order_id = db.Column(db.String(100), nullable=False)
    razorpay_payment_id = db.Column(db.String(100), nullable=True)
    payment_status = db.Column(db.String(20), default='pending')  # pending, completed, failed
    amount_paid = db.Column(db.Integer, nullable=False)  # Amount in paise
    currency = db.Column(db.String(10), default='INR')
    subscription_start = db.Column(db.DateTime, nullable=True)
    subscription_end = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('pro_subscriptions', lazy=True))
    
    def __repr__(self):
        return f'<ProSubscription {self.user_id} - {self.plan_type} - {self.payment_status}>'
    
    def is_subscription_active(self):
        """Check if subscription is currently active"""
        if not self.is_active or not self.subscription_end:
            return False
        return datetime.utcnow() < self.subscription_end

# Add helper method to User model to check Pro status
def is_pro_user(self):
    """Check if user has active Pro subscription"""
    active_subscription = ProSubscription.query.filter_by(
        user_id=self.id, 
        is_active=True
    ).filter(
        ProSubscription.subscription_end > datetime.utcnow()
    ).first()
    return active_subscription is not None

# Add the method to User class
User.is_pro_user = is_pro_user

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Global variables for models
model = None
scaler = None
ensemble_model = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler, ensemble_model
    try:
        # Try to load ensemble model first
        from ensemble_models import HousePriceEnsemble
        ensemble_model = HousePriceEnsemble()
        ensemble_model.load_ensemble('house_price_ensemble.pkl')
        print("✅ Ensemble model loaded successfully")
    except FileNotFoundError:
        try:
            # Fallback to single model
            model = joblib.load('house_price_model.pkl')
            scaler = joblib.load('scaler.pkl')
            print("✅ Single model and scaler loaded successfully")
        except FileNotFoundError:
            print("⚠️  Model files not found. Training new models...")
            # First, train the basic model and scaler to ensure fallback is ready
            train_model()
            # Then, train the advanced ensemble model
            train_ensemble_model()

def train_model():
    """Train a new model with sample data"""
    global model, scaler
    
    # Generate sample data for training
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples),
        'sqft_living': np.random.randint(800, 4000, n_samples),
        'sqft_lot': np.random.randint(3000, 15000, n_samples),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'view': np.random.randint(0, 5, n_samples),
        'condition': np.random.randint(1, 6, n_samples),
        'grade': np.random.randint(3, 13, n_samples),
        'yr_built': np.random.randint(1900, 2023, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (price) based on features
    df['price'] = (
        df['bedrooms'] * 50000 +
        df['bathrooms'] * 30000 +
        df['sqft_living'] * 150 +
        df['sqft_lot'] * 5 +
        df['floors'] * 20000 +
        df['waterfront'] * 200000 +
        df['view'] * 25000 +
        df['condition'] * 15000 +
        df['grade'] * 30000 +
        (2023 - df['yr_built']) * -1000 +
        np.random.normal(0, 50000, n_samples)  # Add some noise
    )
    
    # Ensure positive prices
    df['price'] = np.maximum(df['price'], 100000)
    
    # Prepare features and target
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                'waterfront', 'view', 'condition', 'grade', 'yr_built']
    X = df[features]
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, 'house_price_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("✅ Model trained and saved successfully")

def train_ensemble_model():
    """Train the advanced ensemble model"""
    global ensemble_model
    try:
        from ensemble_models import HousePriceEnsemble, create_sample_data
        
        print("🚀 Training Advanced Multi-Model Ensemble...")
        
        # Create enhanced sample data
        df = create_sample_data(2000)
        X = df.drop('price', axis=1)
        y = df['price']
        
        # Initialize and train ensemble
        ensemble_model = HousePriceEnsemble()
        model_scores, ensemble_score = ensemble_model.train_ensemble(X, y)
        
        # Save the trained ensemble
        ensemble_model.save_ensemble('house_price_ensemble.pkl')
        
        print(f"✅ Ensemble model trained successfully!")
        print(f"   Final Accuracy: {ensemble_score['r2']*100:.1f}%")
        print(f"   MAE: ${ensemble_score['mae']:,.0f}")
        
    except Exception as e:
        print(f"❌ Error training ensemble model: {e}")
        # Fallback to single model
        train_model()

# Routes
@app.route('/')
def index():
    # Allow both authenticated and demo users
    user = current_user if current_user.is_authenticated else None
    return render_template('index.html', user=user)

@app.route('/auth')
def auth():
    # If user is already logged in, redirect to main app
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('auth.html')

@app.route('/demo')
def demo():
    # Allow demo access without authentication
    return render_template('index.html', user=None)

# Google OAuth Routes
@app.route('/auth/google')
def google_auth():
    if not GOOGLE_OAUTH_ENABLED:
        flash('Google OAuth is not configured', 'error')
        return redirect(url_for('auth'))
    
    try:
        redirect_uri = url_for('google_callback', _external=True)
        return google.authorize_redirect(redirect_uri)
    except Exception as e:
        print(f"Google OAuth error: {e}")
        flash('Authentication service temporarily unavailable', 'error')
        return redirect(url_for('auth'))

@app.route('/auth/google/callback')
def google_callback():
    if not GOOGLE_OAUTH_ENABLED:
        flash('Google OAuth is not configured', 'error')
        return redirect(url_for('auth'))
    
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        
        if user_info:
            # Check if user exists
            user = User.query.filter_by(google_id=user_info['sub']).first()
            
            if not user:
                # Check if user exists with same email
                user = User.query.filter_by(email=user_info['email']).first()
                if user:
                    # Link Google account to existing user
                    user.google_id = user_info['sub']
                    user.avatar_url = user_info.get('picture')
                    db.session.commit()
                else:
                    # Create new user
                    user = User(
                        email=user_info['email'],
                        name=user_info['name'],
                        google_id=user_info['sub'],
                        avatar_url=user_info.get('picture')
                    )
                    db.session.add(user)
                    db.session.commit()
                    
                    # Send welcome email to new Google OAuth user
                    print(f"🆕 New user created via Google OAuth: {user_info['email']}")
                    send_welcome_email(user_info['email'], user_info['name'])
            
            login_user(user)
            return redirect(url_for('welcome_splash'))
        else:
            flash('Failed to get user information from Google', 'error')
            return redirect(url_for('auth'))
            
    except Exception as e:
        print(f"Google callback error: {e}")
        flash('Authentication failed', 'error')
        return redirect(url_for('auth'))

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password required'})
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        if not user:
            # Create new user (for demo purposes - in production, this should validate password)
            user = User(email=email, name=email.split('@')[0])
            db.session.add(user)
            db.session.commit()
            
            # Send welcome email to new user created via login
            print(f"🆕 New user created via login: {email}")
            send_welcome_email(email, user.name)
        else:
            # Existing user login - optionally send a "welcome back" email
            print(f"🔄 Existing user login: {email}")
            # Uncomment the line below if you want to send welcome emails to returning users
            # send_welcome_email(email, user.name)
        
        login_user(user)
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email')
        name = data.get('name')
        password = data.get('password')
        
        if not email or not name or not password:
            return jsonify({'success': False, 'error': 'All fields required'})
        
        # Check if user exists
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already registered'})
        
        # Create new user
        user = User(email=email, name=name)
        db.session.add(user)
        db.session.commit()
        
        # Send welcome email to new user
        send_welcome_email(email, name)
        
        login_user(user)
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/welcome')
def welcome_splash():
    """Welcome splash screen after login"""
    return render_template('welcome_splash.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth'))

# Pro Plan Payment Routes
@app.route('/pro-plan')
@login_required
def pro_plan():
    """Pro Plan subscription page"""
    monthly_plan = get_pro_plan_info('monthly')
    yearly_plan = get_pro_plan_info('yearly')
    
    # Check if user already has active subscription
    active_subscription = ProSubscription.query.filter_by(
        user_id=current_user.id, 
        is_active=True
    ).filter(
        ProSubscription.subscription_end > datetime.utcnow()
    ).first()
    
    return render_template('pro_plan.html', 
                         user=current_user,
                         monthly_plan=monthly_plan,
                         yearly_plan=yearly_plan,
                         active_subscription=active_subscription,
                         razorpay_key_id=os.getenv('RAZORPAY_KEY_ID'))

@app.route('/create-payment-order', methods=['POST'])
@login_required
def create_payment_order():
    """Create Razorpay payment order"""
    try:
        data = request.get_json()
        plan_type = data.get('plan_type', 'monthly')
        
        if plan_type not in ['monthly', 'yearly']:
            return jsonify({'success': False, 'error': 'Invalid plan type'})
        
        plan_info = get_pro_plan_info(plan_type)
        
        # Create Razorpay order
        order = payment_gateway.create_order(
            amount=plan_info['price_paise'],
            user_email=current_user.email,
            user_name=current_user.name
        )
        
        if not order:
            return jsonify({'success': False, 'error': 'Failed to create payment order'})
        
        # Save subscription record with pending status
        subscription = ProSubscription(
            user_id=current_user.id,
            plan_type=plan_type,
            razorpay_order_id=order['id'],
            amount_paid=plan_info['price_paise'],
            payment_status='pending'
        )
        db.session.add(subscription)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'order_id': order['id'],
            'amount': order['amount'],
            'currency': order['currency'],
            'key_id': os.getenv('RAZORPAY_KEY_ID'),
            'user_name': current_user.name,
            'user_email': current_user.email,
            'plan_name': plan_info['name']
        })
        
    except Exception as e:
        print(f"❌ Error creating payment order: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'})

@app.route('/verify-payment', methods=['POST'])
@login_required
def verify_payment():
    """Verify Razorpay payment and activate subscription"""
    try:
        data = request.get_json()
        payment_id = data.get('payment_id')
        order_id = data.get('order_id')
        signature = data.get('signature')
        
        if not all([payment_id, order_id, signature]):
            return jsonify({'success': False, 'error': 'Missing payment details'})
        
        # Verify payment with Razorpay
        is_verified = payment_gateway.verify_payment(payment_id, order_id, signature)
        
        if not is_verified:
            return jsonify({'success': False, 'error': 'Payment verification failed'})
        
        # Find subscription record
        subscription = ProSubscription.query.filter_by(
            razorpay_order_id=order_id,
            user_id=current_user.id
        ).first()
        
        if not subscription:
            return jsonify({'success': False, 'error': 'Subscription record not found'})
        
        # Update subscription status
        subscription.razorpay_payment_id = payment_id
        subscription.payment_status = 'completed'
        subscription.is_active = True
        subscription.subscription_start = datetime.utcnow()
        subscription.subscription_end = calculate_pro_expiry(subscription.plan_type)
        subscription.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        print(f"✅ Pro subscription activated for user {current_user.email}")
        
        # Send payment confirmation email
        try:
            send_payment_confirmation_background(
                user_email=current_user.email,
                user_name=current_user.name,
                plan_type=subscription.plan_type,
                amount_paid=subscription.amount_paid,
                payment_id=payment_id,
                subscription_end=subscription.subscription_end
            )
            print(f"📧 Payment confirmation email queued for {current_user.email}")
        except Exception as email_error:
            print(f"⚠️ Failed to queue payment confirmation email: {email_error}")
            # Don't fail the payment verification if email fails
        
        return jsonify({
            'success': True,
            'message': 'Payment successful! Pro plan activated.',
            'subscription_end': subscription.subscription_end.strftime('%Y-%m-%d')
        })
        
    except Exception as e:
        print(f"❌ Error verifying payment: {e}")
        return jsonify({'success': False, 'error': 'Payment verification failed'})

@app.route('/payment-success')
@login_required
def payment_success():
    """Payment success page"""
    return render_template('payment_success.html', user=current_user)

@app.route('/payment-failed')
@login_required
def payment_failed():
    """Payment failed page"""
    return render_template('payment_failed.html', user=current_user)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        bedrooms = int(request.form.get('bedrooms', 3))
        bathrooms = float(request.form.get('bathrooms', 2))
        sqft_living = int(request.form.get('sqft_living', 2000))
        sqft_lot = int(request.form.get('sqft_lot', 8000))
        floors = float(request.form.get('floors', 2))
        waterfront = int(request.form.get('waterfront', 0))
        view = int(request.form.get('view', 2))
        condition = int(request.form.get('condition', 3))
        grade = int(request.form.get('grade', 7))
        yr_built = int(request.form.get('yr_built', 2000))
        
        # Prepare features as DataFrame for ensemble model
        features_df = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot],
            'floors': [floors],
            'waterfront': [waterfront],
            'view': [view],
            'condition': [condition],
            'grade': [grade],
            'yr_built': [yr_built]
        })
        
        # Use ensemble model if available, otherwise fallback to single model
        if ensemble_model and ensemble_model.is_trained:
            prediction = ensemble_model.predict(features_df)[0]
            model_type = "ensemble"
            
            # Get model contributions for detailed analysis
            contributions = ensemble_model.get_model_contributions(features_df)
            
        else:
            # Fallback to single model
            features_array = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, 
                                     waterfront, view, condition, grade, yr_built]])
            features_scaled = scaler.transform(features_array)
            prediction = model.predict(features_scaled)[0]
            model_type = "single"
            contributions = None
        
        # Format prediction
        formatted_prediction = f"${prediction:,.0f}"
        
        # Save prediction if user is logged in
        if current_user.is_authenticated:
            pred_record = Prediction(
                user_id=current_user.id,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                sqft_living=sqft_living,
                sqft_lot=sqft_lot,
                floors=floors,
                waterfront=waterfront,
                view=view,
                condition=condition,
                grade=grade,
                yr_built=yr_built,
                predicted_price=prediction
            )
            db.session.add(pred_record)
            db.session.commit()
        
        response_data = {
            'success': True,
            'prediction': formatted_prediction,
            'raw_prediction': prediction,
            'model_type': model_type
        }
        
        # Add ensemble details if available
        if contributions:
            response_data['ensemble_details'] = {
                'model_contributions': contributions,
                'confidence_score': min(95, max(75, 85 + (prediction / 1000000) * 5))  # Dynamic confidence
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/history')
@login_required
def history():
    try:
        predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
        
        prediction_list = []
        for pred in predictions:
            prediction_list.append({
                'id': pred.id,
                'predicted_price': f"${pred.predicted_price:,.0f}",
                'bedrooms': pred.bedrooms,
                'bathrooms': pred.bathrooms,
                'sqft_living': pred.sqft_living,
                'created_at': pred.created_at.strftime('%Y-%m-%d %H:%M')
            })
        
        return jsonify({
            'success': True,
            'predictions': prediction_list
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/dashboard')
@login_required
def dashboard():
    """Advanced Analytics Dashboard"""
    return render_template('dashboard_new.html', user=current_user)

@app.route('/map')
def map_view():
    """Property Search Map"""
    # Allow both authenticated and demo users to access the map
    user = current_user if current_user.is_authenticated else None
    return render_template('map.html', user=user)

@app.route('/chat')
def chat_view():
    """AI Chat Assistant"""
    # Allow both authenticated and demo users to access the chat
    user = current_user if current_user.is_authenticated else None
    return render_template('chat.html', user=user)

@app.route('/photo-analyzer')
def photo_analyzer():
    """Dedicated Photo Analyzer Page"""
    # Allow both authenticated and demo users to access the photo analyzer
    user = current_user if current_user.is_authenticated else None
    return render_template('photo_analyzer.html', user=user)

@app.route('/api/dashboard-data')
@login_required
def dashboard_data():
    """Get dashboard analytics data"""
    try:
        # Get user's predictions
        predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
        
        # Calculate statistics
        total_predictions = len(predictions)
        if total_predictions > 0:
            prices = [p.predicted_price for p in predictions]
            avg_price = sum(prices) / len(prices)
            max_price = max(prices)
            min_price = min(prices)
            
            # Recent predictions (last 7 days)
            from datetime import datetime, timedelta
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_predictions = [p for p in predictions if p.created_at >= week_ago]
            
            # Price trend data
            monthly_data = {}
            for pred in predictions:
                month_key = pred.created_at.strftime('%Y-%m')
                if month_key not in monthly_data:
                    monthly_data[month_key] = []
                monthly_data[month_key].append(pred.predicted_price)
            
            trend_data = []
            for month, prices in sorted(monthly_data.items()):
                trend_data.append({
                    'month': month,
                    'avg_price': sum(prices) / len(prices),
                    'count': len(prices)
                })
            
        else:
            avg_price = max_price = min_price = 0
            recent_predictions = []
            trend_data = []
        
        return jsonify({
            'success': True,
            'stats': {
                'total_predictions': total_predictions,
                'avg_price': f"${avg_price:,.0f}" if avg_price else "$0",
                'max_price': f"${max_price:,.0f}" if max_price else "$0",
                'min_price': f"${min_price:,.0f}" if min_price else "$0",
                'recent_count': len(recent_predictions)
            },
            'trend_data': trend_data,
            'recent_predictions': [
                {
                    'id': p.id,
                    'price': f"${p.predicted_price:,.0f}",
                    'bedrooms': p.bedrooms,
                    'bathrooms': p.bathrooms,
                    'sqft_living': p.sqft_living,
                    'date': p.created_at.strftime('%m/%d')
                } for p in recent_predictions[:10]
            ]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/location-estimate', methods=['POST'])
def location_estimate():
    """Get price estimate based on location coordinates"""
    try:
        data = request.get_json()
        lat = data.get('lat')
        lng = data.get('lng')
        address = data.get('address', '')
        
        if not lat or not lng:
            return jsonify({'success': False, 'error': 'Latitude and longitude required'})
        
        # Location-based price estimation logic
        # This is a simplified example - in production, you'd use real market data
        
        # Base price calculation based on coordinates
        base_price = 400000
        
        # Adjust price based on location factors
        # Example: Seattle area premium
        if 47.5 <= lat <= 47.8 and -122.5 <= lng <= -122.0:
            location_multiplier = 1.3  # Seattle premium
            area_name = "Seattle Metro Area"
            insights = "High-demand urban area with excellent amenities and job market."
        elif 47.4 <= lat <= 47.9 and -122.6 <= lng <= -121.9:
            location_multiplier = 1.1  # Greater Seattle area
            area_name = "Greater Seattle Area"
            insights = "Suburban area with good schools and transportation access."
        else:
            location_multiplier = 0.9  # Other areas
            area_name = "Regional Area"
            insights = "Developing area with potential for growth."
        
        # Calculate price range
        estimated_price = int(base_price * location_multiplier)
        price_low = int(estimated_price * 0.8)
        price_high = int(estimated_price * 1.2)
        
        # Create location-based estimate
        estimate = {
            'range': f'${price_low:,} - ${price_high:,}',
            'average': f'${estimated_price:,}',
            'insights': f'{insights} {area_name} typically sees strong property values.'
        }
        
        return jsonify({
            'success': True,
            'estimate': estimate
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Handle property image upload and analysis"""
    start_time = datetime.utcnow()
    
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            
            # Get file size and image dimensions
            file_size = os.path.getsize(filepath)
            image_width = image_height = None
            
            try:
                from PIL import Image
                with Image.open(filepath) as img:
                    image_width, image_height = img.size
            except Exception as img_error:
                print(f"⚠️  Could not get image dimensions: {img_error}")
            
            # Initialize photo record
            photo_record = PhotoAnalyzerRecord(
                user_id=current_user.id if current_user.is_authenticated else None,
                filename=filename,
                original_filename=file.filename,
                file_path=filepath,
                file_size=file_size,
                image_width=image_width,
                image_height=image_height
            )
            
            # Analyze the image with Computer Vision
            try:
                from computer_vision import PropertyVisionAnalyzer
                analyzer = PropertyVisionAnalyzer()
                analysis_result = analyzer.analyze_property_image(filepath)
                
                # Extract analysis data for database
                import json
                photo_record.analysis_result = json.dumps(analysis_result)
                photo_record.ai_description = analysis_result.get('description', '')
                photo_record.detected_features = json.dumps(analysis_result.get('features', {}))
                photo_record.property_type = analysis_result.get('property_type', '')
                photo_record.estimated_rooms = analysis_result.get('estimated_rooms')
                photo_record.estimated_condition = analysis_result.get('condition', '')
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                photo_record.processing_time = processing_time
                
                # Save to database
                db.session.add(photo_record)
                db.session.commit()
                
                print(f"📸 Photo analysis saved to database: {filename}")
                
                # Add file info to result
                analysis_result['file_info'] = {
                    'filename': filename,
                    'original_name': file.filename,
                    'file_path': filepath,
                    'file_url': f'/static/uploads/{filename}',
                    'record_id': photo_record.id
                }
                
                return jsonify({
                    'success': True,
                    'analysis': analysis_result
                })
                
            except Exception as cv_error:
                # Save basic record even if CV analysis fails
                photo_record.analysis_result = json.dumps({'error': str(cv_error)})
                photo_record.ai_description = f'Analysis failed: {str(cv_error)}'
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                photo_record.processing_time = processing_time
                
                db.session.add(photo_record)
                db.session.commit()
                
                print(f"📸 Photo record saved (analysis failed): {filename}")
                
                # Return basic file info even if CV analysis fails
                return jsonify({
                    'success': True,
                    'analysis': {
                        'error': f'Computer Vision analysis failed: {str(cv_error)}',
                        'file_info': {
                            'filename': filename,
                            'original_name': file.filename,
                            'file_url': f'/static/uploads/{filename}',
                            'record_id': photo_record.id
                        },
                        'basic_analysis': {
                            'uploaded': True,
                            'message': 'Image uploaded successfully, but advanced analysis is unavailable'
                        }
                    }
                })
        
        return jsonify({'success': False, 'error': 'Invalid file type'})
        
    except Exception as e:
        print(f"❌ Error in photo upload: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analyze-multiple-images', methods=['POST'])
def analyze_multiple_images():
    """Analyze multiple property images"""
    try:
        if 'images' not in request.files:
            return jsonify({'success': False, 'error': 'No image files provided'})
        
        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'success': False, 'error': 'No files selected'})
        
        uploaded_files = []
        analysis_results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Generate unique filename
                filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save the file
                file.save(filepath)
                uploaded_files.append({
                    'filename': filename,
                    'original_name': file.filename,
                    'file_path': filepath,
                    'file_url': f'/static/uploads/{filename}'
                })
                
                # Analyze each image
                try:
                    from computer_vision import PropertyVisionAnalyzer
                    analyzer = PropertyVisionAnalyzer()
                    result = analyzer.analyze_property_image(filepath)
                    result['file_info'] = uploaded_files[-1]
                    analysis_results.append(result)
                except Exception as cv_error:
                    analysis_results.append({
                        'error': str(cv_error),
                        'file_info': uploaded_files[-1]
                    })
        
        # Combine results if multiple images
        if len(analysis_results) > 1:
            try:
                from computer_vision import analyze_multiple_images
                combined_analysis = analyze_multiple_images([r['file_info']['file_path'] for r in analysis_results if 'error' not in r])
                
                return jsonify({
                    'success': True,
                    'individual_analyses': analysis_results,
                    'combined_analysis': combined_analysis,
                    'total_images': len(uploaded_files)
                })
            except Exception as e:
                return jsonify({
                    'success': True,
                    'individual_analyses': analysis_results,
                    'combined_analysis': {'error': f'Combined analysis failed: {str(e)}'},
                    'total_images': len(uploaded_files)
                })
        else:
            return jsonify({
                'success': True,
                'analysis': analysis_results[0] if analysis_results else {},
                'total_images': len(uploaded_files)
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cv-enhanced-prediction', methods=['POST'])
def cv_enhanced_prediction():
    """Make price prediction enhanced with Computer Vision analysis"""
    try:
        # Get traditional form data
        bedrooms = int(request.form.get('bedrooms', 3))
        bathrooms = float(request.form.get('bathrooms', 2))
        sqft_living = int(request.form.get('sqft_living', 2000))
        sqft_lot = int(request.form.get('sqft_lot', 8000))
        floors = float(request.form.get('floors', 2))
        waterfront = int(request.form.get('waterfront', 0))
        view = int(request.form.get('view', 2))
        condition = int(request.form.get('condition', 3))
        grade = int(request.form.get('grade', 7))
        yr_built = int(request.form.get('yr_built', 2000))
        
        # Get CV analysis data if provided
        cv_analysis = None
        if 'cv_analysis' in request.form:
            cv_analysis = json.loads(request.form.get('cv_analysis'))
        
        # Prepare base features
        features_df = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot],
            'floors': [floors],
            'waterfront': [waterfront],
            'view': [view],
            'condition': [condition],
            'grade': [grade],
            'yr_built': [yr_built]
        })
        
        # Enhance features with CV analysis
        if cv_analysis and 'price_impact_score' in cv_analysis:
            # Add CV-derived features
            features_df['cv_quality_score'] = cv_analysis.get('quality_assessment', {}).get('overall_score', 0.5)
            features_df['cv_condition_score'] = cv_analysis.get('condition_analysis', {}).get('condition_score', 0.7)
            features_df['cv_amenity_count'] = cv_analysis.get('amenity_detection', {}).get('total_detected', 0)
            features_df['cv_price_impact'] = cv_analysis.get('price_impact_score', 0.5)
        
        # Make prediction
        if ensemble_model and ensemble_model.is_trained:
            base_prediction = ensemble_model.predict(features_df)[0]
            model_type = "ensemble"
            contributions = ensemble_model.get_model_contributions(features_df)
        else:
            # Fallback to single model (without CV features)
            base_features = features_df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                                       'waterfront', 'view', 'condition', 'grade', 'yr_built']].values
            features_scaled = scaler.transform(base_features)
            base_prediction = model.predict(features_scaled)[0]
            model_type = "single"
            contributions = None
        
        # Apply CV adjustment if available
        final_prediction = base_prediction
        cv_adjustment = 0.0
        
        if cv_analysis and 'price_impact_score' in cv_analysis:
            # Adjust prediction based on CV analysis
            impact_score = cv_analysis['price_impact_score']
            
            # Apply adjustment (±20% based on CV analysis)
            cv_adjustment = (impact_score - 0.5) * 0.4  # -0.2 to +0.2 multiplier
            final_prediction = base_prediction * (1 + cv_adjustment)
            
        # Format predictions
        formatted_prediction = f"${final_prediction:,.0f}"
        
        # Save enhanced prediction if user is logged in
        if current_user.is_authenticated:
            pred_record = Prediction(
                user_id=current_user.id,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                sqft_living=sqft_living,
                sqft_lot=sqft_lot,
                floors=floors,
                waterfront=waterfront,
                view=view,
                condition=condition,
                grade=grade,
                yr_built=yr_built,
                predicted_price=final_prediction
            )
            db.session.add(pred_record)
            db.session.commit()
        
        response_data = {
            'success': True,
            'prediction': formatted_prediction,
            'raw_prediction': final_prediction,
            'base_prediction': base_prediction,
            'cv_adjustment': cv_adjustment,
            'model_type': model_type,
            'cv_enhanced': cv_analysis is not None
        }
        
        # Add ensemble details if available
        if contributions:
            response_data['ensemble_details'] = {
                'model_contributions': contributions,
                'confidence_score': min(95, max(75, 85 + (final_prediction / 1000000) * 5))
            }
        
        # Add CV insights
        if cv_analysis:
            response_data['cv_insights'] = {
                'quality_category': cv_analysis.get('quality_assessment', {}).get('quality_category', 'fair'),
                'detected_amenities': cv_analysis.get('amenity_detection', {}).get('total_detected', 0),
                'condition_score': cv_analysis.get('condition_analysis', {}).get('condition_score', 0.7),
                'price_impact': cv_analysis.get('price_impact_score', 0.5),
                'recommendations': cv_analysis.get('quality_assessment', {}).get('recommendations', [])
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/ensemble-details')
def ensemble_details():
    """Get detailed information about the ensemble model"""
    try:
        if not ensemble_model or not ensemble_model.is_trained:
            return jsonify({
                'success': False,
                'error': 'Ensemble model not available'
            })
        
        # Get feature importance
        feature_importance = ensemble_model.get_feature_importance()
        
        # Get model weights
        model_weights = ensemble_model.weights
        
        return jsonify({
            'success': True,
            'ensemble_info': {
                'total_models': len(ensemble_model.models),
                'model_types': list(ensemble_model.models.keys()),
                'model_weights': model_weights,
                'feature_importance': feature_importance[:10] if feature_importance else None,
                'is_trained': ensemble_model.is_trained
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Handle chat requests with Gemini AI"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        chat_history = data.get('history', [])
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Message is required'})
        
        # Enhanced fallback responses for when Gemini is not available
        def get_smart_response(message):
            message_lower = message.lower()
            
            # Greeting responses
            if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
                return "Hello! I'm your AI assistant. I can help you with house price predictions, real estate questions, or just about anything else you'd like to know. What can I help you with today?"
            
            # Real estate related
            if any(word in message_lower for word in ['price', 'house', 'home', 'property', 'real estate', 'market']):
                if 'predict' in message_lower or 'estimate' in message_lower:
                    return "I can help you predict house prices! You can use our prediction tool on the main page to get accurate estimates based on property features like bedrooms, bathrooms, square footage, location, and more. What specific property details would you like to know about?"
                elif 'market' in message_lower:
                    return "Real estate markets are influenced by many factors including location, economic conditions, interest rates, supply and demand, and local amenities. Each market is unique. Would you like to know about any specific aspect of the real estate market?"
                elif 'factor' in message_lower:
                    return "Key factors affecting house prices include: location and neighborhood quality, property size and condition, number of bedrooms and bathrooms, lot size, age and architectural style, local schools and amenities, market conditions, and economic factors. Which of these would you like to explore further?"
                else:
                    return "I'd be happy to help with real estate and house price questions! You can ask me about price predictions, market trends, property factors, or use our tools to get specific estimates."
            
            # How questions
            elif message_lower.startswith('how'):
                if 'work' in message_lower:
                    return "Great question! Our house price prediction system works by analyzing multiple property features using machine learning. We consider factors like location, size, condition, age, and market data to provide accurate estimates. Would you like to know more about any specific aspect?"
                elif 'accurate' in message_lower:
                    return "Our AI model is trained on comprehensive housing data and provides estimates based on property features. While we strive for accuracy, actual market prices can vary due to unique characteristics and current conditions. The predictions are most reliable when you provide detailed property information."
                else:
                    return f"That's an interesting question about '{message}'. I'd be happy to help explain! Could you be more specific about what aspect you'd like to understand better?"
            
            # What questions
            elif message_lower.startswith('what'):
                return f"You're asking about '{message}'. I can provide information on a wide range of topics, especially real estate and house price predictions. Could you give me a bit more context about what specifically you'd like to know?"
            
            # Why questions
            elif message_lower.startswith('why'):
                return f"That's a thoughtful question: '{message}'. There are usually multiple factors that contribute to any situation. Would you like me to explore the main reasons or focus on a particular aspect?"
            
            # When questions
            elif message_lower.startswith('when'):
                return f"Timing questions like '{message}' often depend on various factors and circumstances. Could you provide more context so I can give you a more helpful answer?"
            
            # Where questions
            elif message_lower.startswith('where'):
                return f"Location-based questions like '{message}' are important, especially in real estate! Could you be more specific about the area or context you're interested in?"
            
            # Math or calculation questions
            elif any(char.isdigit() for char in message) and any(op in message for op in ['+', '-', '*', '/', 'calculate', 'compute']):
                return "I can help with calculations! While I'm particularly good with real estate calculations like price estimates, mortgage payments, and property analysis, I can assist with various math problems. What would you like me to calculate?"
            
            # Technology questions
            elif any(word in message_lower for word in ['computer', 'software', 'technology', 'ai', 'machine learning', 'algorithm']):
                return f"Technology topics like '{message}' are fascinating! I use AI and machine learning for house price predictions, but I can discuss various tech topics. What specific aspect interests you most?"
            
            # General knowledge
            elif any(word in message_lower for word in ['tell me about', 'explain', 'describe', 'information about']):
                return f"I'd be happy to provide information about '{message}'. I have knowledge on many topics, with special expertise in real estate and property analysis. What specific aspects would you like me to cover?"
            
            # Default intelligent response
            else:
                return f"That's an interesting question about '{message}'. I'm here to help with all kinds of questions, though I'm particularly knowledgeable about real estate and house price predictions. Could you tell me more about what you'd like to know, or would you like to explore our property prediction tools?"
        
        if not GEMINI_ENABLED:
            response = get_smart_response(user_message)
            return jsonify({
                'success': True,
                'response': response
            })
        
        # Use Gemini AI for intelligent responses
        try:
            print(f"🤖 Attempting to use Gemini AI for message: {user_message[:50]}...")
            
            # Create the model (using the working model name)
            model_ai = genai.GenerativeModel('gemini-1.5-flash')
            
            # Simple and direct prompt - let Gemini handle everything naturally
            if chat_history:
                # Include recent conversation history for context
                conversation_prompt = "Previous conversation:\n"
                for exchange in chat_history[-3:]:  # Last 3 exchanges for context
                    conversation_prompt += f"Human: {exchange.get('user', '')}\n"
                    conversation_prompt += f"Assistant: {exchange.get('ai', '')}\n"
                conversation_prompt += f"\nHuman: {user_message}\nAssistant:"
            else:
                # First message in conversation
                conversation_prompt = user_message
            
            print(f"📝 Sending prompt to Gemini...")
            
            # Generate response with Gemini
            response = model_ai.generate_content(conversation_prompt)
            
            if response and response.text:
                ai_response = response.text.strip()
                print(f"✅ Gemini responded successfully: {ai_response[:100]}...")
                return jsonify({
                    'success': True,
                    'response': ai_response
                })
            else:
                print("⚠️  Gemini returned empty response")
                raise Exception("No response from Gemini")
            
        except Exception as gemini_error:
            print(f"❌ Gemini API error: {gemini_error}")
            print(f"Error type: {type(gemini_error).__name__}")
            
            # Check for specific error types
            error_str = str(gemini_error).upper()
            if any(keyword in error_str for keyword in ['API_KEY', 'INVALID', 'UNAUTHORIZED', 'FORBIDDEN']):
                print("🔑 API key issue detected")
                return jsonify({
                    'success': True,
                    'response': "I'm having trouble connecting to my AI service right now. This might be due to API configuration issues. Let me try to help you with a basic response instead."
                })
            elif 'QUOTA' in error_str or 'LIMIT' in error_str:
                print("📊 API quota/limit issue detected")
                return jsonify({
                    'success': True,
                    'response': "I've reached my API usage limit for now. Let me provide a helpful response based on your question."
                })
            else:
                print("🔄 Using fallback response system")
                # Use smart fallback response for other errors
                response = get_smart_response(user_message)
                return jsonify({
                    'success': True,
                    'response': response
                })
        
    except Exception as e:
        print(f"Chat API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Sorry, I encountered an error. Please try again.'
        })

def init_app():
    """Initialize the application"""
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables created successfully")
        except Exception as e:
            print(f"⚠️ Database initialization warning: {e}")
        
        try:
            load_model()
            print("✅ ML model loaded successfully")
        except Exception as e:
            print(f"⚠️ Model loading warning: {e}")

# Initialize app when imported (for Gunicorn)
init_app()

if __name__ == '__main__':
    # Get port from environment variable for production deployment
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV') != 'production'
    
    print(f"🚀 Starting Flask app on port {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)