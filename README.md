# 🏠 AI-Powered House Price Predictor

An intelligent web application that predicts house prices using advanced machine learning algorithms and provides comprehensive real estate analysis tools.

## 🚀 Features

### 🤖 Core AI Features
- **Machine Learning Predictions**: Advanced ensemble models for accurate price estimation
- **Photo Analysis**: AI-powered property image analysis using Google Gemini
- **Interactive Maps**: Location-based property insights
- **Real-time Chat**: AI assistant for real estate queries

### 💳 Pro Plan Features
- **Unlimited Predictions**: No daily limits for Pro users
- **Advanced Analytics**: Detailed market trend reports
- **Priority Support**: Faster response times
- **Export to PDF**: Download predictions and reports
- **Ad-free Experience**: Clean, premium interface

### 🔐 User Management
- **Google OAuth**: Secure authentication
- **User Dashboard**: Personalized prediction history
- **Profile Management**: User preferences and settings

### 💰 Payment Integration
- **Razorpay Gateway**: Secure payment processing for Indian users
- **Multiple Plans**: Monthly (₹499) and Yearly (₹4999) subscriptions
- **Email Notifications**: Automated payment confirmations
- **Subscription Management**: Track and manage Pro subscriptions

## 🛠️ Technology Stack

### Backend
- **Flask**: Python web framework
- **SQLAlchemy**: Database ORM
- **PostgreSQL**: Production database
- **scikit-learn**: Machine learning models
- **Pandas & NumPy**: Data processing

### AI & ML
- **Google Gemini AI**: Advanced language model for analysis
- **Random Forest**: Ensemble learning for price prediction
- **Computer Vision**: Image analysis capabilities
- **Natural Language Processing**: Chat functionality

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Interactive user interface
- **Bootstrap**: UI components and styling
- **Chart.js**: Data visualization

### Integrations
- **Google OAuth**: Authentication
- **Razorpay**: Payment processing
- **Gmail SMTP**: Email notifications
- **Google Maps**: Location services

## 📦 Installation

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Gmail account (for SMTP)
- Google Cloud account (for OAuth & Gemini AI)
- Razorpay account (for payments)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/vraj1033/House_Price_Predictor.git
   cd House_Price_Predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv house_predictor_env
   source house_predictor_env/bin/activate  # On Windows: house_predictor_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   - Copy `.env.example` to `.env`
   - Configure all required environment variables:
     - Database URL
     - Google OAuth credentials
     - Gemini API key
     - Gmail SMTP settings
     - Razorpay API keys

5. **Database Setup**
   ```bash
   python init_db.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Flask Configuration
SECRET_KEY=your-secret-key
FLASK_ENV=development

# Database
DATABASE_URL=your-postgresql-url

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Gemini AI
GEMINI_API_KEY=your-gemini-api-key

# Email (Gmail SMTP)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password

# Razorpay Payment Gateway
RAZORPAY_KEY_ID=your-razorpay-key-id
RAZORPAY_KEY_SECRET=your-razorpay-secret-key
```

## 📊 Database Schema

### Users Table
- User authentication and profile information
- Google OAuth integration
- Subscription status tracking

### Predictions Table
- Historical prediction data
- User-specific prediction history
- Performance analytics

### Pro Subscriptions Table
- Payment tracking
- Subscription management
- Razorpay integration data

## 🎯 API Endpoints

### Authentication
- `GET /auth/google` - Google OAuth login
- `GET /auth/google/callback` - OAuth callback
- `POST /logout` - User logout

### Predictions
- `POST /predict` - Generate house price prediction
- `GET /dashboard` - User prediction history
- `POST /analyze-photo` - AI photo analysis

### Payments
- `POST /create-payment-order` - Create Razorpay order
- `POST /verify-payment` - Verify payment signature
- `GET /pro-plan` - Pro plan subscription page

### Chat & AI
- `POST /chat` - AI assistant chat
- `GET /map` - Interactive property map

## 🚀 Deployment

### Production Checklist
- [ ] Set `FLASK_ENV=production`
- [ ] Configure production database
- [ ] Set up SSL certificates
- [ ] Configure domain for OAuth callbacks
- [ ] Switch to Razorpay live keys
- [ ] Set up monitoring and logging

### Recommended Platforms
- **Heroku**: Easy deployment with PostgreSQL addon
- **Railway**: Modern deployment platform
- **DigitalOcean**: VPS with full control
- **AWS/GCP**: Enterprise-grade hosting

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Test specific components:
```bash
python test_payment_email.py  # Test email notifications
python payment_gateway.py     # Test payment integration
```

## 📈 Performance

### Optimization Features
- **Caching**: Model predictions cached for performance
- **Background Processing**: Email sending in separate threads
- **Database Indexing**: Optimized queries for large datasets
- **Image Compression**: Efficient photo processing

## 🔒 Security

### Security Measures
- **Environment Variables**: Sensitive data protection
- **OAuth Authentication**: Secure user login
- **Payment Verification**: Razorpay signature validation
- **Input Sanitization**: XSS and injection prevention
- **HTTPS Enforcement**: Secure data transmission

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vraj Patel**
- GitHub: [@vraj1033](https://github.com/vraj1033)
- Email: vpk2634@gmail.com

## 🙏 Acknowledgments

- **Google Gemini AI** for advanced language processing
- **Razorpay** for seamless payment integration
- **Flask Community** for excellent documentation
- **scikit-learn** for machine learning capabilities

## 📞 Support

For support and queries:
- 📧 Email: vpk2634@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/vraj1033/House_Price_Predictor/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/vraj1033/House_Price_Predictor/discussions)

---

⭐ **Star this repository if you found it helpful!**

🚀 **Ready to predict house prices with AI? Get started now!**