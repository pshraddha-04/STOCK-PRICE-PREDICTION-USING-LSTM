# 📧 Email System Setup Guide

## Overview
Your contact form is already integrated with an email system! Here's how to set it up:

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
python setup_email.py
```

### Option 2: Manual Setup
1. Copy `.env.example` to `.env`
2. Fill in your email credentials in the `.env` file

## 📋 Gmail Setup Instructions

### Step 1: Enable 2-Factor Authentication
1. Go to your Google Account settings
2. Enable 2-Factor Authentication if not already enabled

### Step 2: Generate App Password
1. Go to Google Account → Security → 2-Step Verification
2. Click "App passwords" at the bottom
3. Select "Mail" and your device
4. Copy the generated 16-character password

### Step 3: Configure Environment Variables
Create a `.env` file with:
```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-16-char-app-password
RECIPIENT_EMAIL=where-to-receive-messages@gmail.com
```

## 🔧 Installation
```bash
pip install -r requirement.txt
```

## ▶️ Running the Application
```bash
python scripts/app.py
```

## 🌐 Access Your Application
- Open: http://localhost:5000
- Contact form: http://localhost:5000/contact.html

## ✨ Features
- ✅ Form validation
- ✅ Loading states
- ✅ Success/error notifications
- ✅ Email delivery
- ✅ Responsive design

## 🔍 Testing
1. Fill out the contact form
2. Submit the form
3. Check your recipient email for the message

## 🛠 Troubleshooting

### Common Issues:
- **"Email configuration missing"**: Set up your `.env` file
- **"Authentication failed"**: Use App Password, not regular password
- **"Network error"**: Check if Flask server is running

### Debug Mode:
The Flask app runs in debug mode by default, check console for detailed error messages.

## 📁 File Structure
```
├── scripts/app.py          # Flask backend with email endpoint
├── web/contact.html        # Contact form frontend
├── web/js/contact.js       # Form handling JavaScript
├── web/css/contact_styles.css # Contact page styles
├── .env                    # Email configuration (create this)
└── .env.example           # Template for email config
```

Your email system is ready to use! 🎉