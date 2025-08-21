#!/usr/bin/env python3
"""
Firebase Diagnostic Script (Fixed Version)
Helps troubleshoot Firebase authentication and connection issues
"""

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from google.auth.exceptions import DefaultCredentialsError
from google.auth import default

def check_service_account_file():
    """Check if service account file exists and is valid JSON"""
    print("🔍 Checking service account file...")
    
    cred_path = "escape-ujuzxr-firebase-adminsdk-he895-1a039bd95a.json"
    
    if not os.path.exists(cred_path):
        print(f"❌ Service account file not found: {cred_path}")
        return False
    
    try:
        with open(cred_path, 'r') as f:
            cred_data = json.load(f)
        
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        for field in required_fields:
            if field not in cred_data:
                print(f"❌ Missing required field: {field}")
                return False
        
        print(f"✅ Service account file is valid JSON")
        print(f"   Project ID: {cred_data.get('project_id')}")
        print(f"   Client Email: {cred_data.get('client_email')}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in service account file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading service account file: {e}")
        return False

def check_google_auth():
    """Check if Google Auth can find default credentials"""
    print("\n🔍 Checking Google Auth default credentials...")
    
    try:
        creds, project = default()
        print(f"✅ Default credentials found")
        print(f"   Project: {project}")
        print(f"   Credentials type: {type(creds).__name__}")
        return True
    except DefaultCredentialsError as e:
        print(f"❌ No default credentials found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking default credentials: {e}")
        return False

def test_firebase_initialization():
    """Test Firebase initialization with different methods (matching firebase_service.py logic)"""
    print("\n🔍 Testing Firebase initialization...")
    
    # Clean up any existing Firebase apps first
    try:
        for app in firebase_admin._apps.values():
            app.delete()
    except:
        pass
    
    # Method 1: Default credentials first (matching firebase_service.py logic)
    print("📝 Method 1: Default credentials (recommended)")
    try:
        firebase_admin.initialize_app()
        print("✅ Firebase initialized with default credentials")
        
        # Test connection
        db = firestore.client()
        test_doc = db.collection('_test').document('diagnostic')
        test_doc.get()
        print("✅ Firestore connection successful")
        return True
    except Exception as e:
        error_msg = str(e)
        if "database (default) does not exist" in error_msg:
            print("⚠️ Firestore database not set up. Please enable Firestore in Firebase Console.")
            print("   Visit: https://console.firebase.google.com/project/escape-ujuzxr/firestore")
        elif "404" in error_msg:
            print("⚠️ Firestore database not found. Please check your Firebase project configuration.")
        else:
            print(f"❌ Default credentials method failed: {e}")
        
        # Clean up and try method 2
        try:
            for app in firebase_admin._apps.values():
                app.delete()
        except:
            pass
    
    # Method 2: Service account file (fallback)
    print("\n📝 Method 2: Service account file (fallback)")
    try:
        cred_path = "escape-ujuzxr-firebase-adminsdk-he895-1a039bd95a.json"
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized with service account file")
            
            # Test connection
            db = firestore.client()
            test_doc = db.collection('_test').document('diagnostic')
            test_doc.get()
            print("✅ Firestore connection successful")
            return True
        else:
            print("❌ Service account file not found")
            return False
    except Exception as e:
        print(f"❌ Service account file method failed: {e}")
        return False

def check_environment():
    """Check environment variables and configuration"""
    print("\n🔍 Checking environment...")
    
    # Check for Google Cloud project
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if project_id:
        print(f"✅ GOOGLE_CLOUD_PROJECT: {project_id}")
    else:
        print("⚠️ GOOGLE_CLOUD_PROJECT not set")
    
    # Check for Firebase project
    firebase_project = os.getenv('FIREBASE_PROJECT_ID')
    if firebase_project:
        print(f"✅ FIREBASE_PROJECT_ID: {firebase_project}")
    else:
        print("⚠️ FIREBASE_PROJECT_ID not set")
    
    # Check for application default credentials
    adc_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if adc_path:
        print(f"✅ GOOGLE_APPLICATION_CREDENTIALS: {adc_path}")
        if os.path.exists(adc_path):
            print("✅ Application default credentials file exists")
        else:
            print("❌ Application default credentials file not found")
    else:
        print("⚠️ GOOGLE_APPLICATION_CREDENTIALS not set")

def test_firebase_service():
    """Test the actual Firebase service from firebase_service.py"""
    print("\n🔍 Testing Firebase service integration...")
    
    try:
        from firebase_service import get_firebase_service
        firebase_service = get_firebase_service()
        
        if firebase_service.db is None:
            print("❌ Firebase service failed to initialize")
            return False
        
        # Test basic operations
        test_sessions = firebase_service.list_sessions(limit=1)
        print(f"✅ Firebase service working: {len(test_sessions)} sessions found")
        return True
        
    except Exception as e:
        print(f"❌ Firebase service test failed: {e}")
        return False

def main():
    """Run all diagnostic checks"""
    print("🚀 Firebase Diagnostic Tool (Fixed Version)")
    print("=" * 50)
    
    # Check environment
    check_environment()
    
    # Check service account file
    sa_valid = check_service_account_file()
    
    # Check Google Auth
    auth_valid = check_google_auth()
    
    # Test Firebase initialization
    firebase_works = test_firebase_initialization()
    
    # Test Firebase service integration
    service_works = test_firebase_service()
    
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"Service Account File: {'✅ Valid' if sa_valid else '❌ Invalid/Missing'}")
    print(f"Google Auth: {'✅ Available' if auth_valid else '❌ Not Available'}")
    print(f"Firebase Connection: {'✅ Working' if firebase_works else '❌ Failed'}")
    print(f"Firebase Service: {'✅ Working' if service_works else '❌ Failed'}")
    
    if not firebase_works:
        print("\n🔧 TROUBLESHOOTING SUGGESTIONS:")
        if not sa_valid:
            print("1. Download a new service account key from Firebase Console")
            print("2. Ensure the file is named correctly and in the project root")
        if not auth_valid:
            print("3. Run 'gcloud auth application-default login' to set up default credentials")
        print("4. Check if your Firebase project has Firestore enabled")
        print("5. Verify your service account has the necessary permissions")
        print("6. Visit: https://console.firebase.google.com/project/escape-ujuzxr/firestore")

if __name__ == "__main__":
    main()
