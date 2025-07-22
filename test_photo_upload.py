#!/usr/bin/env python3
"""
Test photo upload functionality directly
"""

import requests
import os
from pathlib import Path

def test_photo_upload():
    """Test photo upload with actual image file"""
    
    # Check if we have a test image
    test_images = ['static/house.png', 'static/logo.png']
    test_image = None
    
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if not test_image:
        print("❌ No test image found. Creating a simple test...")
        return False
    
    print(f"🔍 Testing photo upload with: {test_image}")
    print("=" * 50)
    
    try:
        # Test the endpoint
        with open(test_image, 'rb') as f:
            files = {'image': (os.path.basename(test_image), f, 'image/png')}
            
            print("📤 Sending POST request to /api/upload-image...")
            response = requests.post(
                'http://localhost:5000/api/upload-image', 
                files=files,
                timeout=30
            )
            
            print(f"📥 Response Status: {response.status_code}")
            print(f"📥 Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"✅ Success: {data.get('success', False)}")
                    if data.get('success'):
                        print(f"📊 Analysis data keys: {list(data.get('analysis', {}).keys())}")
                    else:
                        print(f"❌ Error: {data.get('error', 'Unknown error')}")
                except Exception as json_error:
                    print(f"❌ JSON parsing error: {json_error}")
                    print(f"Raw response: {response.text[:200]}...")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                
        return response.status_code == 200
        
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_server_health():
    """Test basic server health"""
    print("🏥 Testing Server Health...")
    print("=" * 30)
    
    try:
        # Test basic route
        response = requests.get('http://localhost:5000/', timeout=5)
        print(f"✅ Home page: {response.status_code}")
        
        # Test photo analyzer page
        response = requests.get('http://localhost:5000/photo-analyzer', timeout=5)
        print(f"✅ Photo analyzer page: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Photo Upload Test")
    print("=" * 50)
    
    # Test server health first
    if test_server_health():
        print("\n")
        # Test photo upload
        success = test_photo_upload()
        
        print("\n" + "=" * 50)
        if success:
            print("✅ Photo upload test PASSED!")
        else:
            print("❌ Photo upload test FAILED!")
    else:
        print("❌ Server is not responding properly!")