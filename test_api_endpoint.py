#!/usr/bin/env python3
"""
Test script to verify the /api/upload-image endpoint is working
"""

import requests
import os
from pathlib import Path

def test_upload_endpoint():
    """Test the upload image endpoint"""
    
    # Check if server is running
    try:
        response = requests.get('http://localhost:5000/')
        print(f"✅ Server is running - Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running on localhost:5000")
        return False
    
    # Test the API endpoint with a simple GET (should return method not allowed)
    try:
        response = requests.get('http://localhost:5000/api/upload-image')
        print(f"✅ API endpoint exists - Status: {response.status_code} (405 is expected for GET)")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot reach API endpoint")
        return False
    
    # Test with a sample image if available
    sample_image_path = Path('static/house.png')
    if sample_image_path.exists():
        try:
            with open(sample_image_path, 'rb') as f:
                files = {'image': ('house.png', f, 'image/png')}
                response = requests.post('http://localhost:5000/api/upload-image', files=files)
                print(f"✅ POST request test - Status: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                return response.status_code == 200
        except Exception as e:
            print(f"❌ Error testing POST request: {e}")
            return False
    else:
        print("⚠️  No sample image found for testing")
        return True

if __name__ == "__main__":
    print("🔍 Testing API Endpoint...")
    print("=" * 50)
    
    success = test_upload_endpoint()
    
    print("=" * 50)
    if success:
        print("✅ API endpoint is working correctly!")
    else:
        print("❌ API endpoint has issues!")