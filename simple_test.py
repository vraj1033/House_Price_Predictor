#!/usr/bin/env python3
"""
Simple test to check if the basic Flask routes work
"""

import requests
import time

def test_basic_routes():
    """Test basic Flask routes"""
    base_url = 'http://localhost:5000'
    
    routes_to_test = [
        '/',
        '/predict-page',
        '/photo-analyzer'
    ]
    
    print("🔍 Testing Basic Routes...")
    print("=" * 50)
    
    for route in routes_to_test:
        try:
            url = f"{base_url}{route}"
            print(f"Testing: {url}")
            
            response = requests.get(url, timeout=10)
            print(f"✅ Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"✅ Route {route} is working")
            else:
                print(f"⚠️  Route {route} returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error for {route}: {e}")
        except requests.exceptions.Timeout as e:
            print(f"❌ Timeout error for {route}: {e}")
        except Exception as e:
            print(f"❌ Error for {route}: {e}")
        
        print("-" * 30)
        time.sleep(1)  # Small delay between requests

def test_api_endpoint_simple():
    """Test the API endpoint with a simple request"""
    print("\n🔍 Testing API Endpoint...")
    print("=" * 50)
    
    try:
        # Test GET request (should return 405 Method Not Allowed)
        response = requests.get('http://localhost:5000/api/upload-image', timeout=10)
        print(f"GET request: Status {response.status_code} (405 expected)")
        
        # Test POST request without data (should return error but not crash)
        response = requests.post('http://localhost:5000/api/upload-image', timeout=10)
        print(f"POST request (no data): Status {response.status_code}")
        print(f"Response: {response.text[:100]}...")
        
    except Exception as e:
        print(f"❌ API endpoint error: {e}")

if __name__ == "__main__":
    test_basic_routes()
    test_api_endpoint_simple()