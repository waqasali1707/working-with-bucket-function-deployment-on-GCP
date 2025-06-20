import requests
import json
import os
import time

# Replace with your actual Cloud Run URL after deployment
SERVICE_URL = os.getenv('SERVICE_URL', 'https://example.com/default-url')

def test_video_processing_api():
    """Test the video processing FastAPI endpoints"""
    
    # Step 1: Submit video processing job
    payload = {
        "bucket_name": "postscrypt",
        "video_path": "3/video1.mp4",
        "signed_url": os.getenv('SIGNED_URL', 'https://example.com/signed-url'),
        "target_phrase": "I quite enjoyed under the dome.",
        "num_chunks": 5  # Split transcription into 5 chunks for processing
    }
    
    print("=== Testing Video Processing API ===")
    print(f"Service URL: {SERVICE_URL}")
    print(f"Video path: {payload['video_path']}")
    print(f"Target phrase: {payload['target_phrase']}")
    print(f"Chunking: {payload['num_chunks']} chunks")
    print("\nStep 1: Submitting video processing job...")
    
    try:
        # Submit job
        response = requests.post(
            f"{SERVICE_URL}/process-video",
            json=payload,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print("❌ ERROR submitting job!")
            print(f"Response: {response.text}")
            return
        
        result = response.json()
        job_id = result.get('job_id')
        print(f"✅ Job submitted successfully!")
        print(f"Job ID: {job_id}")
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
        
        # Step 2: Poll for results
        print("\nStep 2: Checking job status...")
        max_attempts = 60  # Max 10 minutes
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            time.sleep(10)  # Wait 10 seconds between checks
            
            print(f"\nAttempt {attempt}/{max_attempts}: Checking status...")
            
            status_response = requests.get(
                f"{SERVICE_URL}/video-paths/{job_id}",
                timeout=30
            )
            
            if status_response.status_code != 200:
                print(f"❌ ERROR checking status: {status_response.status_code}")
                print(f"Response: {status_response.text}")
                continue
            
            status_data = status_response.json()
            status = status_data.get('status')
            
            print(f"Job status: {status}")
            
            if status == 'completed':
                print("\n✅ VIDEO PROCESSING COMPLETED!")
                print(f"Public video path: {status_data.get('public_video_path')}")
                print(f"Private video path: {status_data.get('private_video_path')}")
                
                if status_data.get('processing_details'):
                    print(f"Processing details: {json.dumps(status_data['processing_details'], indent=2)}")
                break
                
            elif status == 'failed':
                print("\n❌ VIDEO PROCESSING FAILED!")
                print(f"Error: {status_data.get('error')}")
                break
                
            elif status == 'processing':
                print("Still processing...")
            else:
                print(f"Unknown status: {status}")
        
        if attempt >= max_attempts:
            print("\n❌ TIMEOUT: Job did not complete within 10 minutes")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_with_custom_job_id():
    """Test with a custom job ID"""
    
    custom_job_id = f"test-job-{int(time.time())}"
    
    payload = {
        "bucket_name": "postscrypt",
        "video_path": "3/video1.mp4",
        "signed_url": os.getenv('SIGNED_URL', 'https://example.com/signed-url'),
        "target_phrase": "I quite enjoyed under the dome.",
        "num_chunks": 3,
        "job_id": custom_job_id
    }
    
    print(f"=== Testing with custom job ID: {custom_job_id} ===")
    print("Submitting job...")
    
    try:
        response = requests.post(
            f"{SERVICE_URL}/process-video",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Job submitted successfully!")
            print(f"Response: {response.json()}")
            
            # You can now check status using the custom job ID
            print(f"\nTo check status, use: GET {SERVICE_URL}/video-paths/{custom_job_id}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_invalid_job_id():
    """Test getting paths for non-existent job"""
    
    fake_job_id = "non-existent-job-id"
    
    print(f"=== Testing with invalid job ID: {fake_job_id} ===")
    
    try:
        response = requests.get(
            f"{SERVICE_URL}/video-paths/{fake_job_id}",
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 404:
            print("✅ Correctly returned 404 for non-existent job")
        else:
            print("❌ Unexpected response code")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_api_root():
    """Test the root endpoint"""
    
    print("=== Testing API root endpoint ===")
    
    try:
        response = requests.get(
            SERVICE_URL,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API is running!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Unexpected response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("FastAPI Video Processing Test Suite")
    print("=" * 50)
    
    print("\nChoose test:")
    print("1. Full video processing test (submit job and wait for completion)")
    print("2. Test with custom job ID")
    print("3. Test invalid job ID handling")
    print("4. Test API root endpoint")
    print("5. Run all tests")
    
    choice = input("\nEnter choice (1-5, default 1): ").strip()
    
    if choice == "2":
        test_with_custom_job_id()
    elif choice == "3":
        test_invalid_job_id()
    elif choice == "4":
        test_api_root()
    elif choice == "5":
        test_api_root()
        print("\n" + "=" * 50 + "\n")
        test_video_processing_api()
        print("\n" + "=" * 50 + "\n")
        test_with_custom_job_id()
        print("\n" + "=" * 50 + "\n")
        test_invalid_job_id()
    else:
        test_video_processing_api()