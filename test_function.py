import requests
import json
import os

# Replace with your actual Cloud Function URL after deployment
CLOUD_FUNCTION_URL = os.getenv('CLOUD_FUNCTION_URL', 'https://example.com/default-url')

def test_video_processing_with_gcs():
    """Test the video processing Cloud Function using direct GCS access"""
    
    payload = {
        # Use direct GCS access instead of signed URL (more efficient)
        "source_bucket": "postscrypt",
        "source_blob": "3/video1.mp4",
        "target_phrase": "I quite enjoyed under the dome.",
        "original_folder": "3",
        "output_bucket": "postscrypt",  # Use same bucket for output
        "num_chunks": 5  # NEW: Split transcription into 5 chunks for processing
    }
    
    print("Testing video processing function with GCS direct access...")
    print(f"Source: gs://{payload['source_bucket']}/{payload['source_blob']}")
    print(f"Target phrase: {payload['target_phrase']}")
    print(f"Chunking: {payload['num_chunks']} chunks")
    print("Sending request to Cloud Function...")
    
    try:
        response = requests.post(
            CLOUD_FUNCTION_URL,
            json=payload,
            timeout=600  # 10 minute timeout
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Timestamp found: {result.get('timestamp_found')}")
            print(f"Part 1 saved to: {result.get('part1_location')}")
            print(f"Part 2 saved to: {result.get('part2_location')}")
            print(f"Processing details: {result.get('processing_details', {})}")
        else:
            print("❌ ERROR!")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
                
                # Show transcript preview if available for debugging
                if 'transcript_preview' in error_data:
                    print(f"Transcript preview: {error_data['transcript_preview']}")
                if 'total_utterances' in error_data:
                    print(f"Total utterances found: {error_data['total_utterances']}")
                    
            except json.JSONDecodeError:
                print(f"Raw error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Video processing may still be running.")
        print("Check Cloud Function logs for more details.")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_video_processing_with_signed_url():
    """Test the video processing Cloud Function using signed URL (fallback method)"""
    
    # Your signed URL (keep this as fallback)
    SIGNED_URL = os.getenv('SIGNED_URL', 'https://example.com/default-signed-url')
    
    payload = {
        "signed_url": SIGNED_URL,
        "target_phrase": "I quite enjoyed under the dome.",
        "original_folder": "3",
        "output_bucket": "postscrypt",
        "num_chunks": 3  # NEW: Split transcription into 3 chunks for processing
    }
    
    print("Testing video processing function with signed URL...")
    print(f"Target phrase: {payload['target_phrase']}")
    print(f"Chunking: {payload['num_chunks']} chunks")
    print("Sending request to Cloud Function...")
    
    try:
        response = requests.post(
            CLOUD_FUNCTION_URL,
            json=payload,
            timeout=600  # 10 minute timeout
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Timestamp found: {result.get('timestamp_found')}")
            print(f"Part 1 saved to: {result.get('part1_location')}")
            print(f"Part 2 saved to: {result.get('part2_location')}")
        else:
            print("❌ ERROR!")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
                
                # Show transcript preview if available for debugging
                if 'transcript_preview' in error_data:
                    print(f"Transcript preview: {error_data['transcript_preview']}")
                if 'total_utterances' in error_data:
                    print(f"Total utterances found: {error_data['total_utterances']}")
                    
            except json.JSONDecodeError:
                print(f"Raw error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Video processing may still be running.")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_without_chunking():
    """Test the video processing without chunking (original behavior)"""
    
    payload = {
        "source_bucket": "postscrypt",
        "source_blob": "3/video1.mp4",
        "target_phrase": "I quite enjoyed under the dome.",
        "original_folder": "3",
        "output_bucket": "postscrypt"
        # No num_chunks parameter = no chunking (original behavior)
    }
    
    print("Testing video processing function WITHOUT chunking (original behavior)...")
    print(f"Source: gs://{payload['source_bucket']}/{payload['source_blob']}")
    print(f"Target phrase: {payload['target_phrase']}")
    print("Chunking: Disabled")
    print("Sending request to Cloud Function...")
    
    try:
        response = requests.post(
            CLOUD_FUNCTION_URL,
            json=payload,
            timeout=600  # 10 minute timeout
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Timestamp found: {result.get('timestamp_found')}")
            print(f"Part 1 saved to: {result.get('part1_location')}")
            print(f"Part 2 saved to: {result.get('part2_location')}")
            print(f"Processing details: {result.get('processing_details', {})}")
        else:
            print("❌ ERROR!")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
                
                # Show transcript preview if available for debugging
                if 'transcript_preview' in error_data:
                    print(f"Transcript preview: {error_data['transcript_preview']}")
                if 'total_utterances' in error_data:
                    print(f"Total utterances found: {error_data['total_utterances']}")
                    
            except json.JSONDecodeError:
                print(f"Raw error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Video processing may still be running.")
        print("Check Cloud Function logs for more details.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Choose test method:")
    print("1. Direct GCS access with chunking (5 chunks)")
    print("2. Signed URL with chunking (3 chunks)")
    print("3. Without chunking (original behavior)")
    
    choice = input("Enter choice (1, 2, or 3, default 1): ").strip()
    
    if choice == "2":
        test_video_processing_with_signed_url()
    elif choice == "3":
        test_without_chunking()
    else:
        test_video_processing_with_gcs()