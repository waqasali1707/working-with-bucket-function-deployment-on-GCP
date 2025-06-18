import os
import tempfile
import subprocess
import requests
import json
from google.cloud import storage
import openai
from flask import jsonify
import logging
import time

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Environment variables (set these in Cloud Function)
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'postscrypt')

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

def download_video_from_url(signed_url, temp_video_path):
    """Download video from signed URL to temporary file"""
    logger.info(f"Downloading video from signed URL...")
    logger.debug(f"Signed URL: {signed_url[:100]}...")  # Log first 100 chars for debugging
    
    try:
        response = requests.get(signed_url, stream=True, timeout=60)
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"HTTP error {response.status_code}: {response.text[:500]}")
            return False
        
        response.raise_for_status()
        
        content_length = response.headers.get('content-length')
        if content_length:
            logger.info(f"Expected file size: {int(content_length) / (1024*1024):.2f} MB")
        
        downloaded_bytes = 0
        with open(temp_video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
        
        logger.info(f"Video downloaded successfully to {temp_video_path}")
        logger.info(f"Downloaded {downloaded_bytes / (1024*1024):.2f} MB")
        
        # Verify file was created and has content
        if not os.path.exists(temp_video_path):
            logger.error("Downloaded file does not exist")
            return False
        
        file_size = os.path.getsize(temp_video_path)
        if file_size == 0:
            logger.error("Downloaded file is empty")
            return False
        
        logger.info(f"File verification successful: {file_size / (1024*1024):.2f} MB")
        return True
        
    except requests.exceptions.Timeout:
        logger.error("Download timed out after 60 seconds")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error downloading video: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading video: {e}")
        return False

def extract_audio_cloud(video_path, output_audio_path):
    """Extract audio from video using FFmpeg in cloud environment"""
    command = [
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_audio_path
    ]
    
    try:
        logger.info("Extracting audio from video...")
        # Log video duration first
        probe_command = ["ffmpeg", "-i", video_path, "-hide_banner"]
        probe_result = subprocess.run(probe_command, capture_output=True, text=True)
        logger.info(f"Video info: {probe_result.stderr[:500]}")  # Log first 500 chars of video info
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("Audio extraction successful")
        
        # Check audio file size
        audio_size = os.path.getsize(output_audio_path)
        logger.info(f"Extracted audio size: {audio_size / (1024*1024):.2f} MB")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        return False

def extract_audio_partial(video_path, output_audio_path, start_time=None, duration=None):
    """Extract audio from a specific portion of video"""
    command = ["ffmpeg", "-y"]
    
    # Add start time if specified
    if start_time:
        command.extend(["-ss", str(start_time)])
    
    command.extend(["-i", video_path])
    
    # Add duration if specified
    if duration:
        command.extend(["-t", str(duration)])
    
    command.extend([
        "-vn",
        "-acodec", "pcm_s16le", 
        "-ar", "16000", 
        "-ac", "1",
        output_audio_path
    ])
    
    try:
        logger.info(f"Extracting audio (start: {start_time}, duration: {duration})...")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("Audio extraction successful")
        
        # Check audio file size
        audio_size = os.path.getsize(output_audio_path)
        logger.info(f"Extracted audio size: {audio_size / (1024*1024):.2f} MB")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        return False

def transcribe_with_deepgram_cloud(audio_path, model="nova-2"):
    """Transcribe audio using Deepgram API"""
    if not os.path.isfile(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None
    
    # Log audio file size
    audio_size = os.path.getsize(audio_path)
    logger.info(f"Audio file size for transcription: {audio_size / (1024*1024):.2f} MB")
    
    # Check if audio file is very large
    if audio_size > 100 * 1024 * 1024:  # 100MB
        logger.warning(f"Audio file is very large ({audio_size / (1024*1024):.2f} MB), this may cause issues")
    
    # Use nova-3 model like in local version
    model = "nova-3"
    logger.info(f"Transcribing audio with Deepgram {model} model...")
    
    # For Cloud Functions, we might need to use streaming for large files
    if audio_size > 10 * 1024 * 1024:  # If larger than 10MB
        logger.info("Using chunked upload for large audio file")
        return transcribe_large_audio_deepgram(audio_path, model)
    
    try:
        with open(audio_path, 'rb') as audio_file:
            response = requests.post(
                f'https://api.deepgram.com/v1/listen?diarize=true&punctuate=true&utterances=true&model={model}',
                headers={
                    'Authorization': f'Token {DEEPGRAM_API_KEY}',
                    'Content-Type': 'audio/wav'
                },
                data=audio_file,
                timeout=1200  # 20 minute timeout
            )
        
        logger.info(f"Deepgram response status: {response.status_code}")
        
        if response.status_code == 200:
            logger.info("Transcription completed successfully")
            transcription_json = response.json()
            
            # Log transcription statistics
            utterances = transcription_json.get("results", {}).get("utterances", [])
            if utterances:
                total_duration = utterances[-1]["end"] if utterances else 0
                logger.info(f"Transcription stats: {len(utterances)} utterances, {total_duration:.2f} seconds duration")
                
                # Check if we might be hitting a duration limit
                if total_duration < 300:  # Less than 5 minutes
                    logger.warning(f"Transcription seems incomplete - only {total_duration:.2f} seconds transcribed")
                    logger.warning("This might indicate a processing limit was hit")
            
            return transcription_json
        else:
            logger.error(f"Deepgram API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error("Deepgram request timed out")
        return None
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None

def transcribe_large_audio_deepgram(audio_path, model="nova-3"):
    """Transcribe large audio files using streaming"""
    logger.info("Using streaming approach for large audio file")
    
    try:
        # Get file size
        file_size = os.path.getsize(audio_path)
        
        with open(audio_path, 'rb') as audio_file:
            # Use streaming to avoid loading entire file into memory
            headers = {
                'Authorization': f'Token {DEEPGRAM_API_KEY}',
                'Content-Type': 'audio/wav',
            }
            
            # Create a generator to stream the file
            def file_generator(file_obj, chunk_size=1024*1024):  # 1MB chunks
                while True:
                    chunk = file_obj.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            
            response = requests.post(
                f'https://api.deepgram.com/v1/listen?diarize=true&punctuate=true&utterances=true&model={model}',
                headers=headers,
                data=file_generator(audio_file),
                timeout=1800  # 30 minute timeout
            )
        
        if response.status_code == 200:
            logger.info("Large file transcription completed successfully")
            return response.json()
        else:
            logger.error(f"Deepgram API error for large file: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error during large file transcription: {e}")
        return None

def split_transcription_into_chunks(utterances, num_chunks):
    """Split utterances into specified number of chunks"""
    if not utterances or num_chunks <= 1:
        return [utterances]
    
    total_utterances = len(utterances)
    chunk_size = total_utterances // num_chunks
    remainder = total_utterances % num_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(num_chunks):
        # Add one extra utterance to some chunks to handle remainder
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        if start_idx < total_utterances:
            chunk = utterances[start_idx:end_idx]
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
        
        start_idx = end_idx
    
    return chunks

def find_phrase_in_chunk_simple_search(chunk_utterances, target_phrase):
    """Simple search for phrase in a chunk of utterances"""
    target_lower = target_phrase.lower()
    
    # Try exact phrase first
    for i, u in enumerate(chunk_utterances):
        if target_lower in u['transcript'].lower():
            logger.info(f"FOUND PHRASE in chunk utterance {i}: [{u['start']:.2f}s-{u['end']:.2f}s] {u['transcript']}")
            return u['end']
    
    # Try variations
    variations = [
        "I quite enjoyed Under the Dome",
        "quite enjoyed Under the Dome", 
        "enjoyed Under the Dome",
        "Under the Dome"
    ]
    
    for var in variations:
        var_lower = var.lower()
        for i, u in enumerate(chunk_utterances):
            if var_lower in u['transcript'].lower():
                logger.info(f"FOUND VARIATION '{var}' in chunk utterance {i}: [{u['start']:.2f}s-{u['end']:.2f}s] {u['transcript']}")
                return u['end']
    
    return None

def find_phrase_in_chunk_with_gpt4(chunk_utterances, target_phrase, chunk_number):
    """Use GPT-4 to find phrase in a specific chunk"""
    if not chunk_utterances:
        return None
        
    logger.info(f"Analyzing chunk {chunk_number} with GPT-4 (contains {len(chunk_utterances)} utterances)")
    
    # Format chunk transcript for GPT-4
    formatted_transcript = "\n".join([
        f'[{round(u["start"], 2)}s - {round(u["end"], 2)}s] Speaker {u["speaker"]}: {u["transcript"]}'
        for u in chunk_utterances
    ])
    
    chunk_duration = chunk_utterances[-1]["end"] - chunk_utterances[0]["start"] if chunk_utterances else 0
    logger.info(f"Chunk {chunk_number} duration: {chunk_duration:.2f} seconds")
    
    prompt = f"""From the following diarized transcript chunk, find the sentence that contains or is most similar to: "{target_phrase}".

Important: The phrase might appear with slight variations such as:
- "I quite enjoyed Under the Dome" (different capitalization)
- "I quite enjoyed 'Under the Dome'" (with quotes)
- "I quite enjoyed Under the Dome." (with period)
- "i quite enjoyed under the dome" (all lowercase)
- As part of a longer sentence

**CRITICAL**: Perform a CASE-INSENSITIVE search. The capitalization does not matter.
For example, "under the dome" should match "Under the Dome" or "UNDER THE DOME".

Look for the sentence that best matches this phrase semantically, even if the exact wording differs slightly.

Respond with ONLY the numerical end duration of that sentence (just the number).
Do not include any other words, explanations, or punctuation. Your entire response should be just the timestamp number.

If you cannot find anything similar, respond with "NOT_FOUND".

Transcript Chunk {chunk_number}:
---
{formatted_transcript}
---

The end duration of the sentence containing "{target_phrase}" is:"""
    
    try:
        logger.debug(f"Sending chunk {chunk_number} to GPT-4 (model gpt-4o)...")
        response = openai.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that follows instructions precisely. You extract specific data from text without adding any conversational fluff."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-4o",
            temperature=0.1,
            max_tokens=10
        )
        
        timestamp = response.choices[0].message.content.strip()
        logger.debug(f"GPT-4 response for chunk {chunk_number}: {timestamp}")
        
        if timestamp == "NOT_FOUND":
            logger.info(f"GPT-4 could not find the phrase in chunk {chunk_number}")
            return None
        
        # Basic validation: ensure the response looks like a number
        if not timestamp or not any(ch.isdigit() for ch in timestamp):
            logger.warning(f"GPT-4 response for chunk {chunk_number} did not contain a valid timestamp: {timestamp}")
            return None

        logger.info(f"GPT-4 found timestamp in chunk {chunk_number}: {timestamp}")
        return timestamp
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API for chunk {chunk_number}: {e}")
        return None

def find_phrase_timestamp_with_gpt4(transcription_result, target_phrase, num_chunks=None):
    """Use GPT-4 to find the end timestamp of target phrase with optional chunking"""
    logger.info(f"Analyzing transcript to find phrase: '{target_phrase}'")
    
    utterances = transcription_result.get("results", {}).get("utterances", [])
    if not utterances:
        logger.warning("No utterances found in transcript")
        return None
    
    # Log some statistics about the transcript
    total_duration = utterances[-1]["end"] if utterances else 0
    logger.info(f"Transcript contains {len(utterances)} utterances covering {total_duration:.2f} seconds")
    
    # If no chunking requested or only 1 chunk, process normally
    if not num_chunks or num_chunks <= 1:
        logger.info("Processing entire transcript without chunking")
        return find_phrase_timestamp_original(utterances, target_phrase)
    
    # Process with chunking
    logger.info(f"Processing transcript with chunking: {num_chunks} chunks")
    chunks = split_transcription_into_chunks(utterances, num_chunks)
    logger.info(f"Split transcript into {len(chunks)} actual chunks")
    
    for chunk_idx, chunk_utterances in enumerate(chunks, 1):
        logger.info(f"=== PROCESSING CHUNK {chunk_idx}/{len(chunks)} ===")
        
        if not chunk_utterances:
            logger.warning(f"Chunk {chunk_idx} is empty, skipping")
            continue
            
        chunk_start = chunk_utterances[0]["start"]
        chunk_end = chunk_utterances[-1]["end"]
        logger.info(f"Chunk {chunk_idx} timespan: {chunk_start:.2f}s - {chunk_end:.2f}s ({len(chunk_utterances)} utterances)")
        
        # First try simple search on this chunk
        logger.info(f"Trying simple search on chunk {chunk_idx}")
        simple_result = find_phrase_in_chunk_simple_search(chunk_utterances, target_phrase)
        
        if simple_result:
            logger.info(f"✅ PHRASE FOUND in chunk {chunk_idx} via simple search at timestamp: {simple_result}")
            return str(simple_result)
        
        # If simple search fails, try GPT-4 on this chunk
        logger.info(f"Simple search failed for chunk {chunk_idx}, trying GPT-4")
        gpt4_result = find_phrase_in_chunk_with_gpt4(chunk_utterances, target_phrase, chunk_idx)
        
        if gpt4_result:
            logger.info(f"✅ PHRASE FOUND in chunk {chunk_idx} via GPT-4 at timestamp: {gpt4_result}")
            return gpt4_result
        
        logger.info(f"❌ Phrase not found in chunk {chunk_idx}, moving to next chunk")
    
    logger.error("Phrase NOT FOUND in any chunk!")
    return None

def find_phrase_timestamp_original(utterances, target_phrase):
    """Original implementation for processing entire transcript at once"""
    # Format transcript for GPT-4
    formatted_transcript = "\n".join([
        f'[{round(u["start"], 2)}s - {round(u["end"], 2)}s] Speaker {u["speaker"]}: {u["transcript"]}'
        for u in utterances
    ])
    
    # Log the last few utterances to see if we're getting the full transcript
    logger.info("Last 3 utterances in transcript:")
    for u in utterances[-3:]:
        logger.info(f"[{u['start']:.2f}s-{u['end']:.2f}s] {u['transcript']}")
    
    # Add debug: search for the phrase in utterances directly
    logger.info("=== DEBUG: Searching for phrase in utterances ===")
    found_in_utterances = False
    found_timestamp = None
    for i, u in enumerate(utterances):
        if target_phrase.lower() in u['transcript'].lower():
            logger.info(f"FOUND PHRASE in utterance {i}: [{u['start']:.2f}s-{u['end']:.2f}s] {u['transcript']}")
            found_in_utterances = True
            found_timestamp = u['end']  # Store the end timestamp
            break
    
    if not found_in_utterances:
        # Try searching for variations
        variations = [
            "I quite enjoyed Under the Dome",
            "quite enjoyed Under the Dome",
            "enjoyed Under the Dome",
            "Under the Dome"
        ]
        for var in variations:
            for i, u in enumerate(utterances):
                if var.lower() in u['transcript'].lower():
                    logger.info(f"FOUND VARIATION '{var}' in utterance {i}: [{u['start']:.2f}s-{u['end']:.2f}s] {u['transcript']}")
                    found_in_utterances = True
                    found_timestamp = u['end']  # Store the end timestamp
                    break
            if found_in_utterances:
                break
    
    if not found_in_utterances:
        logger.error("Phrase NOT FOUND in any utterance!")
    
    if not formatted_transcript.strip():
        logger.warning("Formatted transcript is empty")
        return None
    
    # Log the length of the formatted transcript being sent to GPT-4
    logger.info(f"Formatted transcript length: {len(formatted_transcript)} characters")
    
    prompt = f"""From the following diarized transcript, find the sentence that contains or is most similar to: "{target_phrase}".

Important: The phrase might appear with slight variations such as:
- "I quite enjoyed Under the Dome" (different capitalization)
- "I quite enjoyed 'Under the Dome'" (with quotes)
- "I quite enjoyed Under the Dome." (with period)
- "i quite enjoyed under the dome" (all lowercase)
- As part of a longer sentence

**CRITICAL**: Perform a CASE-INSENSITIVE search. The capitalization does not matter.
For example, "under the dome" should match "Under the Dome" or "UNDER THE DOME".

Look for the sentence that best matches this phrase semantically, even if the exact wording differs slightly.

Respond with ONLY the numerical end duration of that sentence (just the number).
Do not include any other words, explanations, or punctuation. Your entire response should be just the timestamp number.

If you cannot find anything similar, respond with "NOT_FOUND".

Transcript:
---
{formatted_transcript}
---

The end duration of the sentence containing "{target_phrase}" is:"""
    
    try:
        logger.debug("Sending prompt to GPT-4 (model gpt-4o)...")
        response = openai.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that follows instructions precisely. You extract specific data from text without adding any conversational fluff."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-4o",
            temperature=0.1,
            max_tokens=10
        )
        
        timestamp = response.choices[0].message.content.strip()
        logger.debug("GPT-4 raw response: %s", timestamp)
        
        if timestamp == "NOT_FOUND":
            logger.warning("GPT-4 could not find the phrase in the transcript")
            # Use fallback if we found it in our debug search
            if found_in_utterances and found_timestamp:
                logger.info(f"Using fallback timestamp from debug search: {found_timestamp}")
                return str(found_timestamp)
            return None
        
        # Basic validation: ensure the response looks like a number
        if not timestamp or not any(ch.isdigit() for ch in timestamp):
            logger.warning(
                "GPT-4 response did not contain a valid timestamp. Response: %s",
                timestamp,
            )
            # Use fallback if we found it in our debug search
            if found_in_utterances and found_timestamp:
                logger.info(f"Using fallback timestamp from debug search: {found_timestamp}")
                return str(found_timestamp)
            return None

        logger.info(f"GPT-4 found timestamp: {timestamp}")
        return timestamp
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        # Use fallback if we found it in our debug search
        if found_in_utterances and found_timestamp:
            logger.info(f"Using fallback timestamp from debug search after API error: {found_timestamp}")
            return str(found_timestamp)
        return None

def split_video_at_timestamp(video_path, timestamp_str, part1_path, part2_path):
    """Split video into two parts at specified timestamp"""
    try:
        # Clean timestamp (remove 's' if present)
        timestamp_seconds = float(timestamp_str.strip().replace('s', ''))
        logger.info(f"Splitting video at {timestamp_seconds} seconds")
        
    except (ValueError, AttributeError) as e:
        logger.error(f"Invalid timestamp format: {timestamp_str}")
        return False
    
    # Command for first part (0 to timestamp)
    command1 = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-t", str(timestamp_seconds),
        "-c", "copy",
        part1_path
    ]
    
    # Command for second part (timestamp to end)
    command2 = [
        "ffmpeg", "-y",
        "-ss", str(timestamp_seconds),
        "-i", video_path,
        "-c", "copy",
        part2_path
    ]
    
    try:
        # Create first part
        logger.info("Creating first part of video...")
        subprocess.run(command1, check=True, capture_output=True, text=True)
        logger.info(f"First part created: {part1_path}")
        
        # Create second part
        logger.info("Creating second part of video...")
        subprocess.run(command2, check=True, capture_output=True, text=True)
        logger.info(f"Second part created: {part2_path}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error splitting video: {e.stderr}")
        return False

def upload_to_bucket(local_file_path, bucket_name, destination_blob_name):
    """Upload file to Google Cloud Storage bucket"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        logger.info(f"Uploading {local_file_path} to {destination_blob_name}")
        blob.upload_from_filename(local_file_path)
        logger.info(f"Upload successful: {destination_blob_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading to bucket: {e}")
        return False

def download_video_from_gcs(bucket_name, source_blob_name, temp_video_path):
    """Download video directly from GCS bucket to temporary file"""
    logger.info(f"Downloading video from GCS: {bucket_name}/{source_blob_name}")
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        blob.download_to_filename(temp_video_path)
        logger.info(f"Video downloaded successfully to {temp_video_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading video from GCS: {e}")
        return False

def process_video_cloud(request):
    """Main Cloud Function entry point"""
    logger.info("=== DEPLOYMENT VERSION: 2024-01-17 with debug logging and fallback ===")  # Updated version marker
    
    try:
        # Parse request
        request_json = request.get_json(silent=True)
        if not request_json:
            logger.error("No JSON payload provided")
            return jsonify({'error': 'No JSON payload provided'}), 400
        
        # Accept either signed_url OR bucket/blob parameters
        signed_url = request_json.get('signed_url')
        source_bucket = request_json.get('source_bucket', BUCKET_NAME)
        source_blob = request_json.get('source_blob')
        
        target_phrase = request_json.get('target_phrase', 'I quite enjoyed under the dome.')
        original_folder = request_json.get('original_folder', '3')
        num_chunks = request_json.get('num_chunks', None)  # Optional chunking parameter
        
        # Use the same bucket for output, just different paths within it
        output_bucket = request_json.get('output_bucket', BUCKET_NAME)
        
        logger.info(f"=== STARTING VIDEO PROCESSING ===")
        logger.info(f"Target phrase: '{target_phrase}'")
        logger.info(f"Source bucket: {source_bucket}")
        logger.info(f"Source blob: {source_blob}")
        logger.info(f"Output bucket: {output_bucket}")
        logger.info(f"Chunking: {'Enabled (' + str(num_chunks) + ' chunks)' if num_chunks else 'Disabled'}")
        
        if not signed_url and not source_blob:
            logger.error("Either 'signed_url' or 'source_blob' is required")
            return jsonify({'error': 'Either signed_url or source_blob is required'}), 400
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Define temporary file paths
            temp_video_path = os.path.join(temp_dir, 'input_video.mp4')
            temp_audio_path = os.path.join(temp_dir, 'extracted_audio.wav')
            temp_part1_path = os.path.join(temp_dir, 'part1.mp4')
            temp_part2_path = os.path.join(temp_dir, 'part2.mp4')
            
            # Step 1: Download video
            logger.info("=== STEP 1: DOWNLOADING VIDEO ===")
            if source_blob:
                # Use direct GCS download (more efficient)
                if not download_video_from_gcs(source_bucket, source_blob, temp_video_path):
                    logger.error("Failed to download video from GCS")
                    return jsonify({'error': 'Failed to download video from GCS'}), 500
            else:
                # Use signed URL download (fallback)
                if not download_video_from_url(signed_url, temp_video_path):
                    logger.error("Failed to download video from signed URL")
                    return jsonify({'error': 'Failed to download video'}), 500
            
            # Log video file size
            video_size = os.path.getsize(temp_video_path)
            logger.info(f"Video file size: {video_size / (1024*1024):.2f} MB")
            
            # Step 2: Extract audio
            logger.info("=== STEP 2: EXTRACTING AUDIO ===")
            if not extract_audio_cloud(temp_video_path, temp_audio_path):
                logger.error("Failed to extract audio")
                return jsonify({'error': 'Failed to extract audio'}), 500
            
            # Step 3: Transcribe audio
            logger.info("=== STEP 3: TRANSCRIBING AUDIO ===")
            transcription_result = transcribe_with_deepgram_cloud(temp_audio_path)
            if not transcription_result:
                logger.error("Failed to transcribe audio")
                return jsonify({'error': 'Failed to transcribe audio'}), 500
            
            # Log complete transcript for debugging
            utterances = transcription_result.get("results", {}).get("utterances", [])
            if utterances:
                logger.info(f"Transcription successful: {len(utterances)} utterances found")
                logger.info("=== COMPLETE TRANSCRIPT START ===")
                for i, utterance in enumerate(utterances):
                    logger.info(f"[{utterance['start']:.2f}s-{utterance['end']:.2f}s] Speaker {utterance['speaker']}: {utterance['transcript']}")
                logger.info("=== COMPLETE TRANSCRIPT END ===")
            else:
                logger.warning("No utterances found in transcription result")
            
            # Step 4: Find timestamp with GPT-4
            logger.info("=== STEP 4: FINDING PHRASE TIMESTAMP ===")
            timestamp = find_phrase_timestamp_with_gpt4(transcription_result, target_phrase, num_chunks)
            if not timestamp:
                logger.error(f"I Could not find phrase: {target_phrase}")
                # Return detailed error with transcript preview for debugging
                transcript_preview = ""
                full_transcript = ""
                if utterances:
                    # Show first 10 utterances instead of 5
                    transcript_preview = " | ".join([u['transcript'] for u in utterances[:10]])
                    # Also create a full transcript for logging
                    full_transcript = " ".join([u['transcript'] for u in utterances])
                    # Log the full transcript to help debug
                    logger.error(f"Full transcript (first 1000 chars): {full_transcript[:1000]}")
                
                return jsonify({
                    'error': f'I Could not find phrase: {target_phrase}',
                    'transcript_preview': transcript_preview,
                    'total_utterances': len(utterances),
                    'transcript_snippet': full_transcript[:500]  # Add more context
                }), 500
            
            logger.info(f"Found timestamp: {timestamp}")
            
            # Step 5: Split video
            logger.info("=== STEP 5: SPLITTING VIDEO ===")
            if not split_video_at_timestamp(temp_video_path, timestamp, temp_part1_path, temp_part2_path):
                logger.error("Failed to split video")
                return jsonify({'error': 'Failed to split video'}), 500
            
            # Step 6: Upload both parts to the SAME bucket in different folders
            logger.info("=== STEP 6: UPLOADING SPLIT VIDEOS ===")
            
            # Generate unique filenames with timestamp
            timestamp_suffix = str(int(time.time()))
            
            # Use the original folder structure but within the same bucket
            part1_destination = f"{original_folder}/split_results/part1_{timestamp_suffix}.mp4"
            part2_destination = f"{original_folder}/split_results/part2_{timestamp_suffix}.mp4"
            
            logger.info(f"Uploading part 1 to: {part1_destination}")
            upload_success_1 = upload_to_bucket(temp_part1_path, output_bucket, part1_destination)
            
            logger.info(f"Uploading part 2 to: {part2_destination}")
            upload_success_2 = upload_to_bucket(temp_part2_path, output_bucket, part2_destination)
            
            if not (upload_success_1 and upload_success_2):
                logger.error("Failed to upload split videos")
                return jsonify({'error': 'Failed to upload split videos'}), 500
            
            logger.info("=== VIDEO PROCESSING COMPLETED SUCCESSFULLY ===")
            
            # Return success response
            return jsonify({
                'success': True,
                'message': 'Video processed successfully',
                'timestamp_found': timestamp,
                'target_phrase': target_phrase,
                'part1_location': f"gs://{output_bucket}/{part1_destination}",
                'part2_location': f"gs://{output_bucket}/{part2_destination}",
                'bucket': output_bucket,
                'processing_details': {
                    'utterances_found': len(utterances),
                    'transcript_preview': " | ".join([u['transcript'] for u in utterances[:3]]) if utterances else "No utterances"
                }
            }), 200
    
    except Exception as e:
        logger.error(f"Unexpected error in process_video_cloud: {e}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500