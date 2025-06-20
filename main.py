import os
import tempfile
import subprocess
import requests
import json
from google.cloud import storage
import openai
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import time
import uuid
from datetime import datetime

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

# Initialize FastAPI app
app = FastAPI(title="Video Processing API", version="2.0.0")

# Store processing jobs status
processing_jobs = {}

# -----------------------------------------------------------------------------
# Pydantic models for request/response
# -----------------------------------------------------------------------------

class VideoProcessingRequest(BaseModel):
    bucket_name: str
    video_path: str  # e.g., "3/video1.mp4"
    signed_url: str  # For Deepgram direct transcription
    target_phrase: str = "I quite enjoyed under the dome."
    num_chunks: Optional[int] = None
    job_id: Optional[str] = None  # If not provided, will be generated

class VideoProcessingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class VideoPathsRequest(BaseModel):
    job_id: str

class VideoPathsResponse(BaseModel):
    job_id: str
    status: str
    public_video_path: Optional[str] = None
    private_video_path: Optional[str] = None
    error: Optional[str] = None
    processing_details: Optional[Dict[str, Any]] = None

# -----------------------------------------------------------------------------
# Helper functions (keeping most of the original functions)
# -----------------------------------------------------------------------------

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

def transcribe_with_deepgram_url(signed_url, model="nova-3"):
    """Transcribe audio directly from URL using Deepgram API"""
    logger.info(f"Transcribing audio directly from URL with Deepgram {model} model...")
    
    try:
        # Deepgram can transcribe directly from URL
        response = requests.post(
            f'https://api.deepgram.com/v1/listen?diarize=true&punctuate=true&utterances=true&model={model}',
            headers={
                'Authorization': f'Token {DEEPGRAM_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'url': signed_url
            },
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

def process_video_task(job_id: str, request: VideoProcessingRequest):
    """Background task to process video"""
    logger.info(f"=== STARTING VIDEO PROCESSING FOR JOB {job_id} ===")
    
    try:
        # Update job status
        processing_jobs[job_id] = {
            "status": "processing",
            "started_at": datetime.utcnow().isoformat(),
            "request": request.dict()
        }
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Define temporary file paths
            temp_video_path = os.path.join(temp_dir, 'input_video.mp4')
            temp_part1_path = os.path.join(temp_dir, 'part1.mp4')
            temp_part2_path = os.path.join(temp_dir, 'part2.mp4')
            
            # Step 1: Download video from GCS for splitting
            logger.info("=== STEP 1: DOWNLOADING VIDEO FOR SPLITTING ===")
            if not download_video_from_gcs(request.bucket_name, request.video_path, temp_video_path):
                raise Exception("Failed to download video from GCS")
            
            # Log video file size
            video_size = os.path.getsize(temp_video_path)
            logger.info(f"Video file size: {video_size / (1024*1024):.2f} MB")
            
            # Step 2: Transcribe using Deepgram with signed URL
            logger.info("=== STEP 2: TRANSCRIBING AUDIO WITH DEEPGRAM (USING URL) ===")
            transcription_result = transcribe_with_deepgram_url(request.signed_url)
            if not transcription_result:
                raise Exception("Failed to transcribe audio")
            
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
            
            # Step 3: Find timestamp with GPT-4
            logger.info("=== STEP 3: FINDING PHRASE TIMESTAMP ===")
            timestamp = find_phrase_timestamp_with_gpt4(transcription_result, request.target_phrase, request.num_chunks)
            if not timestamp:
                raise Exception(f"Could not find phrase: {request.target_phrase}")
            
            logger.info(f"Found timestamp: {timestamp}")
            
            # Step 4: Split video
            logger.info("=== STEP 4: SPLITTING VIDEO ===")
            if not split_video_at_timestamp(temp_video_path, timestamp, temp_part1_path, temp_part2_path):
                raise Exception("Failed to split video")
            
            # Step 5: Upload both parts to designated folders
            logger.info("=== STEP 5: UPLOADING SPLIT VIDEOS ===")
            
            # Extract folder and filename from video path
            # e.g., "3/video1.mp4" -> folder="3", filename="video1.mp4"
            path_parts = request.video_path.split('/')
            if len(path_parts) >= 2:
                folder = path_parts[0]
                filename = path_parts[-1]
            else:
                folder = ""
                filename = request.video_path
            
            # Create paths according to new structure
            public_destination = f"{folder}/public/{filename}" if folder else f"public/{filename}"
            private_destination = f"{folder}/private/{filename}" if folder else f"private/{filename}"
            
            logger.info(f"Uploading public part to: {public_destination}")
            upload_success_1 = upload_to_bucket(temp_part1_path, request.bucket_name, public_destination)
            
            logger.info(f"Uploading private part to: {private_destination}")
            upload_success_2 = upload_to_bucket(temp_part2_path, request.bucket_name, private_destination)
            
            if not (upload_success_1 and upload_success_2):
                raise Exception("Failed to upload split videos")
            
            logger.info(f"=== VIDEO PROCESSING COMPLETED SUCCESSFULLY FOR JOB {job_id} ===")
            
            # Update job status with success
            processing_jobs[job_id] = {
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "public_video_path": f"gs://{request.bucket_name}/{public_destination}",
                "private_video_path": f"gs://{request.bucket_name}/{private_destination}",
                "timestamp_found": timestamp,
                "processing_details": {
                    "utterances_found": len(utterances),
                    "transcript_preview": " | ".join([u['transcript'] for u in utterances[:3]]) if utterances else "No utterances"
                }
            }
    
    except Exception as e:
        logger.error(f"Error processing video for job {job_id}: {e}", exc_info=True)
        processing_jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        }

# -----------------------------------------------------------------------------
# FastAPI endpoints
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Processing API",
        "version": "2.0.0",
        "endpoints": {
            "POST /process-video": "Start video processing job",
            "GET /video-paths/{job_id}": "Get processed video paths"
        }
    }

@app.post("/process-video", response_model=VideoProcessingResponse)
async def process_video(request: VideoProcessingRequest, background_tasks: BackgroundTasks):
    """Start video processing job"""
    logger.info("=== RECEIVED VIDEO PROCESSING REQUEST ===")
    
    # Generate job ID if not provided
    job_id = request.job_id or str(uuid.uuid4())
    
    # Validate request
    if not request.bucket_name or not request.video_path or not request.signed_url:
        raise HTTPException(status_code=400, detail="Missing required fields: bucket_name, video_path, or signed_url")
    
    # Check if job already exists
    if job_id in processing_jobs:
        raise HTTPException(status_code=409, detail=f"Job {job_id} already exists")
    
    # Start processing in background
    background_tasks.add_task(process_video_task, job_id, request)
    
    # Return immediate response
    return VideoProcessingResponse(
        job_id=job_id,
        status="accepted",
        message="Video processing job started"
    )

@app.get("/video-paths/{job_id}", response_model=VideoPathsResponse)
async def get_video_paths(job_id: str):
    """Get processed video paths by job ID"""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_info = processing_jobs[job_id]
    
    if job_info["status"] == "processing":
        return VideoPathsResponse(
            job_id=job_id,
            status="processing",
            error=None,
            public_video_path=None,
            private_video_path=None
        )
    elif job_info["status"] == "completed":
        return VideoPathsResponse(
            job_id=job_id,
            status="completed",
            public_video_path=job_info["public_video_path"],
            private_video_path=job_info["private_video_path"],
            processing_details=job_info.get("processing_details")
        )
    else:  # failed
        return VideoPathsResponse(
            job_id=job_id,
            status="failed",
            error=job_info.get("error", "Unknown error"),
            public_video_path=None,
            private_video_path=None
        )

# For Google Cloud Functions Gen2 compatibility
import functions_framework
from flask import Request, Response
import asyncio
from typing import Any

# Create a wrapper function that Cloud Functions can call
@functions_framework.http
def main(request: Request) -> Response:
    """HTTP Cloud Function entry point for FastAPI app."""
    import nest_asyncio
    nest_asyncio.apply()
    
    # Get the request data
    path = request.path
    method = request.method
    headers = dict(request.headers)
    body = request.get_data(as_text=True)
    query_params = dict(request.args)
    
    # Create a mock ASGI scope
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "query_string": request.query_string or b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "server": ("localhost", 8080),
        "scheme": "https",
        "root_path": "",
    }
    
    # Storage for response
    response_data = {"status": 200, "headers": [], "body": b""}
    
    async def receive():
        return {
            "type": "http.request",
            "body": body.encode() if body else b"",
        }
    
    async def send(message):
        if message["type"] == "http.response.start":
            response_data["status"] = message["status"]
            response_data["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            body_part = message.get("body", b"")
            if body_part:
                response_data["body"] += body_part
    
    # Run the FastAPI app
    async def run_app():
        await app(scope, receive, send)
    
    # Execute the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_app())
    finally:
        loop.close()
    
    # Build Flask response
    response = Response(response_data["body"])
    response.status_code = response_data["status"]
    
    # Set headers
    for header_name, header_value in response_data["headers"]:
        if header_name.lower() != b"content-length":  # Flask sets this automatically
            response.headers[header_name.decode()] = header_value.decode()
    
    return response

# Also keep the original for local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)