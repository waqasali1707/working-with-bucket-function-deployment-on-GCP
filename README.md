# Video Processing API

## Overview

This project is a FastAPI-based video processing service deployed on Google Cloud Functions Gen2. It processes videos to find specific phrases and splits them at the detected timestamp. The service uses Deepgram for transcription (directly from URL) and GPT-4 for intelligent phrase detection.

## Features

- **Asynchronous Processing**: Submit jobs and check status later
- **Direct URL Transcription**: Deepgram transcribes directly from signed URLs (no download needed)
- **Intelligent Phrase Detection**: Uses GPT-4 with fallback to simple search
- **Chunked Processing**: Optional transcript chunking for large videos
- **Automatic Video Splitting**: Splits videos into public/private parts at detected timestamp

## API Endpoints

### 1. Submit Video Processing Job
```
POST /process-video
```

Request body:
```json
{
  "bucket_name": "postscrypt",
  "video_path": "3/video1.mp4",
  "signed_url": "https://...",
  "target_phrase": "I quite enjoyed under the dome.",
  "num_chunks": 5,
  "job_id": "custom-job-id"  // Optional
}
```

Response:
```json
{
  "job_id": "uuid-here",
  "status": "accepted",
  "message": "Video processing job started"
}
```

### 2. Get Video Paths
```
GET /video-paths/{job_id}
```

Response (when completed):
```json
{
  "job_id": "uuid-here",
  "status": "completed",
  "public_video_path": "gs://postscrypt/3/public/video1.mp4",
  "private_video_path": "gs://postscrypt/3/private/video1.mp4",
  "processing_details": {
    "utterances_found": 150,
    "transcript_preview": "..."
  }
}
```

## Video Path Structure

For a video at `bucket/3/video1.mp4`, the split videos will be saved as:
- **Public part**: `bucket/3/public/video1.mp4` (before the target phrase)
- **Private part**: `bucket/3/private/video1.mp4` (after the target phrase)

## Setup

1. **Environment Variables**: Set the following environment variables:
   - `DEEPGRAM_API_KEY`: Your Deepgram API key
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `BUCKET_NAME`: Default GCS bucket name
   - `SERVICE_URL`: The Cloud Run service URL (for testing)
   - `SIGNED_URL`: A signed URL for testing

2. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Local Development**:
   ```bash
   uvicorn main:app --reload --port 8080
   ```

## Deployment

Deploy to Google Cloud Functions Gen2 using the provided script:

```bash
# Set environment variables first
export DEEPGRAM_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export BUCKET_NAME="your-bucket"

# Deploy
./deploy_script.sh
```

The deployment script:
- Uses Google Cloud Functions Gen2 (source-based deployment, no Docker required)
- Allocates 8GB memory and 2 CPUs
- Sets a 9-minute timeout
- Configures auto-scaling (0-100 instances)
- Automatically handles FFmpeg installation

## Testing

Run the test suite:

```bash
# Set the service URL
export SERVICE_URL="https://your-service-url.run.app"
export SIGNED_URL="your-signed-url"

# Run tests
python test_function.py
```

Test options:
1. Full video processing test (submit and wait)
2. Test with custom job ID
3. Test invalid job ID handling
4. Test API root endpoint
5. Run all tests

## Generating Signed URLs

To generate a signed URL for GCS:

```bash
gsutil signurl -d 1h service-account.json gs://bucket/path/to/video.mp4
```

## Architecture

The service follows this workflow:

1. **Job Submission**: Client submits video processing request
2. **Background Processing**:
   - Video downloaded from GCS for splitting
   - Audio transcribed directly from signed URL by Deepgram
   - GPT-4 finds the target phrase timestamp
   - Video split at detected timestamp
   - Parts uploaded to public/private folders
3. **Status Checking**: Client polls for job completion

## Notes

- The service uses in-memory job storage (jobs are lost on restart)
- For production, consider using a persistent storage solution
- Signed URLs must be valid for the duration of processing
- FFmpeg is automatically available in Cloud Functions Gen2 runtime
- No Docker configuration needed - deployment is source-based

## License

This project is licensed under the MIT License. 