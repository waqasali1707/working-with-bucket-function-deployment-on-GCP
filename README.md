# Project Name

## Overview

This project processes videos using a Cloud Function. It supports both direct GCS access and signed URLs.

## Setup

1. **Environment Variables**: Set the following environment variables to configure the application:
   - `CLOUD_FUNCTION_URL`: The URL of the Cloud Function.
   - `SIGNED_URL`: The signed URL for accessing video files.

2. **Installation**:
   - Install the required packages using `pip install -r requirements.txt`.

3. **Running Tests**:
   - Run the tests using `python test_function.py` and choose the desired test method.

## Deployment

- Use the `deploy_script.sh` to deploy the application.

## Viewing Cloud Function URL

- After deploying the Cloud Function using the `deploy_script.sh`, the function URL can be viewed in the terminal output. Look for the line that starts with `Function URL:`.

## Generating a Signed URL for a GCP Bucket

To generate a signed URL for a Google Cloud Storage bucket, follow these steps:

1. **Install the Google Cloud SDK**: Ensure you have the Google Cloud SDK installed and authenticated.

2. **Use the `gsutil` command**:
   - Run the following command to generate a signed URL:
     ```bash
     gsutil signurl -d [DURATION] -u [SERVICE_ACCOUNT_JSON] gs://[BUCKET_NAME]/[OBJECT_NAME]
     ```
   - Replace `[DURATION]` with the duration the signed URL should be valid (e.g., `1h` for one hour).
   - Replace `[SERVICE_ACCOUNT_JSON]` with the path to your service account JSON key file.
   - Replace `[BUCKET_NAME]` and `[OBJECT_NAME]` with your bucket and object names.

3. **Example**:
   ```bash
   gsutil signurl -d 1h -u my-service-account.json gs://my-bucket/my-object
   ```

This will output a signed URL that can be used to access the specified object in the bucket.

## Setting Environment Variables

Before running the `deploy_script.sh`, ensure that the necessary environment variables are set. This can be done in the terminal or command prompt:

### On Windows:

1. Open Command Prompt or PowerShell.
2. Set environment variables using the `set` command:
   ```cmd
   set DEEPGRAM_API_KEY=your-deepgram-api-key
   set OPENAI_API_KEY=your-openai-api-key
   set BUCKET_NAME=your-bucket-name
   ```

### On Unix-based Systems (Linux/Mac):

1. Open Terminal.
2. Set environment variables using the `export` command:
   ```bash
   export DEEPGRAM_API_KEY=your-deepgram-api-key
   export OPENAI_API_KEY=your-openai-api-key
   export BUCKET_NAME=your-bucket-name
   ```

## Prerequisites for Running `deploy_script.sh` on Windows

- **Git Bash**: Install Git for Windows, which includes Git Bash, allowing you to run `.sh` scripts.
- **Google Cloud SDK**: Ensure the Google Cloud SDK is installed and authenticated.

## Additional Notes

- Ensure you have the necessary permissions and roles in Google Cloud to deploy functions and access storage buckets.
- Verify that your Google Cloud project is set correctly using `gcloud config set project [PROJECT_ID]`.

## License

This project is licensed under the MIT License. 