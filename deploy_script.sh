#!/bin/bash

# Cloud Function deployment script
# Make sure you have gcloud CLI installed and authenticated

# Function name and configuration
FUNCTION_NAME="process-video"
REGION="europe-west2"

# API Keys - IMPORTANT: Replace these with your actual keys
export DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY:-'your-deepgram-api-key'}
export OPENAI_API_KEY=${OPENAI_API_KEY:-'your-openai-api-key'}
export BUCKET_NAME=${BUCKET_NAME:-'your-bucket-name'}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}📁 Current directory contents:${NC}"
ls -la *.py 2>/dev/null || echo "No Python files found"

echo ""
echo -e "${YELLOW}🔍 Checking for main.py...${NC}"
if [ -f "main.py" ]; then
    echo -e "${GREEN}✅ main.py found${NC}"
    if grep -q "def process_video_cloud" main.py; then
        echo -e "${GREEN}✅ process_video_cloud function found in main.py${NC}"
    else
        echo -e "${RED}❌ process_video_cloud function NOT found in main.py!${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ main.py NOT found!${NC}"
    echo -e "${YELLOW}Looking for other Python files...${NC}"
    
    # Check if there's another Python file that should be renamed
    PYTHON_FILES=(*.py)
    if [ ${#PYTHON_FILES[@]} -eq 1 ] && [ -f "${PYTHON_FILES[0]}" ]; then
        echo -e "${YELLOW}Found ${PYTHON_FILES[0]} - checking if it contains the function...${NC}"
        if grep -q "def process_video_cloud" "${PYTHON_FILES[0]}"; then
            echo -e "${GREEN}✅ Found process_video_cloud in ${PYTHON_FILES[0]}${NC}"
            echo -e "${YELLOW}Copying ${PYTHON_FILES[0]} to main.py...${NC}"
            cp "${PYTHON_FILES[0]}" main.py
        fi
    fi
    
    # Check again if main.py exists now
    if [ ! -f "main.py" ]; then
        echo -e "${RED}❌ Could not create main.py!${NC}"
        exit 1
    fi
fi

# Check requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt NOT found!${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}🧹 Cleaning up previous deployment...${NC}"

# Delete the existing function (if it exists)
echo "Deleting existing function (if any)..."
gcloud functions delete $FUNCTION_NAME --region=$REGION --quiet 2>/dev/null || echo "No existing function to delete"

# Clear local caches and old deployment files
echo "Clearing local caches..."
rm -rf __pycache__ 2>/dev/null
rm -f .gcloudignore 2>/dev/null
rm -f function-source.zip 2>/dev/null

# Wait a moment for deletion to complete
echo "Waiting for cleanup to complete..."
sleep 5

# Create a minimal .gcloudignore
echo "Creating .gcloudignore file..."
cat > .gcloudignore << 'EOF'
# Python cache
__pycache__/
*.pyc

# Virtual environments
venv/
env/
testing_env/
BT_env/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Other scripts and test files
test_*.py
debug_*.py
Video_splitting*.py
check_*.py
deploy_*.sh
*.ps1

# Documentation
*.md
README*

# Logs and debug files
*.log
*.json
EOF

echo ""
echo -e "${GREEN}🚀 Deploying function...${NC}"
echo -e "${CYAN}Using API keys:${NC}"
echo "  DEEPGRAM_API_KEY: ${DEEPGRAM_API_KEY:0:20}..."
echo "  OPENAI_API_KEY: ${OPENAI_API_KEY:0:20}..."
echo "  BUCKET_NAME: $BUCKET_NAME"

# Deploy the function
gcloud functions deploy $FUNCTION_NAME \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point process_video_cloud \
  --source . \
  --memory 8GB \
  --timeout 540s \
  --set-env-vars "LOG_LEVEL=INFO,DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY},OPENAI_API_KEY=${OPENAI_API_KEY},BUCKET_NAME=${BUCKET_NAME}" \
  --region $REGION

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Deployment complete!${NC}"
else
    echo -e "${RED}❌ Deployment failed!${NC}"
    
    # Show build logs
    echo ""
    echo -e "${YELLOW}📋 Checking build logs...${NC}"
    gcloud functions logs read $FUNCTION_NAME --region=$REGION --limit=50
    
    exit 1
fi

# Show the function URL
echo ""
echo -e "${CYAN}Function URL:${NC}"
gcloud functions describe $FUNCTION_NAME --region=$REGION --format="value(httpsTrigger.url)"

# Verify the deployment
echo ""
echo -e "${YELLOW}🔍 Verifying deployment...${NC}"
FUNCTION_STATUS=$(gcloud functions describe $FUNCTION_NAME --region=$REGION --format="value(status)")
echo "Function status: $FUNCTION_STATUS"

# Show recent logs
echo ""
echo -e "${CYAN}📋 Recent logs:${NC}"
gcloud functions logs read $FUNCTION_NAME --region=$REGION --limit=10

# Clean up .gcloudignore after successful deployment
rm -f .gcloudignore 2>/dev/null

echo ""
echo -e "${GREEN}✅ Deployment script completed!${NC}"

# Security reminder
echo ""
echo -e "${YELLOW}⚠️  SECURITY WARNING: This script contains API keys!${NC}"
echo -e "${YELLOW}   Do not commit this file to version control!${NC}"
echo -e "${YELLOW}   Consider using environment variables or a secrets manager for production.${NC}"