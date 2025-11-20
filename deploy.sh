#!/bin/bash
set -e

# Deployment script for Schedule API
SERVER="streameth@pblvrt.com"
REMOTE_DIR="/home/streameth/schedule-api"
LOCAL_DIR="."

echo "🚀 Deploying Schedule API to $SERVER..."

# Create remote directory
echo "📁 Creating remote directory..."
ssh $SERVER "mkdir -p $REMOTE_DIR/data"

# Copy files to remote server
echo "📤 Copying files to remote server..."
rsync -avz --progress \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.db' \
  --exclude '*.csv' \
  --exclude '*.bak' \
  --exclude '.git' \
  $LOCAL_DIR/api.py \
  $LOCAL_DIR/requirements.txt \
  $LOCAL_DIR/Dockerfile \
  $LOCAL_DIR/docker-compose.yml \
  $LOCAL_DIR/docker-stack.yml \
  $LOCAL_DIR/.dockerignore \
  $LOCAL_DIR/service-account.json \
  $LOCAL_DIR/stages.json \
  $SERVER:$REMOTE_DIR/

# Build and start the container on remote server
echo "🐳 Building and starting Docker container..."
ssh $SERVER << 'ENDSSH'
cd /home/streameth/schedule-api
echo "Stopping existing container..."
docker compose down 2>/dev/null || true
echo "Building Docker image..."
docker compose build
echo "Starting container..."
docker compose up -d
echo "Waiting for API to be ready..."
sleep 5
docker compose ps
docker compose logs --tail=20
ENDSSH

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 API Status:"
ssh $SERVER "cd /home/streameth/schedule-api && docker compose ps"
echo ""
echo "🌐 API should be available at: https://devconnect.pblvrt.com"
echo "📚 API Documentation: https://devconnect.pblvrt.com/docs"
echo ""
echo "📝 Useful commands:"
echo "  View logs:    ssh $SERVER 'cd /home/streameth/schedule-api && docker compose logs -f'"
echo "  Restart:      ssh $SERVER 'cd /home/streameth/schedule-api && docker compose restart'"
echo "  Stop:         ssh $SERVER 'cd /home/streameth/schedule-api && docker compose down'"
echo "  Refresh data: curl -X POST https://devconnect.pblvrt.com/refresh"

