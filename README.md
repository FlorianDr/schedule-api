# Schedule API

FastAPI-based REST API for managing event schedules from Google Sheets.

## Features

- 🚀 Fast and async with FastAPI
- 📊 Reads from Google Sheets with batch API calls (rate-limit optimized)
- 💾 SQLite caching for performance
- 🔄 Auto-refresh with manual refresh endpoint
- 🎯 Normalized data output matching JSON schema
- 📝 Support for multiple sheet tabs (Schedule, Digital Assets, Stream Info, Speakers)
- 🐳 Fully dockerized

## API Endpoints

### Core Endpoints
- `GET /events` - List all events
- `GET /events/{folder_id}/schedule` - Get full schedule with metadata
- `GET /events/{folder_id}/sessions` - Get sessions array (JSON schema compliant)
- `GET /schedules` - Get all sessions across all events
- `POST /refresh` - Refresh cache from Google Sheets

### Additional Sheets
- `GET /additional-sheets` - Get all additional sheets data
- `GET /additional-sheets/{sheet_type}` - Get specific sheet type

### Query Parameters
- `?refresh=true` - Force refresh from Google Sheets
- `?include_examples=true` - Include [EXAMPLE] rows (default: false)

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- Google Service Account credentials (`service-account.json`)

### Running with Docker Compose

```bash
# Start the API
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the API
docker-compose down
```

The API will be available at http://localhost:8000

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Local Development

### Prerequisites
- Python 3.13+
- Google Service Account credentials

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Docker Build

```bash
# Build the image
docker build -t schedule-api .

# Run the container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/service-account.json:/app/service-account.json:ro \
  -v $(pwd)/data:/app/data \
  -e DB_PATH=/app/data/cache.db \
  --name schedule-api \
  schedule-api
```

## Configuration

### Environment Variables

- `FOLDER_ID` - Google Drive folder ID containing event subfolders (default: hardcoded in api.py)
- `DB_PATH` - Path to SQLite database file (default: `cache.db`)

### Google Sheets Structure

Each event spreadsheet should have:
- **Main schedule sheet** with columns:
  - Title of the session (required)
  - Day (required)
  - TYPE OF SESSION
  - Start (required)
  - End (required)
  - Speaker 1-6
  - Slides

- **Optional tabs**: Digital Assets, Stream Info, speakers

## Data Persistence

The SQLite cache database is stored in the `./data` directory which is mounted as a Docker volume for persistence across container restarts.

## Health Check

The container includes a health check that verifies the API is responding:

```bash
docker inspect --format='{{json .State.Health}}' schedule-api
```

## Session JSON Schema

Sessions returned by `/events/{folder_id}/sessions` match this schema:

```json
{
  "title": "string (required)",
  "description": "string (required)",
  "day": "string (required, format: DD/MM/YYYY)",
  "start": "string (required, format: HH:MM or HH:MM:SS)",
  "end": "string (required, format: HH:MM or HH:MM:SS)",
  "speakers": ["string array"],
  "placeholderCardUrl": "string (optional, URL)"
}
```

## License

MIT

