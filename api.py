from typing import List, Dict, Any, Optional
import json
import os
import sqlite3
import time
import threading
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field
from openai import OpenAI


SERVICE_ACCOUNT_FILE = "service-account.json"
STAGES_FILE = "stages.json"
# Read-only scopes are sufficient to list and read Sheets
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
]

# Root events folder (one subfolder per event)
FOLDER_ID = "11TpKl8skEaUC-CWBZkxJli79YX--tPsZ"

# Expected headers for schedule sheets
EXPECTED_SCHEDULE_HEADERS = [
    "ID",
    "Title of the session (required)",
    "Day (required)",
    "TYPE OF SESSION",
    "Start (required)",
    "Timer",
    "End (required)",
    "Speaker 1",
    "Speaker 2",
    "Speaker 3",
    "Speaker 4",
    "Speaker 5",
    "Speaker 6",
    "Slides"
]

# OpenAI API key - REQUIRED: set via environment variable OPENAI_API_KEY
# Example: export OPENAI_API_KEY="sk-..."
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Pydantic models for structured output
class Session(BaseModel):
    """Session model matching the JSON schema."""
    id: str = Field(default="", description="Session ID")
    title: str = Field(description="Title of the session")
    day: str = Field(description="Date in DD/MM/YYYY format")
    type: str = Field(default="", description="Type of session")
    start: str = Field(description="Start time in HH:MM format")
    timer: str = Field(default="", description="Timer value")
    end: str = Field(description="End time in HH:MM format")
    speakers: List[str] = Field(default_factory=list, description="List of speaker names")
    placeholderCardUrl: Optional[str] = Field(default=None, description="URL to placeholder card or slides")


class SessionsResponse(BaseModel):
    """Response containing list of sessions."""
    sessions: List[Session] = Field(description="List of parsed sessions")


def load_stages_mapping() -> Dict[str, str]:
    """Load stages.json and create a reverse mapping from event name to stage name."""
    try:
        with open(STAGES_FILE, "r", encoding="utf-8") as f:
            stages_data = json.load(f)
        
        # Create reverse mapping: event_name -> stage_name
        event_to_stage: Dict[str, str] = {}
        for stage_name, event_names in stages_data.items():
            for event_name in event_names:
                if event_name:  # Skip empty strings
                    event_to_stage[event_name] = stage_name
        
        return event_to_stage
    except FileNotFoundError:
        # If stages.json doesn't exist, return empty dict
        return {}
    except json.JSONDecodeError:
        # If stages.json is invalid, return empty dict
        return {}


def get_stage_for_event(event_name: str, stages_mapping: Dict[str, str]) -> str:
    """Get the stage name for a given event name."""
    return stages_mapping.get(event_name, "")


def parse_date_for_sorting(date_str: str) -> tuple:
    """
    Parse date string in DD/MM/YYYY format and return tuple for sorting.
    Returns (year, month, day) or (0, 0, 0) if parsing fails.
    """
    try:
        if not date_str:
            return (0, 0, 0)
        parts = date_str.split("/")
        if len(parts) == 3:
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            return (year, month, day)
    except (ValueError, IndexError):
        pass
    return (0, 0, 0)


def parse_time_for_sorting(time_str: str) -> tuple:
    """
    Parse time string in HH:MM format and return tuple for sorting.
    Returns (hour, minute) or (0, 0) if parsing fails.
    """
    try:
        if not time_str:
            return (0, 0)
        parts = time_str.split(":")
        if len(parts) >= 2:
            hour, minute = int(parts[0]), int(parts[1])
            return (hour, minute)
    except (ValueError, IndexError):
        pass
    return (0, 0)


def parse_sessions_with_openai(raw_rows: List[List[str]], include_examples: bool = False) -> List[Dict[str, Any]]:
    """
    Parse raw sheet rows into normalized session objects using OpenAI API.
    
    Args:
        raw_rows: 2D array of cell values from Google Sheets
        include_examples: Whether to include rows starting with [EXAMPLE]
        
    Returns:
        List of normalized session dictionaries
        
    Raises:
        Exception: If OpenAI API call fails
    """
    if not raw_rows:
        return []
    
    # Convert raw rows to JSON string for OpenAI
    rows_json = json.dumps(raw_rows)
    
    example_instruction = "- Skip rows where title starts with [EXAMPLE]" if not include_examples else "- Include rows where title starts with [EXAMPLE]"
    
    system_prompt = f"""You are a data parser that extracts session information from Google Sheets data.

The input is a 2D array representing rows and columns from a spreadsheet. The first row typically contains headers.

Your task:
1. Identify which columns correspond to: ID, Title, Day (date), Type, Start time, Timer, End time, Speaker columns, and Slides/Placeholder URLs
2. Extract valid session data rows (skip instruction rows, header rows, and empty rows)
3. For speakers: extract names from speaker columns, filtering out instruction text like "CHOSE", "ENTER", "SELECT", "⬇️", "SPEAKER 1", "SPEAKER 2", etc.
4. Only include placeholderCardUrl if there's an actual URL value
5. Normalize all data to match the Session schema
6. Return all valid sessions as a structured list

Important filters:
- Skip rows where title contains "INSTRUCTIONS" or "INSTRUCTION"
{example_instruction}
- Skip rows that are clearly headers (like "Title of the session")
- Skip completely empty rows
- A valid session must have at least a title and a day"""

    try:
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Parse these sheet rows into sessions:\n\n{rows_json}"}
            ],
            response_format=SessionsResponse,
        )
        
        result = completion.choices[0].message.parsed
        if result and result.sessions:
            # Convert Pydantic models to dicts, excluding None values for placeholderCardUrl
            sessions = []
            for session in result.sessions:
                session_dict = session.model_dump(exclude_none=True)
                sessions.append(session_dict)
            return sessions
        return []
        
    except Exception as e:
        print(f"OpenAI parsing error: {e}")
        raise


def get_credentials():
    return service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )


def build_drive_service(creds):
    return build("drive", "v3", credentials=creds)


def build_sheets_service(creds):
    return build("sheets", "v4", credentials=creds)


def list_event_subfolders(drive_service, folder_id: str) -> List[Dict[str, str]]:
    results = drive_service.files().list(  # type: ignore
        q=f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id, name, mimeType)",
    ).execute()
    return results.get("files", [])


def find_sheet_in_folder(drive_service, folder_id: str) -> Dict[str, str] | None:
    results = drive_service.files().list(  # type: ignore
        q=(
            f"'{folder_id}' in parents and "
            "mimeType = 'application/vnd.google-apps.spreadsheet' and trashed = false"
        ),
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id, name, mimeType)",
    ).execute()
    files = results.get("files", [])
    return files[0] if files else None


def read_sheet_records(sheets_service, spreadsheet_id: str, headers: List[str], sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
    # Read all data from a specific sheet or first sheet by default
    range_spec = f"'{sheet_name}'!A:Z" if sheet_name else "A:Z"
    values_resp = (
        sheets_service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_spec)
        .execute()
    )
    rows = values_resp.get("values", [])
    if not rows:
        return []

    # Use the actual first row as headers if it looks like a header row
    start_index = 0
    actual_headers = headers
    
    if rows:
        first_row = [str(c).strip() for c in rows[0]]
        # Check if first row looks like headers (contains "required" or common keywords)
        if any(keyword in str(cell).lower() for cell in first_row for keyword in ["required", "title", "speaker", "day"]):
            actual_headers = first_row
            start_index = 1

    records: List[Dict[str, Any]] = []
    for row in rows[start_index:]:
        padded = list(row) + ([""] * (len(actual_headers) - len(row)))
        record = {actual_headers[i]: padded[i] for i in range(len(actual_headers))}
        records.append(record)
    return records


def read_generic_sheet(sheets_service, spreadsheet_id: str, sheet_name: str) -> List[Dict[str, Any]]:
    """Read a sheet with headers in the first row, return list of dicts."""
    range_spec = f"'{sheet_name}'!A:Z"
    try:
        values_resp = (
            sheets_service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=range_spec)
            .execute()
        )
    except Exception:  # pylint: disable=broad-except
        # Sheet might not exist or might be inaccessible
        return []
    
    rows = values_resp.get("values", [])
    if not rows:
        return []
    
    # First row is headers
    headers = [str(cell).strip() for cell in rows[0]]
    if not headers:
        return []
    
    records: List[Dict[str, Any]] = []
    for row in rows[1:]:
        padded = list(row) + ([""] * (len(headers) - len(row)))
        record = {headers[i]: padded[i] for i in range(len(headers))}
        records.append(record)
    return records


def read_all_sheets_batch(sheets_service, spreadsheet_id: str, schedule_headers: List[str], additional_sheet_names: List[str]) -> Dict[str, Any]:
    """Read main schedule and additional sheets in a single batch API call."""
    # First, get the list of available sheets in the spreadsheet
    max_retries = 3
    retry_delay = 2
    
    try:
        spreadsheet_metadata = (
            sheets_service.spreadsheets()
            .get(spreadsheetId=spreadsheet_id, fields="sheets.properties")
            .execute()
        )
        available_sheet_names = [sheet["properties"]["title"] for sheet in spreadsheet_metadata.get("sheets", [])]
    except Exception:
        # If we can't get metadata, just try with the main sheet
        available_sheet_names = []
    
    # Build ranges for batch request - only include sheets that exist
    ranges = ["A:Z"]  # Main sheet (agenda/schedule)
    sheet_name_mapping = {}  # Track which additional sheets we're requesting
    
    # Create case-insensitive mapping of available sheet names
    available_sheets_lower = {name.lower(): name for name in available_sheet_names}
    
    for sheet_name in additional_sheet_names:
        # Case-insensitive matching
        matched_name = available_sheets_lower.get(sheet_name.lower())
        if matched_name:
            ranges.append(f"'{matched_name}'!A:Z")
            sheet_name_mapping[len(ranges) - 1] = sheet_name  # Use original name for consistency
    
    # Retry logic with exponential backoff for rate limits
    last_error = None
    for attempt in range(max_retries):
        try:
            # Use batchGet to read all ranges in a single API call
            result = (
                sheets_service.spreadsheets()
                .values()
                .batchGet(spreadsheetId=spreadsheet_id, ranges=ranges)
                .execute()
            )
            value_ranges = result.get("valueRanges", [])
            break  # Success, exit retry loop
        except HttpError as e:
            last_error = e
            if e.resp.status == 429:  # Rate limit exceeded
                if attempt < max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Last attempt failed, raise the error
                    print(f"Rate limit exceeded after {max_retries} attempts")
                    raise
            else:
                # Not a rate limit error, raise immediately
                raise
    
    if last_error and not value_ranges:
        raise last_error
    
    response: Dict[str, Any] = {
        "schedule": [],
        "additional": {}
    }
    
    # Return raw rows from main schedule (first range) for OpenAI parsing
    if value_ranges and len(value_ranges) > 0:
        rows = value_ranges[0].get("values", [])
        if rows:
            response["schedule"] = rows
    
    # Process additional sheets - use the mapping to know which sheets we got
    for range_idx, sheet_name in sheet_name_mapping.items():
        if range_idx < len(value_ranges):
            rows = value_ranges[range_idx].get("values", [])
            if rows and len(rows) > 0:
                # First row is headers
                headers = [str(cell).strip() for cell in rows[0] if cell]
                # Filter out empty headers
                headers = [h for h in headers if h]
                if headers:
                    records = []
                    for row in rows[1:]:
                        if row:  # Skip completely empty rows
                            padded = list(row) + ([""] * (len(headers) - len(row)))
                            record = {headers[i]: padded[i] if i < len(padded) else "" for i in range(len(headers))}
                            records.append(record)
                    if records:  # Only add if we have records
                        response["additional"][sheet_name] = records
    
    return response


def is_valid_session(row: Dict[str, Any], include_examples: bool = False) -> bool:
    """Check if a session row is valid (not instructions, headers, or examples)."""
    # Support both old and new field names for backward compatibility
    title = row.get("title", row.get("name", "")).strip()
    day = row.get("day", row.get("date", "")).strip()
    
    # Filter out empty rows
    if not title:
        return False
    
    # Filter out instruction rows
    if "INSTRUCTIONS" in title.upper() or "INSTRUCTION" in title.upper():
        return False
    
    # Filter out header rows (rows where the title matches the expected header)
    if title.lower().startswith("title of the session"):
        return False
    
    # Filter out example rows (configurable)
    if not include_examples and title.startswith("[EXAMPLE]"):
        return False
    
    # Must have at least a title and day to be valid
    if not title or not day:
        return False
    
    return True


def is_valid_row(row: Dict[str, Any], include_examples: bool = False) -> bool:
    """Check if a row is valid data (not instructions, headers, or examples)."""
    # Get first few values to check
    values = [str(v).strip() for v in row.values() if v]
    if not values:
        return False
    
    first_value = values[0] if values else ""
    
    # Filter out instruction indicators (not including EXAMPLE in the list)
    instruction_keywords = ["INSTRUCTIONS", "INSTRUCTION", "ENTER THE", "SELECT THE", "CHOSE", "FILL OUT", "⬇️"]
    if any(keyword in first_value.upper() for keyword in instruction_keywords):
        return False
    
    # Filter out rows that start with [EXAMPLE] (configurable)
    if not include_examples and first_value.startswith("[EXAMPLE]"):
        return False
    
    return True


def normalize_digital_asset(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Digital Assets sheet data."""
    # Common field patterns for digital assets
    name_key = next((k for k in row.keys() if any(x in k.lower() for x in ["name", "title", "asset"])), None)
    type_key = next((k for k in row.keys() if "type" in k.lower()), None)
    url_key = next((k for k in row.keys() if "url" in k.lower() or "link" in k.lower()), None)
    desc_key = next((k for k in row.keys() if "desc" in k.lower() or "notes" in k.lower()), None)
    status_key = next((k for k in row.keys() if "status" in k.lower()), None)
    owner_key = next((k for k in row.keys() if "owner" in k.lower() or "author" in k.lower() or "creator" in k.lower()), None)
    
    # Build normalized object with all original data plus normalized fields
    normalized = {
        "name": (row.get(name_key) or "").strip() if name_key else "",
        "type": (row.get(type_key) or "").strip() if type_key else "",
        "url": (row.get(url_key) or "").strip() if url_key else "",
        "description": (row.get(desc_key) or "").strip() if desc_key else "",
        "status": (row.get(status_key) or "").strip() if status_key else "",
        "owner": (row.get(owner_key) or "").strip() if owner_key else "",
    }
    
    # Include all original fields as well for flexibility
    for key, value in row.items():
        if key not in normalized:
            normalized[key] = (value or "").strip() if isinstance(value, str) else value
    
    return normalized


def normalize_stream_info(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Stream Info sheet data."""
    # Common field patterns for stream info
    platform_key = next((k for k in row.keys() if "platform" in k.lower()), None)
    url_key = next((k for k in row.keys() if "url" in k.lower() or "link" in k.lower() or "stream" in k.lower()), None)
    key_key = next((k for k in row.keys() if "key" in k.lower() or "token" in k.lower()), None)
    status_key = next((k for k in row.keys() if "status" in k.lower()), None)
    start_key = next((k for k in row.keys() if "start" in k.lower()), None)
    end_key = next((k for k in row.keys() if "end" in k.lower()), None)
    notes_key = next((k for k in row.keys() if "note" in k.lower() or "comment" in k.lower()), None)
    
    normalized = {
        "platform": (row.get(platform_key) or "").strip() if platform_key else "",
        "url": (row.get(url_key) or "").strip() if url_key else "",
        "streamKey": (row.get(key_key) or "").strip() if key_key else "",
        "status": (row.get(status_key) or "").strip() if status_key else "",
        "startTime": (row.get(start_key) or "").strip() if start_key else "",
        "endTime": (row.get(end_key) or "").strip() if end_key else "",
        "notes": (row.get(notes_key) or "").strip() if notes_key else "",
    }
    
    # Include all original fields as well
    for key, value in row.items():
        if key not in normalized:
            normalized[key] = (value or "").strip() if isinstance(value, str) else value
    
    return normalized


def normalize_speaker(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize speakers sheet data."""
    # Common field patterns for speakers
    name_key = next((k for k in row.keys() if "name" in k.lower() and "company" not in k.lower()), None)
    email_key = next((k for k in row.keys() if "email" in k.lower() or "mail" in k.lower()), None)
    company_key = next((k for k in row.keys() if "company" in k.lower() or "organization" in k.lower()), None)
    title_key = next((k for k in row.keys() if "title" in k.lower() or "position" in k.lower() or "role" in k.lower()), None)
    bio_key = next((k for k in row.keys() if "bio" in k.lower() or "description" in k.lower()), None)
    photo_key = next((k for k in row.keys() if "photo" in k.lower() or "image" in k.lower() or "picture" in k.lower()), None)
    linkedin_key = next((k for k in row.keys() if "linkedin" in k.lower()), None)
    twitter_key = next((k for k in row.keys() if "twitter" in k.lower() or "x.com" in k.lower()), None)
    website_key = next((k for k in row.keys() if "website" in k.lower() or "url" in k.lower()), None)
    
    normalized = {
        "name": (row.get(name_key) or "").strip() if name_key else "",
        "email": (row.get(email_key) or "").strip() if email_key else "",
        "company": (row.get(company_key) or "").strip() if company_key else "",
        "title": (row.get(title_key) or "").strip() if title_key else "",
        "bio": (row.get(bio_key) or "").strip() if bio_key else "",
        "photoUrl": (row.get(photo_key) or "").strip() if photo_key else "",
        "linkedinUrl": (row.get(linkedin_key) or "").strip() if linkedin_key else "",
        "twitterUrl": (row.get(twitter_key) or "").strip() if twitter_key else "",
        "websiteUrl": (row.get(website_key) or "").strip() if website_key else "",
    }
    
    # Include all original fields as well
    for key, value in row.items():
        if key not in normalized:
            normalized[key] = (value or "").strip() if isinstance(value, str) else value
    
    return normalized


def normalize_additional_sheet(sheet_type: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """Route to the appropriate normalization function based on sheet type."""
    if sheet_type == "Digital Assets":
        return normalize_digital_asset(row)
    elif sheet_type == "Stream Info":
        return normalize_stream_info(row)
    elif sheet_type == "speakers":
        return normalize_speaker(row)
    else:
        # Return raw data for unknown sheet types
        return row


app = FastAPI()

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- Refresh status tracking ---
refresh_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_event": "",
    "started_at": None,
    "completed_at": None,
    "error": None
}
refresh_status_lock = threading.Lock()


# --- SQLite cache helpers ---
DB_PATH = os.getenv("DB_PATH", "cache.db")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              folder_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              sheet_id TEXT,
              sheet_name TEXT,
              updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schedules (
              sheet_id TEXT NOT NULL,
              folder_id TEXT NOT NULL,
              event_name TEXT NOT NULL,
              headers_json TEXT NOT NULL,
              rows_json TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(sheet_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS additional_sheets (
              sheet_id TEXT NOT NULL,
              folder_id TEXT NOT NULL,
              sheet_type TEXT NOT NULL,
              rows_json TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(sheet_id, sheet_type)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def upsert_events(events: List[Dict[str, str]]) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.cursor()
        for e in events:
            cur.execute(
                """
                INSERT INTO events (folder_id, name, sheet_id, sheet_name, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(folder_id) DO UPDATE SET
                  name=excluded.name,
                  sheet_id=excluded.sheet_id,
                  sheet_name=excluded.sheet_name,
                  updated_at=excluded.updated_at
                """,
                (e.get("folderId", ""), e.get("name", ""), e.get("sheetId", ""), e.get("sheetName", ""), now),
            )
        conn.commit()
    finally:
        conn.close()


def upsert_schedule(folder_id: str, event_name: str, sheet_id: str, headers: List[str], rows: List[Dict[str, Any]]) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO schedules (sheet_id, folder_id, event_name, headers_json, rows_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sheet_id) DO UPDATE SET
              folder_id=excluded.folder_id,
              event_name=excluded.event_name,
              headers_json=excluded.headers_json,
              rows_json=excluded.rows_json,
              updated_at=excluded.updated_at
            """,
            (sheet_id, folder_id, event_name, json.dumps(headers), json.dumps(rows), now),
        )
        conn.commit()
    finally:
        conn.close()


def get_cached_events() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT folder_id, name, sheet_id, sheet_name, updated_at FROM events ORDER BY name")
        rows = cur.fetchall()
        
        # Load stages mapping
        stages_mapping = load_stages_mapping()
        
        return [
            {
                "folderId": r[0],
                "name": r[1],
                "sheetId": r[2] or "",
                "sheetName": r[1],  # Use event name instead of sheet name
                "stage": get_stage_for_event(r[1], stages_mapping),
                "updatedAt": r[4],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_cached_schedules() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT folder_id, event_name, sheet_id, headers_json, rows_json, updated_at FROM schedules")
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "folderId": r[0],
                    "event": r[1],
                    "sheetId": r[2],
                    "headers": json.loads(r[3]),
                    "rows": json.loads(r[4]),
                    "updatedAt": r[5],
                }
            )
        return out
    finally:
        conn.close()


def upsert_additional_sheet(sheet_id: str, folder_id: str, sheet_type: str, rows: List[Dict[str, Any]]) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO additional_sheets (sheet_id, folder_id, sheet_type, rows_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(sheet_id, sheet_type) DO UPDATE SET
              folder_id=excluded.folder_id,
              rows_json=excluded.rows_json,
              updated_at=excluded.updated_at
            """,
            (sheet_id, folder_id, sheet_type, json.dumps(rows), now),
        )
        conn.commit()
    finally:
        conn.close()


def get_cached_additional_sheets(folder_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        if folder_id:
            cur.execute("SELECT sheet_type, rows_json FROM additional_sheets WHERE folder_id = ?", (folder_id,))
        else:
            cur.execute("SELECT sheet_type, rows_json FROM additional_sheets")
        rows = cur.fetchall()
        result: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            sheet_type = r[0]
            data = json.loads(r[1])
            if sheet_type not in result:
                result[sheet_type] = []
            result[sheet_type].extend(data)
        return result
    finally:
        conn.close()


def refresh_cache() -> Dict[str, Any]:
    global refresh_status
    
    # Initialize status
    with refresh_status_lock:
        refresh_status = {
            "running": True,
            "progress": 0,
            "total": 0,
            "current_event": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None
        }
    
    try:
        creds = get_credentials()
        drive = build_drive_service(creds)
        sheets = build_sheets_service(creds)
        headers = EXPECTED_SCHEDULE_HEADERS

        subfolders = list_event_subfolders(drive, FOLDER_ID)
        events: List[Dict[str, str]] = []
        for folder in subfolders:
            sheet = find_sheet_in_folder(drive, folder["id"])  # may be None
            events.append(
                {
                    "folderId": folder["id"],
                    "name": folder["name"],
                    "sheetId": sheet["id"] if sheet else "",
                    "sheetName": sheet["name"] if sheet else "",
                }
            )

        upsert_events(events)
        
        # Count events with sheets
        events_with_sheets = [e for e in events if e["sheetId"]]
        
        with refresh_status_lock:
            refresh_status["total"] = len(events_with_sheets)

        refreshed = 0
        additional_sheet_names = ["Digital Assets", "Stream Info", "speakers"]
        
        for e in events:
            if not e["sheetId"]:
                continue
            
            # Update status
            with refresh_status_lock:
                refresh_status["current_event"] = e["name"]
                refresh_status["progress"] = refreshed
            
            try:
                # Use batch read to get all sheets in one API call
                batch_data = read_all_sheets_batch(sheets, e["sheetId"], headers, additional_sheet_names)
                
                # Process main schedule with OpenAI (always include examples in cache, filter on read)
                schedule_rows = batch_data.get("schedule", [])
                valid_sessions = parse_sessions_with_openai(schedule_rows, include_examples=True)
                upsert_schedule(e["folderId"], e["name"], e["sheetId"], headers, valid_sessions)
                
                # Process additional sheets
                for sheet_name, additional_data in batch_data.get("additional", {}).items():
                    if additional_data:
                        normalized_additional = [normalize_additional_sheet(sheet_name, row) for row in additional_data]
                        # Filter out invalid/instruction rows
                        valid_additional = [r for r in normalized_additional if is_valid_row(r)]
                        if valid_additional:
                            upsert_additional_sheet(e["sheetId"], e["folderId"], sheet_name, valid_additional)
                
                refreshed += 1
                
                # Update progress
                with refresh_status_lock:
                    refresh_status["progress"] = refreshed
                
                # Add a small delay between spreadsheets to avoid rate limits
                if refreshed < len(events_with_sheets):
                    time.sleep(0.5)
                    
            except HttpError as err:
                if err.resp.status == 429:
                    # Rate limit hit, wait longer and continue
                    print(f"Rate limit hit for {e['name']}, waiting 10 seconds...")
                    time.sleep(10)
                    continue
                else:
                    # Other error, log and continue
                    print(f"Error reading sheet for {e['name']}: {err}")
                    continue

        result = {"events": len(events), "schedules": refreshed}
        
        # Mark as completed
        with refresh_status_lock:
            refresh_status["running"] = False
            refresh_status["completed_at"] = datetime.now(timezone.utc).isoformat()
            refresh_status["current_event"] = ""
        
        return result
        
    except Exception as e:
        # Mark as failed
        with refresh_status_lock:
            refresh_status["running"] = False
            refresh_status["completed_at"] = datetime.now(timezone.utc).isoformat()
            refresh_status["error"] = str(e)
        raise


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/events")
def get_events(refresh: Optional[bool] = False) -> List[Dict[str, Any]]:
    init_db()
    if refresh:
        refresh_cache()
    cached = get_cached_events()
    if cached:
        return cached
    # Populate once if empty
    refresh_cache()
    return get_cached_events()


@app.get("/events/{folder_id}/schedule")
def get_event_schedule(folder_id: str, refresh: Optional[bool] = False, include_examples: Optional[bool] = False) -> Dict[str, Any]:
    init_db()
    if refresh:
        refresh_cache()
    # Try cached schedule first
    cached_all = get_cached_schedules()
    result: Dict[str, Any] = {}
    for entry in cached_all:
        if entry["folderId"] == folder_id:
            result = entry.copy()
            break

    if not result:
        # If not cached yet, refresh only this folder
        creds = get_credentials()
        drive = build_drive_service(creds)
        sheet_file = find_sheet_in_folder(drive, folder_id)
        if not sheet_file:
            raise HTTPException(status_code=404, detail="No Google Sheet found in this folder")

        sheets = build_sheets_service(creds)
        headers = EXPECTED_SCHEDULE_HEADERS
        additional_sheet_names = ["Digital Assets", "Stream Info", "speakers"]
        
        # Use batch read to get all sheets in one API call
        batch_data = read_all_sheets_batch(sheets, sheet_file["id"], headers, additional_sheet_names)
        
        # Process main schedule with OpenAI
        schedule_rows = batch_data.get("schedule", [])
        valid_rows = parse_sessions_with_openai(schedule_rows, include_examples=include_examples)
        
        # Need event name for cache record
        # mypy/pylance cannot see dynamic methods on Resource
        folder_info = drive.files().get(fileId=folder_id, supportsAllDrives=True, fields="id,name").execute()  # type: ignore  # pylint: disable=no-member
        upsert_schedule(folder_id, folder_info.get("name", ""), sheet_file["id"], headers, valid_rows)
        
        # Process additional sheets
        for sheet_name, additional_data in batch_data.get("additional", {}).items():
            if additional_data:
                normalized_additional = [normalize_additional_sheet(sheet_name, row) for row in additional_data]
                # Filter out invalid/instruction rows
                valid_additional = [r for r in normalized_additional if is_valid_row(r, include_examples=include_examples)]
                if valid_additional:
                    upsert_additional_sheet(sheet_file["id"], folder_id, sheet_name, valid_additional)
        
        result = {
            "folderId": folder_id,
            "event": folder_info.get("name", ""),
            "sheetId": sheet_file["id"],
            "headers": headers,
            "rows": valid_rows,
        }
    else:
        # If using cached data, apply the filter based on include_examples parameter
        if not include_examples:
            result["rows"] = [r for r in result.get("rows", []) if is_valid_session(r, include_examples=False)]
    
    # Add additional sheets data (also filtered based on include_examples)
    additional_sheets = get_cached_additional_sheets(folder_id)
    if not include_examples:
        filtered_additional = {}
        for sheet_type, rows in additional_sheets.items():
            filtered_rows = [r for r in rows if is_valid_row(r, include_examples=False)]
            if filtered_rows:
                filtered_additional[sheet_type] = filtered_rows
        result["additionalSheets"] = filtered_additional
    else:
        result["additionalSheets"] = additional_sheets
    
    return result


@app.get("/events/{folder_id}/sessions")
def get_event_sessions(folder_id: str, refresh: Optional[bool] = False, include_examples: Optional[bool] = False) -> List[Dict[str, Any]]:
    """
    Get sessions for a specific event as an array of Session objects matching the JSON schema.
    Returns only the sessions array without metadata, sorted by date and start time.
    """
    # Get the full schedule data
    schedule_data = get_event_schedule(folder_id, refresh, include_examples)
    sessions = schedule_data.get("rows", [])
    
    # Sort by date (day), then by start time
    sessions.sort(key=lambda s: (
        parse_date_for_sorting(s.get("day", "")),
        parse_time_for_sorting(s.get("start", ""))
    ))
    
    return sessions


@app.get("/schedules")
def get_all_schedules(refresh: Optional[bool] = False, include_examples: Optional[bool] = False) -> List[Dict[str, Any]]:
    """Get all schedules across all events. Each session includes an 'event' field indicating which event it belongs to. Results are sorted by date and start time."""
    init_db()
    if refresh:
        refresh_cache()
    cached = get_cached_schedules()
    if cached:
        # Flatten to a single list of rows with event name
        flattened: List[Dict[str, Any]] = []
        for entry in cached:
            event_name = entry.get("event", "")
            rows = entry.get("rows", [])
            # Filter based on include_examples
            if not include_examples:
                rows = [r for r in rows if is_valid_session(r, include_examples=False)]
            for r in rows:
                flattened.append({"event": event_name, **r})
        
        # Sort by date (day), then by start time
        flattened.sort(key=lambda s: (
            parse_date_for_sorting(s.get("day", "")),
            parse_time_for_sorting(s.get("start", ""))
        ))
        
        return flattened
    # If empty, refresh and return
    refresh_cache()
    return get_all_schedules(refresh=False, include_examples=include_examples)


@app.get("/sessions")
def get_filtered_sessions(
    stage: Optional[str] = None,
    date: Optional[str] = None,
    refresh: Optional[bool] = False,
    include_examples: Optional[bool] = False
) -> List[Dict[str, Any]]:
    """
    Get all sessions filtered by stage and/or date.
    
    Parameters:
    - stage: Filter by stage name (e.g., "M1", "M2", "Auditorium", etc.)
    - date: Filter by date (e.g., "22/11/2025")
    - refresh: Force refresh cache from Google Sheets
    - include_examples: Include example rows from sheets
    
    Returns a list of sessions with event name and stage included.
    """
    init_db()
    if refresh:
        refresh_cache()
    
    # Load stages mapping
    stages_mapping = load_stages_mapping()
    
    # Get all schedules with event names
    all_sessions = get_all_schedules(refresh=False, include_examples=include_examples)
    
    # Filter by stage if provided
    if stage:
        filtered_sessions = []
        for session in all_sessions:
            event_name = session.get("event", "")
            event_stage = get_stage_for_event(event_name, stages_mapping)
            if event_stage.lower() == stage.lower():
                # Add stage to the session data
                session_with_stage = {**session, "stage": event_stage}
                filtered_sessions.append(session_with_stage)
        all_sessions = filtered_sessions
    else:
        # Add stage to all sessions even if not filtering
        for session in all_sessions:
            event_name = session.get("event", "")
            event_stage = get_stage_for_event(event_name, stages_mapping)
            session["stage"] = event_stage
    
    # Filter by date if provided
    if date:
        all_sessions = [s for s in all_sessions if s.get("day", "") == date]
    
    # Sort by date (day), then by start time
    all_sessions.sort(key=lambda s: (
        parse_date_for_sorting(s.get("day", "")),
        parse_time_for_sorting(s.get("start", ""))
    ))
    
    return all_sessions


@app.get("/events/{folder_id}/additional-sheets")
def get_event_additional_sheets(folder_id: str, refresh: Optional[bool] = False, include_examples: Optional[bool] = False) -> Dict[str, Any]:
    """Get all additional sheets (Digital Assets, Stream Info, speakers) for a specific event."""
    init_db()
    if refresh:
        refresh_cache()
    all_sheets = get_cached_additional_sheets(folder_id=folder_id)
    
    # Filter based on include_examples
    if not include_examples:
        filtered = {}
        for sheet_type, rows in all_sheets.items():
            filtered_rows = [r for r in rows if is_valid_row(r, include_examples=False)]
            if filtered_rows:
                filtered[sheet_type] = filtered_rows
        return filtered
    
    return all_sheets


@app.get("/events/{folder_id}/additional-sheets/{sheet_type}")
def get_event_additional_sheet_by_type(folder_id: str, sheet_type: str, refresh: Optional[bool] = False, include_examples: Optional[bool] = False) -> List[Dict[str, Any]]:
    """Get a specific type of additional sheet for a specific event."""
    init_db()
    if refresh:
        refresh_cache()
    all_sheets = get_cached_additional_sheets(folder_id=folder_id)
    rows = all_sheets.get(sheet_type, [])
    
    # Filter based on include_examples
    if not include_examples:
        rows = [r for r in rows if is_valid_row(r, include_examples=False)]
    
    return rows


@app.get("/refresh/status")
def get_refresh_status() -> Dict[str, Any]:
    """
    Get the current status of the background refresh job.
    Returns progress, current event being processed, and completion status.
    """
    with refresh_status_lock:
        status = refresh_status.copy()
    
    # Calculate percentage if total is known
    percentage = None
    if status["total"] > 0:
        percentage = round((status["progress"] / status["total"]) * 100, 1)
    
    return {
        **status,
        "percentage": percentage
    }


@app.post("/refresh")
def post_refresh(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Trigger a background refresh of the cache from Google Sheets.
    Returns immediately while the refresh runs in the background.
    Use GET /refresh/status to track progress.
    """
    init_db()
    background_tasks.add_task(refresh_cache)
    return {
        "status": "ok",
        "message": "Refresh started in background. This may take a few minutes to complete.",
        "status_endpoint": "/refresh/status"
    }


