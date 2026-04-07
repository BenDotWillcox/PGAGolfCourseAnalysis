"""Data Golf API client — pulls player skill ratings, rankings, and decompositions."""

import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent.parent / "data"

# Load API key from .env.local
load_dotenv(Path(__file__).parent.parent / ".env.local")
API_KEY = os.getenv("DataGolf_API_Key", "")

BASE_URL = "https://feeds.datagolf.com"


def _get(endpoint: str, params: dict | None = None) -> dict:
    params = params or {}
    params["key"] = API_KEY
    params.setdefault("file_format", "json")
    resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_skill_ratings() -> pd.DataFrame:
    """Current SG skill ratings for ~434 active players."""
    data = _get("preds/skill-ratings")
    df = pd.DataFrame(data["players"])
    df["last_updated"] = data["last_updated"]
    return df


def fetch_rankings() -> pd.DataFrame:
    """Top 500 Data Golf rankings with overall skill estimate."""
    data = _get("preds/get-dg-rankings")
    df = pd.DataFrame(data["rankings"])
    df["last_updated"] = data["last_updated"]
    return df


def fetch_player_decompositions() -> tuple[pd.DataFrame, dict]:
    """Course-fit decompositions for the current week's field."""
    data = _get("preds/player-decompositions")
    df = pd.DataFrame(data["players"])
    meta = {
        "event_name": data.get("event_name"),
        "course_name": data.get("course_name"),
        "last_updated": data.get("last_updated"),
    }
    return df, meta


def fetch_field_updates() -> tuple[pd.DataFrame, dict]:
    """Current tournament field and event info."""
    data = _get("field-updates")
    meta = {
        "event_name": data.get("event_name"),
        "course_name": data.get("course_name"),
    }
    # field-updates returns a different shape — players under various keys
    if "field" in data:
        df = pd.DataFrame(data["field"])
    else:
        df = pd.DataFrame()
    return df, meta


def save_skill_ratings():
    """Pull skill ratings and save to CSV."""
    df = fetch_skill_ratings()
    df.to_csv(DATA_DIR / "skill_ratings.csv", index=False)
    print(f"Saved {len(df)} player skill ratings")
    return df


def save_rankings():
    """Pull rankings and save to CSV."""
    df = fetch_rankings()
    df.to_csv(DATA_DIR / "rankings.csv", index=False)
    print(f"Saved {len(df)} player rankings")
    return df


if __name__ == "__main__":
    save_skill_ratings()
    save_rankings()
