"""
BTC Oracle - Supabase Helper
Talks to Supabase using its REST API directly.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}


def insert(table, data):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    resp = requests.post(url, json=data, headers=HEADERS)
    if resp.status_code in (200, 201):
        return resp.json()
    else:
        print(f"Insert error ({table}): {resp.status_code} - {resp.text}")
        return None


def select(table, params=""):
    url = f"{SUPABASE_URL}/rest/v1/{table}?{params}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Select error ({table}): {resp.status_code} - {resp.text}")
        return []


def update(table, match_column, match_value, data):
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_column}=eq.{match_value}"
    resp = requests.patch(url, json=data, headers=HEADERS)
    if resp.status_code in (200, 204):
        return resp.json() if resp.text else True
    else:
        print(f"Update error ({table}): {resp.status_code} - {resp.text}")
        return None
