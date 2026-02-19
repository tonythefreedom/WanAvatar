#!/usr/bin/env python3
"""Re-authenticate YouTube OAuth2 token (headless server).

1. Run this script → prints an auth URL
2. Open the URL in your local browser → authorize with Google
3. Browser redirects to localhost (page won't load — that's OK)
4. Copy the FULL URL from the browser address bar
5. Paste it here → token is saved

Usage:
    python refresh_youtube_token.py
"""
import json
from google_auth_oauthlib.flow import Flow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRET = "client_secret.json"
TOKEN_FILE = "token.json"
REDIRECT_URI = "http://localhost:1"  # won't connect, but code is in the URL


def main():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET, scopes=SCOPES, redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline", prompt="consent"
    )

    print("\n" + "=" * 60)
    print("Open this URL in your browser:\n")
    print(auth_url)
    print("\n" + "=" * 60)
    print("\nAfter authorizing, the browser will try to redirect to")
    print("http://localhost:1/... (it will fail to load — that's OK).")
    print("Copy the FULL URL from the address bar and paste it below.\n")

    redirect_response = input("Paste the full redirect URL here: ").strip()

    flow.fetch_token(authorization_response=redirect_response)
    creds = flow.credentials

    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())

    print(f"\nToken saved to {TOKEN_FILE}")
    print(f"Refresh token present: {bool(creds.refresh_token)}")


if __name__ == "__main__":
    main()
