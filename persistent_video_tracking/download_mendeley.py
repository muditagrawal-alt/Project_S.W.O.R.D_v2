import requests
import json
import os
import cv2

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

def fetch_mendeley_dataset():
    # Attempting to fetch the dataset metadata
    url = "https://data.mendeley.com/api/datasets/bbzpxhd22j/2"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('files', [])
        else:
            print(f"Error fetching metadata: {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception: {e}")
        return []

if __name__ == "__main__":
    files = fetch_mendeley_dataset()
    print(f"Found {len(files)} files in the dataset.")
    if files:
        print(json.dumps(files[:5], indent=2))
