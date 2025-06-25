# src/ingest.py

import requests
from bs4 import BeautifulSoup
import os

def fetch_uscis_text(url):
    """Fetch and clean text from a given USCIS webpage URL"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    return text

def save_to_file(content, filepath="data/uscis_content.txt"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    # Example USCIS page: All forms index
    url = "https://www.uscis.gov/forms/all-forms"
    print(f"Fetching content from {url}")
    content = fetch_uscis_text(url)
    save_to_file(content)
    print("Content saved to data/uscis_content.txt")
