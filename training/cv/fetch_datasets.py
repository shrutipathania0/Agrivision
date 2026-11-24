"""
AgriVision Dataset Ingestion Helper
Finds, downloads, and structures public agricultural datasets for Rice, Wheat, Potato, Tomato, and PlantDoc.
"""

import os
import sys
import json
import urllib.request
from pathlib import Path


def search_github(query, limit=5):
    url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&order=desc"
    req = urllib.request.Request(url, headers={"User-Agent": "AgriVision-Auditor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            results = []
            for item in data.get("items", [])[:limit]:
                results.append({
                    "name": item["full_name"],
                    "url": item["html_url"],
                    "clone_url": item["clone_url"],
                    "stars": item["stargazers_count"],
                    "desc": item.get("description", "")
                })
            return results
    except Exception as e:
        print(f"Error searching github for '{query}': {e}")
        return []


if __name__ == "__main__":
    print("Searching GitHub for Rice leaf disease datasets...")
    rice_repos = search_github("rice leaf disease dataset")
    for r in rice_repos:
        print(f"  - {r['name']} ({r['stars']} stars): {r['clone_url']}")
        
    print("\nSearching GitHub for Wheat disease datasets...")
    wheat_repos = search_github("wheat rust leaf disease dataset")
    for r in wheat_repos:
        print(f"  - {r['name']} ({r['stars']} stars): {r['clone_url']}")
        
    print("\nSearching GitHub for PlantDoc dataset...")
    plantdoc_repos = search_github("PlantDoc-Dataset")
    for r in plantdoc_repos:
        print(f"  - {r['name']} ({r['stars']} stars): {r['clone_url']}")
