"""
Search and inspect GitHub repositories for verified Rice and Wheat datasets
"""

import urllib.request
import urllib.parse
import json

def search_repo_code(query):
    url = f"https://api.github.com/search/code?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AgriVision-Auditor/1.0", "Accept": "application/vnd.github.v3+json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            print(f"Results for '{query}': {data.get('total_count', 0)}")
            for item in data.get("items", [])[:5]:
                repo = item["repository"]["full_name"]
                path = item["path"]
                print(f"  Repo: {repo} | File: {path}")
    except Exception as e:
        print(f"Error for '{query}': {e}")


def search_repo_names(query):
    url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AgriVision-Auditor/1.0", "Accept": "application/vnd.github.v3+json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            print(f"Repo search for '{query}': {data.get('total_count', 0)}")
            for item in data.get("items", [])[:5]:
                print(f"  {item['full_name']} ({item['stargazers_count']} stars) - Size: {item['size']} KB | URL: {item['html_url']}")
    except Exception as e:
        print(f"Error for '{query}': {e}")


if __name__ == "__main__":
    print("--- Searching for Rice Leaf Disease Datasets ---")
    search_repo_names("rice leaf diseases dataset")
    search_repo_names("rice leaf blast brown spot")

    print("\n--- Searching for Wheat Disease Datasets ---")
    search_repo_names("wheat rust dataset")
    search_repo_names("wheat leaf disease dataset")
