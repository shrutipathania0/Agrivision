import urllib.request
import json

url = "https://api.github.com/search/repositories?q=wheat+disease+images&sort=stars"
req = urllib.request.Request(url, headers={"User-Agent": "AgriVision"})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    for item in data.get("items", [])[:8]:
        size_mb = item.get("size", 0) / 1024
        print(f"{item['full_name']} ({item['stargazers_count']} stars) - {size_mb:.2f} MB: {item['html_url']}")
