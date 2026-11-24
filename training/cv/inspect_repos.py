import urllib.request
import json

def list_contents(repo, path=""):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "AgriVision"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, list):
                print(f"\n--- {repo}/{path} ---")
                for item in data:
                    print(f"  [{item['type']}] {item['name']} ({item.get('size', 0)} bytes)")
            elif isinstance(data, dict):
                print(f"File: {data.get('name')}")
    except Exception as e:
        print(f"Error {repo}/{path}: {e}")

list_contents("MHassaanButt/Rice-Disease-Classfication")
list_contents("pamd005/Rice-Leaf-Disease-Detection")
list_contents("haithem000/Wheat-Dataset")
list_contents("Shant-Thakur/YR-22-23")
list_contents("Swetakit/Crop-Disease-prediction")
