import urllib.request
import json

def check(repo, path=""):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "AgriVision"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"\n{repo}/{path}:")
            for item in data:
                print(f"  [{item['type']}] {item['name']}")
    except Exception as e:
        print(f"Error {repo}: {e}")

check("aadium/wheat-disease-detection")
check("BrianGithinji-BMG/Wheat-Disease-and-Pest-Diagnosis-Model")

check("aadium/wheat-disease-detection", "cropDiseaseDataset")

check("aadium/wheat-disease-detection", "cropDiseaseDataset/Wheat_Leaf_Rust")
