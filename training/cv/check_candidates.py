import urllib.request
import json

repos = [
    'priyanshu60-pc/Plant-Disease-Detection-CNN',
    'praneethreddymandha/PlantDiseaseDetector',
    'Swetakit/Crop-Disease-prediction',
    'MHassaanButt/Rice-Disease-Classfication',
    'pamd005/Rice-Leaf-Disease-Detection',
    'haithem000/Wheat-Dataset',
    'Shant-Thakur/YR-22-23'
]

for r in repos:
    req = urllib.request.Request(f'https://api.github.com/repos/{r}', headers={'User-Agent': 'AgriVision'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            size_mb = data.get("size", 0) / 1024
            print(f"{r}: Size = {size_mb:.2f} MB | Description: {data.get('description', '')[:60]}")
    except Exception as e:
        print(f"{r}: Error {e}")
