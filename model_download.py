import requests
import trimesh
import zipfile
import os
import shutil

zip_save_path = "test_dwn.zip"
extract_path = "zip_extracted"
model_save_path = "test_dwn.obj"

# Place access token from sketchfab here
ACCESS_TOKEN = ""
UID = "129bba873f8140d99c04b4c205b4fc7f"

response = requests.get(f'https://api.sketchfab.com/v3/models/{UID}/download', headers={'authorization': f'Token {ACCESS_TOKEN}'})

shutil.rmtree(extract_path)

download_url = ""
if response.status_code == 200:
    download_info = response.json()
    download_url = download_info['source']['url']

response = requests.get(download_url, stream=True)

if response.status_code == 200:
    with open(zip_save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("Model download finished")

# Extracting from zip archive
with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


gltf_file = ""
second_zip = ""
# Converting to obj format if needed
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith('.gltf'):
            gltf_file = os.path.join(root, file)
            break
        if file.endswith('.zip'):
            second_zip = os.path.join(root, file)
            break

if gltf_file:
    mesh = trimesh.load(gltf_file)
    mesh.export(model_save_path)
elif second_zip:
    with zipfile.ZipFile(second_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.gltf'):
                gltf_file = os.path.join(root, file)
                break
