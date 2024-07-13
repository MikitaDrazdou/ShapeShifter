import requests
import trimesh
import zipfile
import os
import shutil


class DownloadModel:
    def __init__(self, zip_save_path, zip_extract_path, token_path):
        self.zip_save_path = zip_save_path
        self.zip_extract_path = zip_extract_path
        
        self.token = self.retrieve_token(token_path)


    def retrieve_token(self, token_path):
        f = open(token_path, "r")
        return f.read()
    

    def download(self, model_uid):    

        zip_extract_path = self.zip_extract_path
        zip_save_path = self.zip_save_path

        try:
            shutil.rmtree(zip_extract_path)
            os.remove(zip_save_path)
        except Exception as e:
            pass

        zip_save_path = self.zip_save_path

        # Place access token from sketchfab here
        ACCESS_TOKEN = self.token
        UID = model_uid

        response = requests.get(f'https://api.sketchfab.com/v3/models/{UID}/download', headers={'authorization': f'Token {ACCESS_TOKEN}'})

        download_url = ""
        if response.status_code == 200:
            download_info = response.json()
            download_url = download_info['gltf']['url']
        else:
            raise Exception('Error while trying to get download link')

        response = requests.get(download_url, stream=True)

        if response.status_code == 200:
            with open(zip_save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print("Model download finished")
        elif response.status_code == 429:
            raise Exception('API Download rate exceeded')
        else:
            raise Exception('Error while trying to download model')


    def extract_from_zip(self):
        zip_extract_path = self.zip_extract_path
        zip_save_path = self.zip_save_path

        # Extracting from zip archive
        with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
            zip_ref.extractall(zip_extract_path)


        gltf_file = ""
        second_zip = ""
        # Converting to obj format if needed
        for root, dirs, files in os.walk(zip_extract_path):
            for file in files:
                if file.endswith('.zip'):
                    second_zip = os.path.join(root, file)
                    with zipfile.ZipFile(second_zip, 'r') as zip_ref:
                        zip_ref.extractall(zip_extract_path)

        for root, dirs, files in os.walk(zip_extract_path):
            for file in files:
                if file.endswith('.gltf'):
                    gltf_file = os.path.join(root, file)

        '''
        if gltf_file:
            mesh = trimesh.load(gltf_file)
            mesh.export(model_save_path)
        '''
        
        return gltf_file