import requests
import zipfile
import os
import shutil


class DownloadModel:
    def __init__(self, zip_save_path, zip_extract_path, token_path):
        self.zip_save_path = zip_save_path
        self.zip_extract_path = zip_extract_path
        
        self.token = self.retrieve_token(token_path)
        self.TOKENS = ["445f5c25521d4043ba342377deede8e7", "34f44f5b82be4f69a63d1962f3eadcf6", "d2f2dbdb7dbe4710bf75cb2a4f8db948", "1e96343599c34cba8bf72a06d0609c1d", "ab8152088a214b14a7009cc647c562de", 
                       "74bb2ed762684ad29ad691d616a4d0e8", "bfe2b124c6a1430c99fe2b87cde5b55d", "9211e965134a4bd786b35db2cf576f5e", "c0b8bc69938a48028ae47742fdad4b2b", "792350cdc7814e76882f27d02061b9e9",
                       "5d23d13ec9c545c4b065bb8ad2966523", "76a3f9d50bf241479466240f5a9be957", "9eefa83fa461425b931d9baedc5398db", "c3436e858349499a87e27f574f633cf1", "68b3e30df1714a2d85e819c8d232ac16",
                       "20762906e88d4d3aa26424f0addb33b5", "e15113337ac04c8a94285d7cb5f66835", "b4051854270b489ba27faeb6fe5a91a1", "1a4d3666d1b945b9ab9cf8b4f3ea267f", "5f929e88def34e4dbdb9427d2b05deb0"
                       ]
        self.token = self.TOKENS[0]


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
        elif response.status_code == 429:
            self.token = self.TOKENS[(self.TOKENS.index(self.token) + 1) % len(self.TOKENS)]
            raise Exception('API Download rate exceeded')
        else:
            raise Exception('Error while trying to get download link')

        response = requests.get(download_url, stream=True)

        if response.status_code == 200:
            with open(zip_save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        elif response.status_code == 429:
            self.token = self.TOKENS[(self.TOKENS.index(self.token) + 1) % len(self.TOKENS)]
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
        
        return gltf_file