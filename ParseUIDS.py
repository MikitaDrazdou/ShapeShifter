import requests
import time

class ParseUIDS:
    def __init__(self, uid_path):
        self.uid_path = uid_path


    def fetch_uids(self, BASE_URL, HEADERS, params):
        uids = []
        
        next_url = BASE_URL
        while len(uids) < 5000 and next_url:
            response = requests.get(next_url, headers=HEADERS, params=params if next_url == BASE_URL else None)
            if response.status_code == 200:
                data = response.json()
                models = data.get('results', [])
                
                for model in models:
                    if len(uids) >= 5000:
                        break
                    uid = model['uid']
                    uids.append(uid)
                
                # Get the next URL for pagination
                next_url = data.get('next')
                if next_url:
                    print(f'Fetching next page: {next_url}')
            else:
                print(f'Error fetching models: {response.status_code}')
                break

            # Respect API rate limits
            time.sleep(1)  # Adjust the sleep time based on API rate limits

        return uids

    def parse(self):

        # Replace 'YOUR_API_TOKEN' with your actual Sketchfab API token
        f = open("TOKEN.txt", 'r')
        API_TOKEN = f.read()
        HEADERS = {
            'Authorization': f'Token {API_TOKEN}'
        }

        BASE_URL = 'https://api.sketchfab.com/v3/models'

        # Define the parameters for the request
        params = {
            'downloadable': 'true',  # Filter to only include downloadable models
            'license': 'CC0,CC-BY,CC-BY-SA',  # Filter to include only free models (with permissive licenses)
            'sort_by': '-likeCount',  # Optional: Sort by most liked models
            'type': 'models',
            'q': ''  # Optional: Add any specific search query
        }

        # Fetch the first 5000 UIDs
        uids = self.fetch_uids(BASE_URL, HEADERS, params)

        f = open(self.uid_path, 'w')
        for uid in uids:
            f.write(f"{uid}\n")

parser = ParseUIDS("uids.txt")
parser.parse()