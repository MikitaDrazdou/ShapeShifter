from DownloadModel import DownloadModel
from Preprocess import Preprocess

import time

class PrepareData:
    def __init__(self, uids_path):
        self.uids = self.parse_uids(uids_path)

    def prepare(self, image_folder_path, view_num):
        download_model = DownloadModel("model.zip", "extracted/", "TOKEN.txt")
        preprocess = Preprocess()

        image_cnt = 0
        for uid in self.uids:
            time.sleep(2)

            try:
                download_model.download(uid)
            except Exception as e:
                print("Error while downloading occured")

            try:
                model_temp_path = download_model.extract_from_zip()
            except Exception as e:
                print("Error while extracting occured")

            print("Model extracted")

            try:
                preprocess.prepare(model_temp_path, image_folder_path, image_cnt)
            except Exception as e:
                print("Error while creating shades occured")
                
            print("Shades created")
            image_cnt += view_num


    def parse_uids(self, uids_path):
        f = open(uids_path, 'r')

        uids = []
        for line in f.readlines():
            uids.append(line[:len(line)-1])

        return uids
    

prepare_data = PrepareData("uids.txt")
prepare_data.prepare('images', 48)
        