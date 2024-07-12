from DownloadModel import DownloadModel
from Preprocess import Preprocess

import time

class PrepareData:
    def __init__(self, uids_path):
        self.uids = self.parse_uids(uids_path)

    def prepare(self, temp_model_path, image_folder_path, view_num):
        download_model = DownloadModel("model.zip", "extracted/", "TOKEN.txt")
        preprocess = Preprocess()

        image_cnt = 0
        for uid in self.uids:
            time.sleep(2)
            download_model.download(uid)
            download_model.extract_from_zip(temp_model_path)
            print("Model extracted")
            preprocess.prepare(temp_model_path, image_folder_path, image_cnt)
            print("Shades created")
            image_cnt += view_num


    def parse_uids(self, uids_path):
        f = open(uids_path, 'r')

        uids = []
        for line in f.readlines():
            uids.append(line[:len(line)-1])

        return uids
    

prepare_data = PrepareData("uids.txt")
prepare_data.prepare('model.obj', 'images', 48)
        