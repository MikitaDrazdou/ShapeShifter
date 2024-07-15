from DownloadModel import DownloadModel
from Preprocess import Preprocess
from ImageToEmbed import ImageToEmbed
from KeyValueDB import KeyValueDB
from VectorDB import VectorDB

import time

class PrepareData:
    def __init__(self, uids_path):
        self.uids = self.parse_uids(uids_path)

    def prepare(self, image_folder_path, view_num):
        download_model = DownloadModel("model.zip", "extracted/", "TOKEN.txt")
        preprocess = Preprocess()
        image_converter = ImageToEmbed()

        qdrant = VectorDB()
        qdrant.createCollection("vector_collection", 2048)

        postgre = KeyValueDB()
        postgre.createTable("vector_model")

        image_cnt = 0
        for uid in self.uids:
            time.sleep(5)

            try:
                download_model.download(uid)
            except Exception as e:
                print("Error while downloading occured")

            print("Model downloaded, extracting...")

            try:
                model_temp_path = download_model.extract_from_zip()
            except Exception as e:
                print("Error while extracting occured")

            print("Model extracted, creating shades...")

            try:
                preprocess.prepare(model_temp_path, image_folder_path, image_cnt)
            except Exception as e:
                print("Error while creating shades occured")
            
            print("Shades created, creating embeddings...")
            
            embeddings = []
            try:
                for i in range(view_num):
                    embeddings.append(image_converter.convert(image_folder_path + "/" + str(image_cnt + i) + ".png"))
            except Exception as e:
                print(e)
                print("Error while creating embeddings occured")

            print("Embeddings created, writing to vector database...")

            for i, embedding in enumerate(embeddings):
                qdrant.addVector("vector_collection", embedding, image_cnt + i)

            print("Wrote to vector database, writing to key-value database...")

            for i, embedding in enumerate(embeddings):
                postgre.addImage(table_name="vector_model", embedding=str(embedding), url=str(uid))

            print("All done")
            
            image_cnt += view_num
        
        postgre.cur.close()
        postgre.conn.close()


    def parse_uids(self, uids_path):
        f = open(uids_path, 'r')

        uids = []
        for line in f.readlines():
            uids.append(line[:len(line)-1])

        return uids
    

prepare_data = PrepareData("uids.txt")
prepare_data.prepare('images', 48)
        