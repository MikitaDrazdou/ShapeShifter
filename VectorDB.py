from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv

class VectorDB:
     
    def __init__(self):
        load_dotenv()
        self.client = QdrantClient(
            url=os.getenv("VECTOR_DB_URL"), 
            api_key=os.getenv("VECTOR_DB_API"),
        )

    def createCollection(self, collection_name, vector_size):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    
    def addVector(self, collection_name, embedding, i):
        operation_info = self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(id=i, vector=embedding),
            ],
        )
        print(operation_info)

    def searchNearest(self, collection_name, embedding):
        search_result = self.client.search(
            collection_name=collection_name, query_vector=embedding, limit=2
        )
        return search_result

qdrant = VectorDB()
qdrant.createCollection("vector_collection", 4)
qdrant.addVector("vector_collection", [1,2,3,99], 0)
qdrant.addVector("vector_collection", [1,2,3,100], 1)
search_result = qdrant.searchNearest("vector_collection", [1,2,3,1010])
print(search_result)