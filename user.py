from ImageToEmbed import ImageToEmbed
from VectorDB import VectorDB
from KeyValueDB import KeyValueDB
from PrepareImage import PrepareImage


prep = PrepareImage()
prep.remove_bg("image.png", "image_no_bg.png")
prep.black_out("image_no_bg.png", "image_black.png")
prep.add_white_background("image_black.png", "image_processed.png")

converter = ImageToEmbed()
embedding = converter.convert("image_processed.png")

vec_db = VectorDB()
result = vec_db.searchNearest("vector_collection", embedding)
#print(result)
result_0 = result[0].vector
result_1 = result[1].vector
result_2 = result[2].vector

query = ""
for num in result_0:
    query += str(num) + " "

sketchfab_prefix = "https://sketchfab.com/3d-models/"

key_db = KeyValueDB()
print(sketchfab_prefix + str(key_db.getUrl("vector_model", query)[0][0]))

query = ""
for num in result_1:
    query += str(num) + " "

print(sketchfab_prefix + str(key_db.getUrl("vector_model", query)[0][0]))

query = ""
for num in result_2:
    query += str(num) + " "

print(sketchfab_prefix + str(key_db.getUrl("vector_model", query)[0][0]))