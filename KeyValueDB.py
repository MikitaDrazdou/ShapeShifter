import psycopg2
import os
from dotenv import load_dotenv

class KeyValueDB:
     
    def __init__(self):
        load_dotenv()
        self.conn = psycopg2.connect(
                        database = "search_project", 
                        user = "postgres", 
                        host= 'localhost',
                        password = os.getenv("KEYVALUE_DB_PASSWORD"),
                        port = 5432
                    )
        # Open a cursor to perform database operations
        self.cur = self.conn.cursor()

    def createTable(self, table_name):
        self.cur.execute(f"""CREATE TABLE {table_name}(
                vector TEXT PRIMARY KEY,
                url TEXT NOT NULL);""")
        self.conn.commit()
        
    def addImage(self, table_name, embedding, url):
        self.cur.execute(f"""INSERT INTO 
                            {table_name}(vector, url) 
                            VALUES('{embedding}','{url}');""")
        self.conn.commit()

    def getUrl(self, table_name, embedding):
        self.cur.execute(f"""SELECT url
                            FROM {table_name}
                            WHERE vector = '{embedding}';""")
        search_result = self.cur.fetchall() 
        return search_result