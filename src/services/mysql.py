import os
import mysql.connector
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class MySQLService:
    def __init__(self) -> None:
        self.host = os.getenv("HOST")
        self.database = os.getenv("DATABASE")
        self.user = os.getenv("USER")
        self.password = os.getenv("PASSWORD")

    def connect_to_db(self):
        return mysql.connector.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password,
        )

    def execute_query(self, query, read_only=True):
        resp = None
        try:
            db = self.connect_to_db()
            if read_only:
                resp = pd.read_sql_query(query, db)
            else:
                mycursor = db.cursor()
                mycursor.execute(query)

                db.commit()
            db.close()
        except Exception as e:
            print(e)
        return resp
