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
    
    def execute_multiple_queries(self, queries):
        try:
            db = self.connect_to_db()
            mycursor = db.cursor()
            for query in queries:
                mycursor.execute(query)

            db.commit()
            db.close()
        except Exception as e:
            print(e)

    def get_data(self, table_name, columns=None, where_clause=None, order_by_clause=None):
        try:
            # If columns are not specified, fetch all columns (*)
            columns_str = "*" if columns is None else ", ".join(columns)

            # Build the SQL query with the optional WHERE clause
            query = f"SELECT {columns_str} FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"

            if order_by_clause:
                query += f" ORDER BY {order_by_clause}"

            df = self.execute_query(query)

            return df
        except mysql.connector.Error as e:
            print(f"Error fetching data: {e}")
            return None
