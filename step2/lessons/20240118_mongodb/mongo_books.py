from pymongo import MongoClient
import pandas as pd

DB_NAME = 'books_db'

client = MongoClient('mongodb://localhost:27017/')
client.drop_database(DB_NAME) if DB_NAME in client.list_database_names() else False

db = client[DB_NAME]
collection = db['books']

document = [{'name_book': 'Duma Key', 'actor': 'Stephen King', 'release_day': 2008},
            {'name_book': 'Bible', 'actor': '', 'release_day': 850}
            ]

collection.insert_many(document)

db_filter = {'release_day': {'$lt': 1950}}
df = pd.DataFrame(list(collection.find(db_filter)))

print(df)

