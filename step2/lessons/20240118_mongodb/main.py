from pymongo import MongoClient

db_name = 'my_db_test'

client = MongoClient('mongodb://localhost:27017/')

client.drop_database(db_name) if db_name in client.list_database_names() else False

db = client['my_db_test']
collection = db['my_db_collection']

document = [{'first_name': 'John', 'last_name': 'Doe', 'age': 30, 'film': 'Inception', 'role': 'Architect'},
            {'first_name': 'Olivia', 'last_name': 'Smith', 'age': 30, 'film': 'The Matrix', 'role': 'Neo'},
            {'first_name': 'Isabella', 'last_name': 'Wilson', 'age': 30, 'film': 'The Godfather', 'role': 'Michael Corleone'},]

result = collection.insert_many(document)

# collection.update_one({'name': 'John Doe'}, {'$set': {'age': 31}})

result = collection.find()

for document in result:
    del document['_id']

    docs = ''
    for doc in document.values():
        docs += str(doc) + ', '

    print(docs)
client.close()
