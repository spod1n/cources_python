from pymongo import MongoClient
import pandas as pd

DB_NAME = 'books2_db'

client = MongoClient('mongodb://localhost:27017/')
client.drop_database(DB_NAME) if DB_NAME in client.list_database_names() else False

db = client[DB_NAME]
collection = db['books']

doc = [
    {'name_book': 'The Catcher in the Rye', 'actor': 'J.D. Salinger', 'release_day': 1951, 'country': 'USA'},
    {'name_book': 'One Hundred Years of Solitude', 'actor': 'Gabriel Garcia Marquez', 'release_day': 1967, 'country': 'Colombia'},
    {'name_book': 'To Kill a Mockingbird', 'actor': 'Harper Lee', 'release_day': 1960, 'country': 'USA'},
    {'name_book': '1984', 'actor': 'George Orwell', 'release_day': 1949, 'country': 'United Kingdom'},
    {'name_book': 'The Great Gatsby', 'actor': 'F. Scott Fitzgerald', 'release_day': 1925, 'country': 'USA'},
    {'name_book': 'War and Peace', 'actor': 'Leo Tolstoy', 'release_day': 1869, 'country': 'Russia'},
    {'name_book': 'Brave New World', 'actor': 'Aldous Huxley', 'release_day': 1932, 'country': 'United Kingdom'},
    {'name_book': 'The Hobbit', 'actor': 'J.R.R. Tolkien', 'release_day': 1937, 'country': 'United Kingdom'},
    {'name_book': 'Pride and Prejudice', 'actor': 'Jane Austen', 'release_day': 1813, 'country': 'United Kingdom'},
    {'name_book': 'The Lord of the Rings', 'actor': 'J.R.R. Tolkien', 'release_day': 1954, 'country': 'United Kingdom'},
    {'name_book': 'Crime and Punishment', 'actor': 'Fyodor Dostoevsky', 'release_day': 1866, 'country': 'Russia'},
    {'name_book': 'The Alchemist', 'actor': 'Paulo Coelho', 'release_day': 1988, 'country': 'Brazil'},
    {'name_book': 'Moby-Dick', 'actor': 'Herman Melville', 'release_day': 1851, 'country': 'USA'},
    {'name_book': 'The Odyssey', 'actor': 'Homer', 'release_day': -720, 'country': 'Greece'},
    {'name_book': 'Frankenstein', 'actor': 'Mary Shelley', 'release_day': 1818, 'country': 'United Kingdom'},
    {'name_book': "Alice's Adventures in Wonderland", 'actor': 'Lewis Carroll', 'release_day': 1865, 'country': 'United Kingdom'},
    {'name_book': 'The Road', 'actor': 'Cormac McCarthy', 'release_day': 2006, 'country': 'USA'},
    {'name_book': 'The Brothers Karamazov', 'actor': 'Fyodor Dostoevsky', 'release_day': 1880, 'country': 'Russia'},
    {'name_book': 'The Count of Monte Cristo', 'actor': 'Alexandre Dumas', 'release_day': 1844, 'country': 'France'},
    {'name_book': 'The Picture of Dorian Gray', 'actor': 'Oscar Wilde', 'release_day': 1890, 'country': 'United Kingdom'}
]

collection.insert_many(doc)

db_filter = {'release_day': {'$eq': 1951}}

result = collection.find(db_filter)

for res in result:
    print(list(res.values())[1:])
