# Connect with BD
import sqlite3
from faker import Faker

# create a connection to the database
conn = sqlite3.connect('example.db')

print(conn)
# create a table
conn.execute('''CREATE TABLE IF NOT EXISTS
             users (id INTEGER PRIMARY KEY, name TEXT)''')

fake = Faker()
Faker.seed(42)

for i in range(1, 120):
    conn.execute("INSERT INTO users (id, name) VALUES (?, ?)", (i, fake.name()))


# commit the changes
conn.commit()
# close the connection
conn.close()




