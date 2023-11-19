import sqlite3

conn = sqlite3.connect('example.db')

# retrieve data from the table
cursor = conn.execute("SELECT id, name FROM users")
cursor = cursor.fetchall()

print(cursor)
