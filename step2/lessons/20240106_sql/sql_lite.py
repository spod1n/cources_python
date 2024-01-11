import sqlite3

conn = sqlite3.connect('mydatabase.db')

cur = conn.cursor()

cur.execute('CREATE TABLE IF NOT EXISTS mytable (id INTEGER PRIMARY KEY, name TEXT)')

cur.execute("INSERT INTO mytable (name) VALUES (?)", ('Jonh',))

cur.execute('SELECT * FROM mytable')
result = cur.fetchall()
print(result)

conn.commit()

conn.close()
