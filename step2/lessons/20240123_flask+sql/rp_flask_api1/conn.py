import sqlite3

conn = sqlite3.connect('people.db')
columns = [
    'id INTEGER PRIMARY KEY',
    'lname VARCHAR UNIQUE',
    'fname VARCHAR',
    'timestamp DATETIME'
]

create_table_cmd = f"CREATE TABLE person ({','.join(columns)})"
# conn.execute(create_table_cmd)

people = [
    "1, 'Fairy', 'Tooth', '2024-01-25 19:30:15'",
    "2, 'Ruprecht', 'Knelt', '2024-01-25 19:30:15'",
    "3, 'Bunny', 'Easter', '2024-01-25 19:30:15'"
]

for person_data in people:
    insert_cmd = f'INSERT INTO person VALUES ({person_data})'
    conn.execute(insert_cmd)
    conn.commit()

conn.close()
