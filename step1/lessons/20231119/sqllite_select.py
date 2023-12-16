import sqlite3

conn = sqlite3.connect('example.db')

# Task 1. Знайти всіх людей у кого на 3 позиції є літера f
cursor = conn.execute("SELECT id, name FROM users WHERE name LIKE '__f%'")
print(cursor.fetchall())

# Task 2. Знайти всіх людей у кого всього 14 символів імя
cursor = conn.execute("SELECT id, name FROM users WHERE LENGTH(name) = 14")
data = cursor.fetchall()
print(data) if cursor else print('Null rows result')

# Task 3. Знайти всіх людей у кого id < 50
cursor = conn.execute("SELECT id, name FROM users WHERE id < 50")
print(cursor.fetchall())

conn.commit()
conn.close()


