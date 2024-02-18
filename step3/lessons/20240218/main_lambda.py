# example:
#
# people = [('Аnna', 30), ('Ivan', 25), ('Mariya', 35), ('Petro', 28)]
# sorted_people = sorted(people, key=lambda x: x[1])
# print(sorted_people)

# data = (('a', 121, 'abc'), ('c', 0, 'bca'), ('b', 12, 'ccc'))
# d = sorted(data, key=lambda x: x[1], reverse=True)
# print(data)
# print(d)

mountains = {'Еверест': 8848, 'Канченджунга': 8586, 'Лхоцзе': 8516, 'Макалу': 8485, 'Чо-Ойю': 8201, 'Даулаґірі': 8167,
             'Манаслу': 8163, 'Нанга Парбат': 8126, 'Анапурна': 8091, 'Гашербрум I': 8080}

mountains_height_sort = sorted(mountains, key=lambda x: x[1], reverse=True)
mountains_len_sort = sorted(mountains, key=lambda x: len(x[0]))

print('Height mountains: ', mountains_height_sort)
print('Len name mountains: ', mountains_len_sort)