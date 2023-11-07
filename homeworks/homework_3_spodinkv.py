passing_range = [0, 100]    # діапазон оцінок
students = {}               # словник студентів

# кількість студентів на іспиті визначає професор Грубл
students_count = int(input('Введіть кількість студентів, які здавали екзамен: ').strip())

# при кожній ітерації вводжу оцінку і позначку
for student in range(1, students_count + 1):
    flag_result = True      # прапорець - чи коректно професор Грубл ввів значення

    while flag_result:      # цикл, якщо значення ведено некоректно
        student_rating = input(f'Введіть оцінку іспиту та позначку студента #{student}: ').split(' ')

        if len(student_rating) == 2:    # перевірка, чи два значення введено
            student_value = int(student_rating[0].strip())
            student_mark = student_rating[1].capitalize().strip()

            if student_mark in ['Passed', 'Failed'] and 0 < student_value <= 100:   # перевірка на адекватність даних
                flag_result = False     # змінюю прапорець, якщо все ок
                students[f'Student {student}'] = (student_value, student_mark)      # записую дані в словник з кортежом
            else:
                print('Помилка! Введіть оцінку від 0 до 100 та позначку "Passed" або "Failed" через пробіл.')
        else:
            print('Помилка! Дані введено некоректно.')

# ітерую отримані дані з клавіатури
for exam_result in students.items():
    value, mark = exam_result[1]                            # розпаковка кортежа

    if mark == 'Passed' and value < passing_range[1]:       # знаходжу мін. оцінку при позначці passed
        passing_range[1] = value
    elif mark == 'Failed' and value > passing_range[0]:     # знаходжу макс. оцінку при позначці failed
        passing_range[0] = value

    # перевіряю чи макс оцінка з позначкою failed не вища за мін. оцінку з позначкою passed
    if passing_range[0] >= passing_range[1]:
        print(f'Професор Грубл був непослідовним.')         # якщо так - повідомляю юзера і виходжу з циклу
        break
# якщо ітерація закінчилася не через оператор break
else:
    print('Професор Грубл був послідовним. '
          f'Поріг складання іспиту в діапазоні {passing_range[0] + 1} - {passing_range[1]}')
