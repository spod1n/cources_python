""" Klymnetii Spodin. Homework numpy """

import numpy as np
from PIL import Image

""" Створення масиву та операції """
# Створіть два одновимірних масиви розміром 5×5 кожен, заповніть їх випадковими числами.
array1 = np.random.randint(0, 10, size=(5, 5))
array2 = np.random.randint(0, 10, size=(5, 5))
print('Масив 1:', array1, sep='\n', end='\n\n')
print('Масив 2:', array2, sep='\n', end='\n\n')

# Потім виконайте наступні операції та виведіть результат:
# Сума елементів кожного масиву.
sum_array1 = np.sum(array1)
sum_array2 = np.sum(array2)
print('Сума елементів масиву 1:', sum_array1, sep=' ', end='\n\n')
print('Сума елементів масиву 2:', sum_array2, sep=' ', end='\n\n')

# Різниця між елементами двох масивів.
diff_arrays = array1 - array2
print('Різниця між елементами двох масивів:', diff_arrays, sep='\n', end='\n\n')

# Перемноження елементів двох масивів.
prod_arrays = array1 * array2
print('Перемноження елементів двох масивів:', prod_arrays, sep='\n', end='\n\n')

# Підняття кожного елемента першого масиву до ступеня відповідного елемента другого масиву.
power_arrays = np.power(array1, array2)
print('Підняття елементів масиву 1 до ступеня елементів масиву 2:', power_arrays, sep='\n', end='\n\n')

"""Індексація та вибірка даних"""
# Створіть двовимірний масив 8×8, який відображає шахову дошку (1 – біла клітина, 0 – чорна клітина).
chessboard = np.zeros(shape=(8, 8), dtype=int)
chessboard[1::2, ::2] = 1
chessboard[::2, 1::2] = 1
print('Шахова дошка:', chessboard, sep='\n', end='\n\n')

# Використовуйте індексацію для виведення на екран:

# Рядок, представляючи 3-й рядок з дошки.
print('3-й рядок з дошки:', chessboard[2, :], sep='\n', end='\n\n')

# Стовпець, представляючи 5-й стовпець з дошки.
print('5-й стовпець з дошки:', chessboard[:, 4].reshape((-1, 1)), sep='\n', end='\n\n')

# Підмасив, представляючи частину дошки розміром 3×3 в лівому верхньому куті.
print('Підмасив 3x3 в лівому верхньому куті:', chessboard[:3, :3], sep='\n', end='\n\n')

"""Статистика та робота зі зображеннями"""
# Завантажте зображення (наприклад, за допомогою бібліотеки Pillow).
image = Image.open('win_logo.jpg')

# Перетворіть його в тривимірний NumPy-масив та виведіть інформацію про розмір та тип даних.
image_np = np.array(image)

# Проведіть статистичний аналіз:
# Знайдіть середнє значення, мінімум та максимум для кожного каналу зображення (R, G, B).
for i, channel in enumerate(('R', 'G', 'B')):
    mean_channel = np.mean(image_np[:, :, i])
    min_channel = np.min(image_np[:, :, i])
    max_channel = np.max(image_np[:, :, i])

    print(f"Канал зображення '{channel}':",
          f'{mean_channel=}', f'{min_channel=}', f'{max_channel=}', sep='\n', end='\n\n')

# Підрахуйте загальну суму інтенсивності пікселів та виведіть її.
total_intensity_sum = np.sum(image_np)
print('Загальна сума інтенсивності пікселів: ', total_intensity_sum, end='\n\n')

# Виконайте нормалізацію зображення, розділивши значення кожного пікселя на максимальне значення.
normalized_image = image_np / 255.0
