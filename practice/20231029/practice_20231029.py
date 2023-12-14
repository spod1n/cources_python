'''2. Використайте filter() для знаходження всіх парних чисел у списку: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] → [2, 4, 6, 8, 10].'''
arr = [a for a in range(1, 11)]

arr2 = list(filter(lambda x: x % 2 == 0, arr))
print(arr2)

'''3. Використайте reduce() для обчислення добутку всіх елементів у списку: [1, 2, 3, 4, 5] → 120 (1 * 2 * 3 * 4 * 5).'''
from functools import reduce

arr = [a for a in range(1, 6)]
num2 = reduce(lambda a, b: a * b, arr)
print(num2)

'''4. Використайте map() для перетворення списку зі стрічок на список квадратів цих чисел: [1, 2, 3, 4, 5] → [1, 4, 9, 16, 25].'''
arr = [a for a in range(1, 6)]

arr2 = list(map(lambda x: pow(x, 2), arr))
print(arr2)

'''5. Використайте filter() для знаходження всіх додатних чисел у списку: [1, -2, 3, -4, 5, -6, 7, -8, 9, -10] → [1, 3, 5, 7, 9].'''
arr = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]

arr2 = list(filter(lambda x: x > 0, arr))
print(arr2)