from functools import reduce

'''6. Використайте reduce() для обчислення суми всіх елементів у списку: [1, 2, 3, 4, 5] → 15 (1 + 2 + 3 + 4 + 5).'''
arr = [a for a in range(1, 6)]
num2 = reduce(lambda a, b: a + b, arr)
print(num2)

'''7. Використайте map() для перетворення списку зі рядок на список їхніх довжин: ['apple', 'banana', 'cherry'] → [5, 6, 6].'''
arr = ['apple', 'banana', 'cherry']

arr2 = list(map(lambda x: len(x), arr))
print(arr2)

'''8. Використайте filter() для знаходження всіх елементів списку, які більше за середнє значення списку: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] → [6, 7, 8, 9, 10].'''
arr = [a for a in range(1, 11)]

arr2 = list(filter(lambda x: x > (sum(arr) / len(arr)), arr))
print(arr2)

'''9. Використайте reduce() для обчислення найбільшого елемента у списку: [1, 5, 3, 9, 2] → 9.'''
arr = [1, 5, 3, 9, 2]
num2 = reduce(lambda a, b: max(a, b), arr)
print(num2)

'''10. Використайте map() для перетворення списку зі рядок на список великих літер: ['apple', 'banana', 'cherry'] → ['APPLE', 'BANANA', 'CHERRY'].'''
arr = ['apple', 'banana', 'cherry']

arr2 = list(map(lambda x: x.upper(), arr))
print(arr2)