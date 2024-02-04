import re

text = input('Введіть слово із 5 букв: ')
pattern = '^a...s$'
result = re.match(pattern, text)

print(result)
print(True) if result else print(False)
