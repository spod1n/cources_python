import re

text = '39801 356 2102 1111 55443'
pattern = r'(\d{3}) (\d{2})'

result = re.search(pattern, text)

print(result.start())
print(result.end())
print(result.groupdict())

print(result)
print(True) if result else print(False)
