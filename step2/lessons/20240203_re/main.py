import re

text = "Apple and banana are fruits"
string = 'Aa'

matches = re.findall(rf'\b[{string}]\W+', text)
print(matches)
