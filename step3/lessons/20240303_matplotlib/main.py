import requests
import pandas as pd
from bs4 import BeautifulSoup

url = {
    'base': 'https://finance.i.ua/',
    'suffix': ['usd', 'eur', 'pln']
}

response = requests.get(url['base'] + 'eur', verify=False)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table', {'id': 'latest_currency_container'})
# Якщо ID не знайдено, можна використовувати щось на зразок:
# table = soup.find('table', class_='table table-data tablesorter tablesorter-default')

# Парсинг таблиці і створення списку списків
data = []
for row in table.find_all('tr'):
    row_data = [cell.text.strip() for cell in row.find_all(['th', 'td'])]
    data.append(row_data)

# Видалення пустих списків
data = [row for row in data if row]

# Створення датафрейму
columns = data[0]
df = pd.DataFrame(data[1:], columns=columns)

print(df.to_markdown())
