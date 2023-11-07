"""
practice 2023-11-07
"""

import requests
from requests import HTTPError
import datetime as dt

"""Task 1. Parse PrivatBank Api, write function to see currency
*Add to git file, with flake8, gitignore, requirments.txt

# Task 2. Save currency in file, name of file: currency_today_date_privat
"""

def get_request(url: str) -> dict:
    try:
        response = requests.get(url, verify=False)
        return response.json() if 200 <= response.status_code < 300 else f'error status code: {response.status_code}'
    except HTTPError as http_exc:
        print(f'error http: {http_exc}')
        return {}
    except Exception as exc:
        print(f'global error: {exc}')
        return {}


def save_result(result_json: dict):
    with open('currency_today_date_privat', mode='w') as file:
        for row in result_json:
            dt_today = dt.datetime.today().strftime('%Y-%m-%d')
            file.writelines(f"[{dt_today}]  Currency: {row['ccy']}. Buy: {row['buy']}. Sale: {row['buy']}\n")


PB_API_1 = 'https://api.privatbank.ua/p24api/pubinfo?json&exchange&coursid=5'
result = get_request(PB_API_1)

save_result(result) if result else False

