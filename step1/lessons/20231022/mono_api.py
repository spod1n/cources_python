import requests


def check_status_code(status_code: int) -> bool:
    match status_code:
        case 200:
            return True
        case 404:
            return False
        case _ as e:
            return False


req = requests.get('https://api.monobank.ua/bank/currency', verify=False)

print(req, req.status_code)

currency = {
    840: 'usd',
    978: 'eur'
}

if check_status_code(req.status_code):
    print('We can work with currency')

    for i in req.json():
        if i['currencyCodeA'] in (840, 978):
            print(f"Code is: {i['currencyCodeA']} Image: {currency[i['currencyCodeA']]} Buy: {i['rateBuy']}")

else:
    print(f'we can\'t work with currency {check_status_code(req)}')
