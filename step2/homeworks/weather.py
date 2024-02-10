import requests


def get_weather(url: str, params: dict) -> float:
    """ Функція для отримання погодних умов вказаного міста з OpenWeatherMap API.

    :param url: URL-адреса API ля отримання погоди
    :param params: параметри місця отримання погоди (назва міста та ключ API)
    :return: температура на поточний момент в місті за Фаренгейтом
    """
    response = requests.get(url, params=params)
    return response.json()['main']['temp'] if response.status_code == 200 else None


if __name__ == '__main__':
    # перевіряю відповідь API
    result = get_weather('http://api.openweathermap.org/data/2.5/weather',
                         {'q': 'Kyiv', 'appid': 'API_KEY'}
                         )
    print(type(result), result, sep=', ')
