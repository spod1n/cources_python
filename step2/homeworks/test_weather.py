import unittest
from unittest.mock import patch

from weather import get_weather

# constants
URL_BASE = 'http://api.openweathermap.org/'
URL_SUFFIX = 'data/2.5/weather'

PARAMS = {'q': 'Kyiv',
          'appid': 'API_KEY'
          }


class TestWeatherAPI(unittest.TestCase):
    @patch('weather.requests.get')
    def test_get_weather(self, mock_get: unittest.mock.Mock) -> None:
        """ Функція для тестування взаємодії з API

        :param mock_get: штучний об'єкт бібліотеки unittest
        :return: None
        """
        # створюю штучний об'єкт (мок) відповіді
        response_mock = unittest.mock.Mock()

        # налаштовую поведінку мока запиту
        mock_get.return_value = response_mock

        # викликаю функцію отримання погоди з окремого модулю
        result = get_weather(url=URL_BASE + URL_SUFFIX, params=PARAMS)

        # перевіряю чи функція повертає очікуване значення
        if response_mock.status_code == 200:
            self.assertEqual(result, response_mock.json()['main']['temp'])
        else:
            self.assertIsNone(result)

        # перевіряю чи функція взаємодіє з API правильно
        mock_get.assert_called_once_with(URL_BASE + URL_SUFFIX, params=PARAMS)


if __name__ == '__main__':
    # викликаю юніттест
    unittest.main()
