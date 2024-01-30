import requests
from unittest.mock import MagicMock, patch
import pytest


def get_weather(city):
    # Функція, яка отримує погоду для вказаного міста з OpenWeatherMap API
    api_key = "33d99a1c99c5ea82e6aaff8592cd6fc3"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def test_get_weather():
    # Підмінюємо функцію requests.get за допомогою MagicMock
    with patch('requests.get') as mock_get:
        # Задаємо значення, яке функція get поверне при виклику
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"temp": 25, "description": "Sunny"}
        mock_get.return_value = mock_response

        # Викликаємо функцію, яка взаємодіє з API
        result = get_weather("Kyiv")

        # Перевірка, чи функція взаємодіє з API правильно
        assert result == {"temp": 25, "description": "Sunny"}
        mock_get.assert_called_once_with("https://api.openweathermap.org/data/2.5/weather?q=Kyiv&appid=your_api_key")


if __name__ == "__main__":
    pytest.main()
