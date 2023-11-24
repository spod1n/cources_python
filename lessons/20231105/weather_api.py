# Weather App
import requests

API_KEY = "33d99a1c99c5ea82e6aaff8592cd6fc3"

city = input('Enteryou city: ').strip().capitalize()

url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
response = requests.get(url, verify=False)

if response.status_code == 200:
    data = response.json()
    print(data, data.keys())

    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    description = data["weather"][0]["description"]
    wind = data

    print(f"Погода в місті {city}: {description}")
    print(f"Температура: {temp}°F, відчувається як {feels_like}°F")
else:
    print("Не вдалося отримати дані про погоду")
