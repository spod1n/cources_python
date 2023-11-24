"""
Базовий клас Vehicle:
Атрибути: make, model.
Метод: honk(), який виводить "Beep!".
Два проміжних підкласи від Vehicle:
LandVehicle: Клас для наземних транспортних засобів.
Додатковий метод: drive(), який виводить "Їдемо по суші".
WaterVehicle: Клас для водних транспортних засобів.
Додатковий метод: sail(), який виводить "Пливемо по воді".
Підкласи від LandVehicle:
Car: Наслідується від LandVehicle.
Унікальний метод: open_trunk(), який виводить "Багажник відкрито".
Truck: Наслідується від LandVehicle.
Унікальний метод: load_cargo(), який виводить "Вантаж завантажено".
Підкласи від WaterVehicle:
Boat: Наслідується від WaterVehicle.
Унікальний метод: anchor(), який виводить "Човен на якорі".
Ship: Наслідується від WaterVehicle.
Унікальний метод: sound_horn(), який виводить "Гудок корабля".
"""


class Vehicle:
    def __init__(self, make, model):
        self.make = make
        self.model = model

    def honk(self):
        print("Beep!")


class LandVehicle(Vehicle):
    def drive(self):
        print("Go on the ground")


class WaterVehicle(Vehicle):
    def sail(self):
        print("Go on the water")


class Car(LandVehicle):
    def open_trunk(self):
        print("Trunk is open")


class Truck(LandVehicle):
    def load_cargo(self):
        print("The cargo is full")


class Boat(WaterVehicle):
    def anchor(self):
        print("Boat is on anchor")


class Ship(WaterVehicle):
    def sound_horn(self):
        print("There's a sound of horn")


car = Car("Toyota", "BMW")
car.honk()
car.drive()
car.open_trunk()

truck = Truck("Volvo", "Kamaz")
truck.honk()
truck.drive()
truck.load_cargo()

boat = Boat("ffjuyf", "uykfuyfg")
boat.honk()
boat.sail()
boat.anchor()

ship = Ship("vafvbdf", "srtht")
ship.honk()
ship.sail()
ship.sound_horn()