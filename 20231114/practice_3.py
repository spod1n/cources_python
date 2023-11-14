'''> 10. Створіть клас Shape для представлення геометричної фігури.
Клас повинен мати атрибути name та sides та методи для виведення інформації про фігуру та обчислення периметру.
Класи Rectangle та Circle повинні бути нащадками класу Shape.'''


class Shape:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f'You shape is {self.name}.'


class Rectangle(Shape):
    def __init__(self, name, width: float, height: float):
        super().__init__(name)
        self.width = width
        self.height = height

    def __str__(self):
        return f"Perimeter '{self.name}': {self.__perimeter()}."

    def __perimeter(self):
        return 2 * (self.width + self.height)


class Circle(Shape):
    def __init__(self, name, radius):
        super().__init__(name)
        self._pi = 3.14159
        self.radius = radius

    def __str__(self):
        return f"Perimeter '{self.name}': {self.__perimeter()}."

    def __perimeter(self):
        return 2 * self._pi * self.radius


shape1 = Shape('Figure1')
shape2 = Rectangle('Figure2', 10, 20)
shape3 = Circle('Figure3', 15)

print(shape1)
print(shape2)
print(shape3)

