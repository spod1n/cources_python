
class Figure:
    def __init__(self, name: str):
        self.name = name

    def area(self):
        ...
    def perimeter(self):
        ...
    def __str__(self):
        return True


class Circle(Figure):
    def __init__(self, name, width: float, height: float):
        self._pi = 3.14159


class Rectangle(Shape):
    def __init__(self, name, width: float, height: float):
        super().__init__(name)
        self.width = width
        self.height = height

    def __str__(self):
        return f"Perimeter '{self.name}': {self._perimeter()}."

    def _area(self):
        return self.width + self.height

    def _perimeter(self):
        return 2 * (self.width + self.height)


class Triangle(Figure):
    ...
