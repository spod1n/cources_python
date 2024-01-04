class Component:
    def operation(self):
        ...


class ConcreteComponent(Component):
    def operation(self):
        return 'Базова операція'


class Decorator(Component):
    def __init__(self, component):
        self._component = component

    def operation(self):
        return self._component.operation()


class ConcreteDecoratorA(Decorator):
    def operation(self):
        return f'Додаткова операції А, після {self._component.operation()}'


class ConcreteDecoratorB(Decorator):
    def operation(self):
        return f'Додаткова операції Б, після {self._component.operation()}'


if __name__ == '__main__':
    base_component = ConcreteComponent()
    decoreted_component = ConcreteDecoratorA(ConcreteDecoratorB(base_component))

    result = decoreted_component.operation()
    print(result)