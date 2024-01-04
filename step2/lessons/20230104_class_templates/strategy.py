class Strategy:
    def execute(self):
        ...


class ConcreteStrategy1(Strategy):
    def execute(self):
        return 'Виконання стратегії #1'


class ConcreteStrategy2(Strategy):
    def execute(self):
        return 'Виконання стратегії #2'


class ConcreteStrategy3(Strategy):
    def execute(self):
        return 'Виконання стратегії #3'


class Context:
    def __init__(self, strategy):
        self._strategy = strategy

    def set_strategy(self, strategy):
        self._strategy = strategy

    def execute_operation(self):
        return self._strategy.execute()


if __name__ == '__main__':
    strategy1 = ConcreteStrategy1()
    strategy2 = ConcreteStrategy2()
    strategy3 = ConcreteStrategy3()

    context = Context(strategy1)
    result1 = context.execute_operation()
    print(result1)

    context.set_strategy(strategy2)
    result2 = context.execute_operation()
    print(result2)

    context.set_strategy(strategy3)
    result3 = context.execute_operation()
    print(result3)
