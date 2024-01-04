class State:
    def handle(self):
        ...


class ConcreteState1(State):
    def handle(self):
        return 'Виконання операції для стану #1'


class ConcreteState2(State):
    def handle(self):
        return 'Виконання операції для стану #2'


class Context:
    def __init__(self, state):
        self._state = state

    def set_state(self, state):
        self._state = state

    def request(self):
        return self._state.handle()


if __name__ == '__main__':
    state1 = ConcreteState1()
    state2 = ConcreteState2()

    context = Context(state1)
    result1 = context.request()
    print(result1)

    context.set_state(state2)
    result2 = context.request()
    print(result2)
