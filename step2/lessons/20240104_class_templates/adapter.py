class Target:
    def request(self):
        ...


class Adaptee:
    def specific_request(self):
        return 'Специфічний запит'


class Adapter(Target):
    def __init__(self, adapteee):
        self._adaptee = adapteee

    def request(self):
        return f'Адаптований: {self._adaptee.specific_request()}'


if __name__ == '__main__':
    adaptee = Adaptee()
    adapter = Adapter(adaptee)

    result = adapter.request()
    print(result)
