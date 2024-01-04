class Observer:
    def update(self, message):
        ...


class ConcreteObserver(Observer):
    def update(self, message):
        print(f'Отримано повідомлення: {message}')


class Subject:
    def __init__(self):
        self._observers = []

    def add_observer(self, observer):
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def notify_observers(self, message):
        for observer in self._observers:
            observer.update(message)


if __name__ == '__main__':
    subject = Subject()
    observer1 = ConcreteObserver()
    observer2 = ConcreteObserver()

    subject.add_observer(observer1)
    subject.add_observer(observer2)

    subject.notify_observers('Нове повідомлення для спостерігачів.')
