class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, value):
        # self.value = value
        if not self.__initialized:
            self.value = value
            self.__initialized = True


singleton1 = Singleton(1)
print(singleton1.value)

singleton2 = Singleton(2)
print(singleton2.value)
