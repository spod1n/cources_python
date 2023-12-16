import time

def name_validator(func):
    cache = {}
    def wrapper(arg):
        if arg not in cache:
            result = func(arg)
            cache[arg] = result
        else:
            print('Taken from cache:', cache[arg])

    return wrapper


@name_validator
def calc(num):
    var = num * 100
    time.sleep(1)
    print(var)
    return var


calc(10)
calc(10)