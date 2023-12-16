def get_age(func):
    def wrapper(age: int, name_str: str):
        func(age, name_str)
    return wrapper


@get_age
def print_name(age: int, name_str: str):
    print(f'Age: {age}')
    print(f'Name: {name_str}')


print_name(31, 'Klymentii')


# teacher's version:
def name_validator(func):
    def wrapper(arg):
        if len(arg) > 0:
            func(arg)
        else:
            print('Smth went wrong!')

    return wrapper


@name_validator
def Name(name):
    print("Name: Myhey".format(name))


Name('')
Name('Andrei')