def log_function_info(func):
    def wrapper(*args, **kwargs):
        print(f'Виклик функції: {func.__name__}')
        print(f'Параметри: {args} {kwargs}')
    return wrapper


@log_function_info
def example_function(arg1, arg2, kwarg1, kwarg2):
    pass

example_function(1, 2, kwarg1='a', kwarg2='b')