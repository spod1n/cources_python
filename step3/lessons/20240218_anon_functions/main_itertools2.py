import itertools

# data = [1, 2, 3]
data = range(0, 10)
print(data, type(data))

data = itertools.cycle(data)
print(data, type(data))
print(next(data), type(next(data)))
print(next(data), type(next(data)))
print(next(data), type(next(data)))

