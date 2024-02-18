import itertools


c = itertools.count(start=10, step=5)
print(c, type(c))
print(next(c), type(next(c)))
print(next(c))

o = zip(itertools.count(), [11, 22, 33])
for i in o:
    print(i)

