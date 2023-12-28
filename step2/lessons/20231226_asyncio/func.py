import asyncio


async def f(x):
    y = await z(x)
    return y


async def g(x):
    yield x


async def m(x):
    async for value in gen(x):
        yield value


async def z(x):
    return x


async def gen(x):
    y = await z(x)
    yield y