"""
Задача 1: Асинхронний таймер із зупинкою

Створіть асинхронний таймер, який виводить повідомлення кожні 2 секунди.
Протягом 10 секунд програма повинна виводити повідомлення, а потім зупинятися.
"""


import asyncio


async def msg():
    while True:
        print('Hello world')
        await asyncio.sleep(2)


async def msg2():
    while True:
        print('Hello world2')
        await asyncio.sleep(2)


async def main():
    # await asyncio.gather(msg())
    task = asyncio.create_task(msg())
    task2 = asyncio.create_task(msg2())
    await asyncio.sleep(10)

    task.cancel()
    task2.cancel()

if __name__ == '__main__':
    asyncio.run(main())
