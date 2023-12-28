"""
Задача 2: Асинхронна обробка списку елементів з функцією зворотнього виклику

Створіть асинхронну програму для обробки списку чисел.
Кожне число повинно бути підняте до квадрату, а результати обробки виведені на екран.
"""

import asyncio


async def process_items(items, callback):
    for item in items:
        result = await callback(item)
        print(f"Processed item: {item}, Result:{result}")


async def squre_callback(number):
    return number ** 2


async def squre_callback3(number):
    await asyncio.sleep(2)
    return number ** 3


async def main():
    items = [1, 2, 3, 4, 5]
    print("Processing items...")
    await process_items(items, squre_callback)
    await process_items(items, squre_callback3)


if __name__ == "__main__":
    asyncio.run(main())
