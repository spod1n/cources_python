"""
Задача 3: Паралельний асинхронний запуск завдань

Створіть асинхронну програму, яка паралельно виконує два завдання
з різними тривалостями та виводить результат їх виконання.
"""

import asyncio


async def task_with_delay(name: str, delay: int):
    print(f'Start task {name}')
    await asyncio.sleep(delay)
    print(f'End task {name} after {delay} sec.')


async def main():
    task1 = task_with_delay('Task1', 10)
    task2 = task_with_delay('Task2', 5)

    await asyncio.gather(task1, task2)


if __name__ == "__main__":
    asyncio.run(main())
