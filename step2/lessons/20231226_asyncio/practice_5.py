"""
Створіть асинхронну програму, яка очікує введення користувача та виводить введені дані на екран.
Програма повинна продовжувати працювати, доки користувач не введе слово "exit".
"""

import asyncio


async def inp():
    while True:
        user_input = input('Input string: ')
        if user_input.lower() == 'exit':
            print('shutdown -y -g0')
            break
        print(f"You input: '{user_input}'. Maybe you input 'exit'?")


async def main():
    await asyncio.gather(inp())

if __name__ == "__main__":
    asyncio.run(main())
