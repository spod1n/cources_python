import asyncio

async def enter():
    enter_1 = str(input("Enter your text: "))
    return enter_1

async def processing_input(enter_1: str):
    if enter_1.lower() == "exit":
        return 1
    else:
        return 0

async def main():
    result = await asyncio.gather(processing_input(await enter()))
    if result:
        print(await enter())
    else:
        return 0

if __name__ == "__main__":
    asyncio.run(main())