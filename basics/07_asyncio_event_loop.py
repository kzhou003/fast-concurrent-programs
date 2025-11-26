import asyncio
import random


async def task_A(end_time):
    print("task_A called")
    await asyncio.sleep(random.randint(0, 5))
    if (asyncio.get_event_loop().time() + 1.0) < end_time:
        await asyncio.sleep(1)
        await task_B(end_time)


async def task_B(end_time):
    print("task_B called")
    await asyncio.sleep(random.randint(0, 5))
    if (asyncio.get_event_loop().time() + 1.0) < end_time:
        await asyncio.sleep(1)
        await task_C(end_time)


async def task_C(end_time):
    print("task_C called")
    await asyncio.sleep(random.randint(0, 5))
    if (asyncio.get_event_loop().time() + 1.0) < end_time:
        await asyncio.sleep(1)
        await task_A(end_time)


async def main():
    loop = asyncio.get_event_loop()
    end_loop = loop.time() + 60
    await task_A(end_loop)


if __name__ == '__main__':
    asyncio.run(main())

