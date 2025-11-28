import asyncio
import sys


async def first_coroutine(num):
    count = 0
    for i in range(1, num + 1):
        count += 1
    await asyncio.sleep(4)
    result = f'First coroutine (sum of N ints) result = {count}'
    print(result)
    return result


async def second_coroutine(num):
    count = 1
    for i in range(2, num + 1):
        count *= i
    await asyncio.sleep(4)
    result = f'Second coroutine (factorial) result = {count}'
    print(result)
    return result


async def main():
    # Check if arguments are provided, otherwise use defaults
    if len(sys.argv) < 3:
        print("Usage: python3 10_asyncio_and_futures.py <num1> <num2>")
        print("Example: python3 10_asyncio_and_futures.py 10 5")
        print("\nNo arguments provided, using defaults: num1=10, num2=5\n")
        num1, num2 = 10, 5
    else:
        num1 = int(sys.argv[1])
        num2 = int(sys.argv[2])

    tasks = [
        asyncio.create_task(first_coroutine(num1)),
        asyncio.create_task(second_coroutine(num2))
    ]

    await asyncio.gather(*tasks)


if __name__ == '__main__':
    asyncio.run(main())
