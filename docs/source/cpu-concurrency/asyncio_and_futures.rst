Asyncio and Futures

Overview
--------
This script demonstrates asyncio with command-line arguments, executing two mathematical computations (sum and factorial) concurrently and returning their results.

File Location
``basics/10_asyncio_and_futures.py``

Key Concepts

Command-Line Arguments
~~~~~~~~~~~~~~~~~~~~~~
Uses ``sys.argv`` to accept input values from the command line, making the script configurable.

Async Task Results
~~~~~~~~~~~~~~~~~~
Coroutines can return values that can be collected and used by the caller.

Concurrent Computation
~~~~~~~~~~~~~~~~~~~~~~
Two independent computations run concurrently, each with their own delay.

Code Breakdown

First Coroutine - Sum of N Integers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    async def first_coroutine(num):
        count = 0
        for i in range(1, num + 1):
            count += 1
        await asyncio.sleep(4)
        result = f'First coroutine (sum of N ints) result = {count}'
        print(result)
        return result


Computes a simple sum (which equals ``num``) and waits 4 seconds before returning.

Second Coroutine - Factorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    async def second_coroutine(num):
        count = 1
        for i in range(2, num + 1):
            count *= i
        await asyncio.sleep(4)
        result = f'Second coroutine (factorial) result = {count}'
        print(result)
        return result


Computes factorial and also waits 4 seconds before returning.

Main Function
~~~~~~~~~~~~~
.. code-block:: python

    async def main():
        num1 = int(sys.argv[1])
        num2 = int(sys.argv[2])

        tasks = [
            asyncio.create_task(first_coroutine(num1)),
            asyncio.create_task(second_coroutine(num2))
        ]

        await asyncio.gather(*tasks)


Reads command-line arguments, creates two tasks, and runs them concurrently.

Python 3.12 Updates

Changes Made
~~~~~~~~~~~~
1. **Removed ``asyncio.Future()`` pattern**
   - Old approach used explicit Future objects and callbacks
   - Modern approach: tasks return values directly
   - Simpler and more Pythonic

2. **Replaced ``@asyncio.coroutine`` with ``async def``**
   - Modern syntax for defining coroutines

3. **Replaced ``yield from`` with ``await``**
   - Standard async/await syntax

4. **Removed ``add_done_callback()`` pattern**
   - Old: ``future.add_done_callback(got_result)``
   - New: Direct return values and ``await``
   - Callbacks are unnecessary with modern async/await

5. **Removed ``future.set_result()`` pattern**
   - Old: Manually setting future results
   - New: Simply return values from async functions

6. **Simplified to ``asyncio.create_task()`` and ``asyncio.gather()``**
   - Cleaner task management
   - Results can be captured if needed

7. **Updated to ``asyncio.run()``**
   - Automatic event loop management

8. **Modernized string formatting**
   - f-strings throughout

Before vs After Comparison

Old Pattern (Deprecated)
~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    future = asyncio.Future()
    future.add_done_callback(callback_function)
    await coroutine(future, value)
    future.set_result(result)


New Pattern (Modern)
~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    result = await coroutine(value)
    # Or with tasks:
    task = asyncio.create_task(coroutine(value))
    result = await task


Execution Flow

::

    main() starts
      |
      v
    Reads command-line arguments
      |
      v
    Creates 2 tasks
      |
      +---> Task 1: first_coroutine(num1)
      |
      +---> Task 2: second_coroutine(num2)
      |
      v
    Both tasks execute concurrently
      |
      v
    gather() waits for both
      |
      v
    main() completes


Usage
-----

.. code-block:: bash

    python3 10_asyncio_and_futures.py <num1> <num2>


Examples
--------

Example 1
~~~~~~~~~

.. code-block:: bash

    python3 10_asyncio_and_futures.py 10 5

Output:

::

    First coroutine (sum of N ints) result = 10
    Second coroutine (factorial) result = 120


Example 2
~~~~~~~~~

.. code-block:: bash

    python3 10_asyncio_and_futures.py 100 7

Output:

::

    First coroutine (sum of N ints) result = 100
    Second coroutine (factorial) result = 5040


Performance

- **Sequential execution**: Would take 8 seconds (4 + 4)
- **Concurrent execution**: Takes only 4 seconds (both run together)
- Both coroutines sleep simultaneously
- Demonstrates time savings from concurrency

Key Takeaways

1. **Direct Returns**: Modern async functions return values directly, no need for Future objects
2. **Task Results**: ``asyncio.gather()`` can collect return values: ``results = await asyncio.gather(*tasks)``
3. **Simplicity**: Callback-based code replaced with simple await expressions
4. **Command-Line Integration**: Async programs can easily accept CLI arguments
5. **Concurrent I/O**: Multiple operations run together during wait times

Modern Best Practices

1. **Avoid explicit Future objects** in application code
2. **Use ``asyncio.create_task()``** to schedule coroutines
3. **Use ``asyncio.gather()``** to wait for multiple tasks
4. **Return values directly** from async functions
5. **Use ``asyncio.run()``** as the entry point

When to Use This Pattern

- Multiple independent async operations
- Need to collect results from concurrent tasks
- Operations have waiting periods (I/O, sleep, network)
- Want clean, readable async code without callbacks

Capturing Results

If you need the return values:
.. code-block:: python

async def main():
    tasks = [
        asyncio.create_task(first_coroutine(num1)),
        asyncio.create_task(second_coroutine(num2))
    ]
    results = await asyncio.gather(*tasks)
    # results[0] = first coroutine result
    # results[1] = second coroutine result
::

