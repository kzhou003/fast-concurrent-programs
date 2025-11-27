Asyncio Event Loop
==================

Overview
--------
This script demonstrates asyncio's event loop by implementing a simple task scheduling system where three tasks (A, B, C) call each other in rotation for 60 seconds.

File Location
-------------
``basics/07*asyncio*event*loop.py``

Key Concepts
------------

Asyncio Event Loop
~~~~~~~~~~~~~~~~~~
The event loop is the core of asyncio's asynchronous execution model. It manages and schedules coroutines, handles I/O operations, and coordinates concurrent tasks.

Async/Await Pattern
~~~~~~~~~~~~~~~~~~~
Modern Python uses ``async def`` to define coroutines and ``await`` to yield control back to the event loop while waiting for operations to complete.

Code Breakdown
--------------

Task Functions
~~~~~~~~~~~~~~
.. code-block:: python

async def task*A(end*time):
    print("task*A called")
    await asyncio.sleep(random.randint(0, 5))
    if (asyncio.get*event*loop().time() + 1.0) < end*time:
        await asyncio.sleep(1)
        await task*B(end*time)
::


Each task:
1. Prints its name
2. Sleeps for a random duration (0-5 seconds)
3. Checks if there's time remaining before end*time
4. If time remains, sleeps for 1 more second and calls the next task
5. Creates a chain: A → B → C → A (repeating)

Main Function
~~~~~~~~~~~~~
.. code-block:: python

async def main():
    loop = asyncio.get*event*loop()
    end*loop = loop.time() + 60
    await task*A(end*loop)
::


Sets up the event loop to run for 60 seconds and initiates the task chain.

Entry Point
~~~~~~~~~~~
.. code-block:: python

if **name** == '**main**':
    asyncio.run(main())
::


``asyncio.run()`` is the modern way to execute the top-level async function, handling event loop creation and cleanup automatically.

Python 3.12 Updates
-------------------

Changes Made
~~~~~~~~~~~~
1. **Replaced blocking ``time.sleep()`` with ``await asyncio.sleep()``**
   - ``time.sleep()`` blocks the entire event loop
   - ``asyncio.sleep()`` yields control, allowing other tasks to run

2. **Converted to async/await syntax**
   - Replaced callback-based event loop manipulation (``loop.call*later``, ``loop.call*soon``)
   - Used modern async function chaining instead

3. **Updated to ``asyncio.run()``**
   - Replaced manual event loop management (``get*event*loop()``, ``run*forever()``, ``close()``)
   - ``asyncio.run()`` is simpler and handles cleanup automatically

4. **Removed ``loop.stop()``**
   - No longer needed with async/await pattern
   - Function naturally completes when time expires

Execution Flow
--------------

::

main() starts
  ↓
task*A() executes
  ↓ (sleeps 0-5 seconds)
  ↓ (sleeps 1 second)
  ↓
task*B() executes
  ↓ (sleeps 0-5 seconds)
  ↓ (sleeps 1 second)
  ↓
task*C() executes
  ↓ (sleeps 0-5 seconds)
  ↓ (sleeps 1 second)
  ↓
task*A() executes again
  ↓
... (repeats for ~60 seconds)
  ↓
Time expires, chain stops
::


Usage
-----
.. code-block:: bash

python3 07*asyncio*event*loop.py
::


The script will run for approximately 60 seconds, printing task names as they execute.

Output Example
--------------
::

task*A called
task*B called
task*C called
task*A called
task*B called
task_C called
...
::


Key Takeaways
-------------

1. **Non-blocking Sleep**: Always use ``asyncio.sleep()`` in async code, never ``time.sleep()``
2. **Event Loop Time**: Use ``loop.time()`` for precise timing within the event loop
3. **Async Chains**: Coroutines can call other coroutines using ``await``
4. **Modern asyncio**: ``asyncio.run()`` is the preferred way to run async code in Python 3.7+
