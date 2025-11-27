Concurrent Futures Pooling
==========================

Overview
--------
This script demonstrates the performance differences between sequential execution, thread pool execution, and process pool execution using Python's ``concurrent.futures`` module.

File Location
-------------
``basics/06*concurrent*futures*pooling.py``

Key Concepts
------------

Executor Pattern
~~~~~~~~~~~~~~~~
The script uses the executor pattern from ``concurrent.futures`` which provides a high-level interface for asynchronously executing callables.

Execution Models
~~~~~~~~~~~~~~~~

1. **Sequential Execution**: Tasks run one after another on a single thread
2. **ThreadPoolExecutor**: Tasks run concurrently using multiple threads (limited by GIL for CPU-bound tasks)
3. **ProcessPoolExecutor**: Tasks run in parallel using multiple processes (true parallelism for CPU-bound tasks)

Code Breakdown
--------------

CPU-Intensive Task
~~~~~~~~~~~~~~~~~~
.. code-block:: python

def count(number):
    for i in range(0, 10000000):
        i += 1
    return i * number
::

A CPU-bound operation that performs 10 million iterations to demonstrate performance differences.

Evaluation Function
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

def evaluate(item):
    result*item = count(item)
    print(f'Item {item}, result {result*item}')
::

Wrapper function that executes the CPU-intensive task and prints the result.

Sequential Execution
~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

start*time = time.perf*counter()
for item in number*list:
    evaluate(item)
print(f'Sequential Execution in {time.perf*counter() - start*time} seconds')
::

Baseline performance - executes all tasks sequentially in the main thread.

Thread Pool Execution
~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

with concurrent.futures.ThreadPoolExecutor(max*workers=5) as executor:
    for item in number*list:
        executor.submit(evaluate, item)
::

Uses a pool of 5 threads. Due to Python's Global Interpreter Lock (GIL), CPU-bound tasks don't see much performance improvement with threads.

Process Pool Execution
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

with concurrent.futures.ProcessPoolExecutor(max*workers=5) as executor:
    for item in number*list:
        executor.submit(evaluate, item)
::

Uses a pool of 5 separate processes. This bypasses the GIL and provides true parallel execution for CPU-bound tasks.

Performance Expectations
------------------------

For CPU-bound tasks:
- **Sequential**: Baseline performance
- **ThreadPool**: Similar or slightly slower than sequential (due to GIL and threading overhead)
- **ProcessPool**: Significantly faster (near-linear speedup based on number of cores)

Python 3.12 Updates
-------------------

Changes Made
~~~~~~~~~~~~
1. Replaced deprecated ``time.clock()`` with ``time.perf*counter()``
   - ``time.clock()`` was removed in Python 3.8
   - ``time.perf*counter()`` provides higher resolution and is the recommended timing function

2. Updated string formatting from ``%`` to f-strings
   - More readable and Pythonic
   - Better performance

Usage
-----
.. code-block:: bash

python3 06*concurrent*futures_pooling.py
::


Output Example
--------------
::

Item 1, result 10000000
Item 2, result 20000000
...
Sequential Execution in 3.47 seconds
...
Thread Pool Execution in 3.31 seconds
...
Process Pool Execution in 1.23 seconds
::


When to Use Each Approach
-------------------------

- **Sequential**: Simple tasks, I/O-bound operations, or when order matters
- **ThreadPool**: I/O-bound tasks (network requests, file operations)
- **ProcessPool**: CPU-bound tasks that need true parallelism
