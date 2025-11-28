Why Different Approaches for CPU-bound vs I/O-bound Problems?
=============================================================

A deep dive into the technical reasons behind choosing multiprocessing for CPU-bound tasks and threading/asyncio for I/O-bound tasks in Python.

Table of Contents
-----------------
1. `The Fundamental Problem: The GIL <#the-fundamental-problem-the-gil>`_
2. `How CPU and I/O Operations Differ <#how-cpu-and-io-operations-differ>`_
3. `Why Multiprocessing for CPU-bound <#why-multiprocessing-for-cpu-bound>`_
4. `Why Threading for I/O-bound <#why-threading-for-io-bound>`_
5. `Why Asyncio for I/O-bound <#why-asyncio-for-io-bound>`_
6. `Deep Dive: What Happens Under the Hood <#deep-dive-what-happens-under-the-hood>`_
7. `Performance Analysis <#performance-analysis>`_
8. `Decision Tree <#decision-tree>`_

---

The Fundamental Problem: The GIL
--------------------------------

What is the GIL?
~~~~~~~~~~~~~~~~

The **Global Interpreter Lock (GIL)** is a mutex (mutual exclusion lock) in CPython that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously.

Why Does Python Have a GIL?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Historical Context:
+---------------------------------------------+
| Python was designed in the late 1980s       |
| when single-core CPUs were the norm        |
|                                             |
| Design Decision:                            |
| * Simple memory management (ref counting)   |
| * Easy C extension integration              |
| * Thread-safe by default                    |
| * Trade-off: One GIL = Simple design       |
+---------------------------------------------+
::


How the GIL Works
~~~~~~~~~~~~~~~~~

.. code-block:: python

Conceptual representation of GIL behavior
=========================================

Thread 1: [Acquire GIL] -> Execute Python Code -> [Release GIL]
                                                      down
Thread 2:                    [Waiting...]  -> [Acquire GIL] -> Execute
.. code-block:: text




**Key Point**: Only ONE thread can execute Python bytecode at a time, even on a multi-core CPU.

GIL Behavior with Different Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

CPU-bound operation
===================
def cpu_intensive():
    total = 0
    for i in range(10*000*000):
        total += i
    return total

Thread 1 acquires GIL -> executes -> releases after ~100 bytecodes -> repeat
=========================================================================
Thread 2 waits -> acquires GIL -> executes -> releases -> repeat
============================================================
Result: Threads take TURNS, no parallel execution
=================================================
.. code-block:: text




.. code-block:: python

I/O-bound operation
===================
def io_intensive():
    response = requests.get('https://api.example.com/data')
    return response.json()

Thread 1 acquires GIL -> starts I/O -> RELEASES GIL during I/O wait
=================================================================
Thread 2 can now acquire GIL and execute while Thread 1 waits
=============================================================
Result: Threads can work while others are waiting for I/O
=========================================================
::


The Critical Difference
~~~~~~~~~~~~~~~~~~~~~~~

| Operation Type | GIL Released During Operation? | Result |
|----------------|-------------------------------|--------|
| CPU-bound (pure Python) | [[FAIL]] No | Threads execute sequentially |
| I/O-bound (network, disk) | [[OK]] Yes | Threads can work concurrently |
| C extensions (NumPy, etc.) | [[OK]] Often yes | Can achieve parallelism |

---

How CPU and I/O Operations Differ
---------------------------------

CPU-bound Operations
~~~~~~~~~~~~~~~~~~~~

**Definition**: Operations where execution time is determined by CPU processing speed.

**Characteristics**:
::

CPU Usage:  ######################## (100%)
I/O Wait:   (minimal or none)
Bottleneck: CPU cycles

Example Timeline:
0ms   --[>] Processing --[>] Processing --[>] Processing --[>] Done
         (CPU busy)      (CPU busy)      (CPU busy)
::


**What happens**:
1. CPU fetches instructions from memory
2. CPU executes mathematical/logical operations
3. CPU writes results to memory/registers
4. Repeat continuously

**No waiting** - CPU is constantly working.

I/O-bound Operations
~~~~~~~~~~~~~~~~~~~~

**Definition**: Operations where execution time is determined by waiting for input/output.

**Characteristics**:
::

CPU Usage:  #_______#_______# (sporadic, mostly idle)
I/O Wait:   _########_######## (most of the time)
Bottleneck: Waiting for external resources

Example Timeline:
0ms   --[>] Request --[>] Waiting... --[>] Waiting... --[>] Response --[>] Process
         (CPU)        (I/O device)   (I/O device)   (network)   (CPU)
          100mus           50ms           50ms         100ms      1ms
::


**What happens**:
1. CPU initiates I/O request (network/disk)
2. **CPU sits idle waiting** for response
3. I/O device/network does the work
4. Response arrives
5. CPU processes the response (brief)

**Lots of waiting** - CPU is idle most of the time.

Real-World Analogy
~~~~~~~~~~~~~~~~~~

**CPU-bound** (Computing factorial):
::

You: Calculate 1000! in your head
     +-[>] You must think continuously
     +-[>] Cannot do anything else while thinking
     +-[>] Limited by your brain's processing speed
.. code-block:: text




**I/O-bound** (Ordering pizza):
.. code-block:: text



You: Call pizza place -> Wait 30 min -> Receive pizza
     +-[>] Phone call: 1 minute (active)
     +-[>] Waiting: 29 minutes (idle - can do other things!)
     +-[>] Receive: 1 minute (active)
     +-[>] Limited by pizza shop's speed, not yours
.. code-block:: text




---

Why Multiprocessing for CPU-bound
---------------------------------

The Problem with Threading for CPU-bound
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

CPU-intensive task with threading
=================================
import threading
import time

def cpu_work():
    total = sum(i_i for i in range(10*000*000))
    return total

Sequential
==========
start = time.perf_counter()
cpu_work()
cpu_work()
print(f"Sequential: {time.perf_counter() - start:.2f}s")
Output: Sequential: 2.50s
=========================

Threading (SAME TIME OR WORSE!)
===============================
start = time.perf_counter()
t1 = threading.Thread(target=cpu_work)
t2 = threading.Thread(target=cpu_work)
t1.start(); t2.start()
t1.join(); t2.join()
print(f"Threading: {time.perf_counter() - start:.2f}s")
Output: Threading: 2.55s (no improvement!)
==========================================
::


**Why no speedup?**

::

With GIL (Threading):
Core 1: [Thread 1][Thread 2][Thread 1][Thread 2][Thread 1][Thread 2]
Core 2: [idle....................................................]
Core 3: [idle....................................................]
Core 4: [idle....................................................]
Time:   ==============================================================================================================[>]

Result: Only using 1 core, taking turns due to GIL
::


The Solution: Multiprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

CPU-intensive task with multiprocessing
=======================================
from concurrent.futures import ProcessPoolExecutor
import time

def cpu_work():
    return sum(i_i for i in range(10*000*000))

Multiprocessing
===============
start = time.perf_counter()
with ProcessPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(cpu_work) for * in range(2)]
    results = [f.result() for f in futures]
print(f"Multiprocessing: {time.perf_counter() - start:.2f}s")
Output: Multiprocessing: 1.30s (nearly 2x speedup!)
===================================================
::


**Why it works:**

::

Without GIL (Multiprocessing):
Process 1 on Core 1: [########################] Complete
Process 2 on Core 2: [########################] Complete
Process 3 on Core 3: [idle]
Process 4 on Core 4: [idle]
Time:                ==================================================[>]

Result: Using 2 cores in TRUE parallel execution
::


How Multiprocessing Bypasses the GIL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each process has:
- **Its own Python interpreter**
- **Its own GIL** (doesn't interfere with other processes)
- **Its own memory space**
- **Its own process ID**

.. code-block:: text


    +------------------+  +------------------+  +------------------+
    |   Process 1      |  |   Process 2      |  |   Process 3      |
    |                  |  |                  |  |                  |
    |  +------------+  |  |  +------------+  |  |  +------------+  |
    |  | GIL #1     |  |  |  | GIL #2     |  |  |  | GIL #3     |  |
    |  +------------+  |  |  +------------+  |  |  +------------+  |
    |  +------------+  |  |  +------------+  |  |  +------------+  |
    |  | Interpreter|  |  |  | Interpreter|  |  |  | Interpreter|  |
    |  +------------+  |  |  +------------+  |  |  +------------+  |
    |  +------------+  |  |  +------------+  |  |  +------------+  |
    |  |   Memory   |  |  |  |   Memory   |  |  |  |   Memory   |  |
    |  +------------+  |  |  +------------+  |  |  +------------+  |
    +------------------+  +------------------+  +------------------+
    CPU Core 1           CPU Core 2           CPU Core 3

.. code-block:: text




Trade-offs of Multiprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Advantages**:
- [[OK]] True parallelism - uses multiple CPU cores
- [[OK]] No GIL interference between processes
- [[OK]] Process isolation (crash in one doesn't affect others)
- [[OK]] Can achieve near-linear speedup for CPU-bound tasks

**Disadvantages**:
- [[FAIL]] Higher memory usage (each process has full Python interpreter)
- [[FAIL]] Slower startup time (creating processes is expensive)
- [[FAIL]] Inter-process communication is complex and slow
- [[FAIL]] Cannot share memory directly (must pickle/unpickle data)

When the Trade-off is Worth It
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text



Memory cost per process: ~10-50 MB
Speedup for CPU-bound tasks: 2-8x (depending on cores)

Example:
Task: Process 1000 images (CPU-intensive)
|- Sequential: 100 seconds
|- Threading: 100 seconds (no improvement due to GIL)
+- Multiprocessing (4 cores): 27 seconds
   +- Memory cost: 150 MB vs 50 MB
   +- Worth it? YES! (3.7x speedup for 100MB extra)
.. code-block:: text




---

Why Threading for I/O-bound
---------------------------

The Problem: Wasted Time
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Sequential I/O operations
=========================
import requests
import time

urls = ['https://api1.com', 'https://api2.com', 'https://api3.com']

start = time.perf_counter()
for url in urls:
    response = requests.get(url)  # Takes 2 seconds each
    process(response)
print(f"Sequential: {time.perf_counter() - start:.2f}s")
Output: Sequential: 6.00s (2s + 2s + 2s)
========================================

Timeline:
0s    2s    4s    6s
|-----+-----+-----|
|Wait1|Wait2|Wait3|  <- CPU is IDLE during all this time!
::


The Solution: Threading
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Threaded I/O operations
=======================
import threading
import requests
import time

def fetch(url):
    response = requests.get(url)
    process(response)

urls = ['https://api1.com', 'https://api2.com', 'https://api3.com']

start = time.perf_counter()
threads = [threading.Thread(target=fetch, args=(url,)) for url in urls]
for t in threads: t.start()
for t in threads: t.join()
print(f"Threading: {time.perf_counter() - start:.2f}s")
Output: Threading: 2.05s (all wait in parallel!)
================================================

Timeline:
0s    2s
|-----|
|Wait1|  <- Thread 1
|Wait2|  <- Thread 2
|Wait3|  <- Thread 3 (all waiting simultaneously)
::


Why Threading Works for I/O
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The GIL is Released During I/O Operations!**

.. code-block:: python

What happens under the hood:
============================

Thread 1: [Acquire GIL] -> Start network request -> [Release GIL] -> Wait...
                                                        down
Thread 2:                    [Acquire GIL] -> Start disk read -> [Release GIL]
                                                                      down
Thread 3:                                      [Acquire GIL] -> Start DB query
::


**Key Insight**: While Thread 1 waits for network I/O, Threads 2 and 3 can start their I/O operations. All three are waiting simultaneously!

How the OS Helps
~~~~~~~~~~~~~~~~

When Python releases the GIL during I/O:

::

Python Thread          Operating System           I/O Device
    |                        |                        |
    |--- Read file ---------->|                        |
    | [Release GIL]          |--- Send request ------->|
    |                        |                        |
    | (Thread sleeps)        |                     (Working)
    |                        |                        |
    |                        |<--- Data ready ---------|
    |<--- Data ready ---------|                        |
    | [Acquire GIL]          |                        |
    |--- Process data        |                        |
::


The OS handles I/O asynchronously while the thread waits, allowing other threads to work.

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Real example: Downloading 10 web pages
======================================

Sequential (1 thread):
|- Page 1: 0.5s
|- Page 2: 0.5s
|- ...
+- Page 10: 0.5s
Total: 5.0 seconds

Threading (10 threads):
|- All pages: 0.5s (in parallel)
+- Total: 0.5 seconds

Speedup: 10x! (near-perfect for I/O-bound)
::


Why Not Multiprocessing for I/O?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Threading for I/O:
|- Memory: 50 MB (one process, 10 threads)
|- Startup: Instant (threads are lightweight)
|- Communication: Shared memory (easy)
+- Performance: 10x speedup

Multiprocessing for I/O:
|- Memory: 500 MB (10 processes x 50 MB each)
|- Startup: Slow (creating 10 processes)
|- Communication: IPC (complex, slow)
+- Performance: 10x speedup (same as threading!)

Verdict: Threading is MORE EFFICIENT for I/O
.. code-block:: text




Threading Trade-offs
~~~~~~~~~~~~~~~~~~~~

**Advantages**:
- [[OK]] Lightweight (minimal memory overhead)
- [[OK]] Fast to create/destroy
- [[OK]] Easy data sharing (shared memory)
- [[OK]] Perfect for I/O-bound tasks

**Disadvantages**:
- [[FAIL]] No speedup for CPU-bound tasks (GIL)
- [[FAIL]] Race conditions possible with shared state
- [[FAIL]] More complex debugging
- [[FAIL]] Limited by GIL for Python code execution

---

Why Asyncio for I/O-bound
-------------------------

The Problem with Threading: Overhead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Creating 10,000 threads:
|- Memory: 10,000 threads x 8 MB stack = 80 GB! (impossible)
|- Context switching: OS must switch between 10,000 threads
|- Overhead: Significant CPU time spent on thread management
+- Result: System becomes unresponsive

Creating 10,000 asyncio tasks:
|- Memory: ~10-100 MB total (tasks are lightweight)
|- Context switching: Controlled by Python (no OS involvement)
|- Overhead: Minimal
+- Result: Efficient and responsive
::


Asyncio: Cooperative Multitasking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Threading** (Preemptive - OS decides when to switch):
::

OS: "Thread 1, you've used enough CPU, I'm switching to Thread 2"
Thread 1: "But I'm not done!"
OS: "Too bad, Thread 2's turn now"
.. code-block:: text




**Asyncio** (Cooperative - code decides when to yield):
::

Task 1: "I'm about to wait for network, let me yield control"
Event Loop: "Thanks! I'll run Task 2 now"
Task 2: "I'm about to wait for disk, let me yield"
Event Loop: "Got it! I'll check if Task 1's network response arrived"
::


How Asyncio Works
~~~~~~~~~~~~~~~~~

.. code-block:: python

Asyncio example: 10,000 concurrent requests
===========================================
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [f'https://api.example.com/item/{i}' for i in range(10000)]
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

asyncio.run(main())
Can handle 10,000 requests efficiently!
=======================================
::


Event Loop Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

::

Event Loop (Single Thread):
+---------------------------------------------------------+
|                                                         |
|  Ready Queue: [Task 1, Task 5, Task 12, ...]          |
|                                                         |
|  Waiting for I/O: {Task 2: socket 1,                   |
|                    Task 3: socket 2,                   |
|                    Task 4: socket 3, ...}              |
|                                                         |
|  Flow:                                                  |
|  1. Get next ready task                                |
|  2. Run until it awaits something                      |
|  3. Check which I/O operations completed               |
|  4. Move completed tasks to ready queue                |
|  5. Repeat                                             |
|                                                         |
+---------------------------------------------------------+
::


Asyncio vs Threading: Detailed Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Threading approach
==================
import threading
import requests

def fetch_url(url):
    response = requests.get(url)
    return response.text

urls = [f'https://api.example.com/{i}' for i in range(1000)]
threads = [threading.Thread(target=fetch_url, args=(url,)) for url in urls]

Problem: Creating 1000 threads!
===============================
Memory: ~8 GB (1000 x 8 MB stack per thread)
============================================
OS overhead: Managing 1000 threads
==================================
.. code-block:: text




.. code-block:: python

Asyncio approach
================
import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [f'https://api.example.com/{i}' for i in range(1000)]
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

asyncio.run(main())

Solution: Single thread, 1000 lightweight tasks
===============================================
Memory: ~50 MB total
====================
OS overhead: None (all managed by Python)
=========================================
::


Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Metric | Threading (1000 ops) | Asyncio (1000 ops) |
|--------|---------------------|-------------------|
| Memory Usage | ~8 GB | ~50 MB |
| Context Switch | OS-level (slow) | Python-level (fast) |
| Scalability | ~1000s | ~100,000s |
| Startup Time | Slow (create threads) | Fast (create tasks) |
| CPU Overhead | High (OS scheduling) | Low (event loop) |

When Asyncio Shines
~~~~~~~~~~~~~~~~~~~

**Perfect for**:
::

[[OK]] Web servers (handle many simultaneous connections)
[[OK]] Web scraping (thousands of HTTP requests)
[[OK]] Database queries (many concurrent queries)
[[OK]] Microservices (coordinating many API calls)
[[OK]] Chat applications (many idle connections)
[[OK]] IoT systems (many devices sending data)
::


**Not ideal for**:
.. code-block:: text



[[FAIL]] CPU-intensive tasks (use multiprocessing)
[[FAIL]] Blocking libraries (must use async-compatible libraries)
[[FAIL]] Simple scripts with few I/O operations (threading is simpler)
.. code-block:: text




Asyncio Trade-offs
~~~~~~~~~~~~~~~~~~

**Advantages**:
- [[OK]] Extremely lightweight (handle 100,000+ concurrent operations)
- [[OK]] Low memory overhead
- [[OK]] Fast context switching (Python-level)
- [[OK]] Single-threaded (no race conditions)
- [[OK]] Explicit concurrency (clear control flow with ``await``)

**Disadvantages**:
- [[FAIL]] Requires async-compatible libraries (can't use standard ``requests``, etc.)
- [[FAIL]] Learning curve (async/await paradigm)
- [[FAIL]] Viral nature (once you go async, everything must be async)
- [[FAIL]] No speedup for CPU-bound tasks
- [[FAIL]] One blocking operation blocks everything

---

Deep Dive: What Happens Under the Hood
--------------------------------------

CPU-bound with Threading: The GIL Dance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Example: Two threads computing sum
==================================
import threading

def compute_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total

t1 = threading.Thread(target=compute_sum, args=(10*000*000,))
t2 = threading.Thread(target=compute_sum, args=(10*000*000,))
::


**What actually happens**:

.. code-block:: text



Time ->
0ms    Thread 1: [Acquire GIL]
1ms              : Execute: total = 0
2ms              : Execute: total += 1
3ms              : Execute: total += 2
...
100ms            : [Release GIL] (every ~100 bytecodes or 5ms)
100ms  Thread 2:                 [Acquire GIL]
101ms            :                 Execute: total = 0
102ms            :                 Execute: total += 1
...
200ms            :                 [Release GIL]
200ms  Thread 1: [Acquire GIL]
...
(continues alternating)

Result: Threads take turns executing Python bytecode
No parallelism for CPU work!
.. code-block:: text




CPU-bound with Multiprocessing: True Parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from concurrent.futures import ProcessPoolExecutor

def compute_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total

with ProcessPoolExecutor(max_workers=2) as executor:
    future1 = executor.submit(compute_sum, 10*000*000)
    future2 = executor.submit(compute_sum, 10*000*000)
::


**What actually happens**:

.. code-block:: text



Time ->
0ms    Process 1 (Core 1): [Start] Create interpreter, load code
10ms                       : Execute: total = 0
11ms                       : Execute: total += 1
12ms                       : Execute: total += 2
...
1000ms                     : [Done] Return result via pipe

0ms    Process 2 (Core 2): [Start] Create interpreter, load code
10ms                       : Execute: total = 0
11ms                       : Execute: total += 1
12ms                       : Execute: total += 2
...
1000ms                     : [Done] Return result via pipe

Result: Both processes execute SIMULTANEOUSLY on different cores
True parallelism!
.. code-block:: text




I/O-bound with Threading: GIL Released
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

import threading
import requests

def fetch_url(url):
    response = requests.get(url)  # I/O operation
    return response.text

t1 = threading.Thread(target=fetch_url, args=('https://api1.com',))
t2 = threading.Thread(target=fetch_url, args=('https://api2.com',))
::


**What actually happens**:

.. code-block:: text



Time ->
0ms    Thread 1: [Acquire GIL]
1ms             : Prepare HTTP request
2ms             : [Release GIL] <- Call to C library (requests)
2ms             : [OS: Send network packet]
3ms             : [OS: Waiting for response...]

2ms    Thread 2:                [Acquire GIL] <- Can run while T1 waits!
3ms             :                Prepare HTTP request
4ms             :                [Release GIL] <- Call to C library
4ms             :                [OS: Send network packet]
5ms             :                [OS: Waiting for response...]

200ms           : [OS: T1's response arrives]
200ms  Thread 1: [Acquire GIL]
201ms           : Process response
202ms           : [Done]

210ms           : [OS: T2's response arrives]
210ms  Thread 2: [Acquire GIL]
211ms           : Process response
212ms           : [Done]

Result: Both threads waited concurrently (overlapped I/O)
Total time: ~210ms instead of 400ms sequential
.. code-block:: text




I/O-bound with Asyncio: Event Loop Magic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        task1 = asyncio.create_task(fetch_url(session, 'https://api1.com'))
        task2 = asyncio.create_task(fetch_url(session, 'https://api2.com'))
        results = await asyncio.gather(task1, task2)
::


**What actually happens**:

.. code-block:: text



Time -> (Single Thread)
0ms    Event Loop: Create task1
1ms              : Create task2
2ms              : Run task1 until await
3ms    Task 1   : Send HTTP request
4ms              : await response.get() <- Yields control
4ms    Event Loop: task1 waiting for I/O, switch to task2
5ms    Task 2   : Send HTTP request
6ms              : await response.get() <- Yields control
6ms    Event Loop: Both tasks waiting, check I/O status
...
200ms  Event Loop: task1's I/O completed
200ms  Task 1   : Process response
201ms            : Return result
201ms  Event Loop: task1 done, check task2
210ms  Event Loop: task2's I/O completed
210ms  Task 2   : Process response
211ms            : Return result
212ms  Event Loop: Both tasks done, gather returns

Result: Single thread efficiently managing multiple I/O operations
No thread overhead, same concurrency benefit
.. code-block:: text




---

Performance Analysis
--------------------

Benchmark: CPU-bound Task (Computing pi)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

def compute_pi(iterations):
    """CPU-intensive: Monte Carlo pi approximation"""
    inside = 0
    for * in range(iterations):
        x, y = random.random(), random.random()
        if x_x + y_y <= 1:
            inside += 1
    return 4 * inside / iterations

ITERATIONS = 10*000*000
TASKS = 4
.. code-block:: text




**Results on 4-core CPU**:

| Approach | Time | Speedup | Memory |
|----------|------|---------|---------|
| Sequential | 10.0s | 1.0x | 50 MB |
| Threading (4 threads) | 10.2s | 0.98x [[FAIL]] | 55 MB |
| Asyncio (4 tasks) | 10.1s | 0.99x [[FAIL]] | 52 MB |
| Multiprocessing (4 proc) | 2.7s | 3.7x [[OK]] | 200 MB |

**Analysis**:
- Threading/Asyncio: No improvement (GIL limitation)
- Multiprocessing: Near-linear speedup (3.7x on 4 cores)
- Memory trade-off is worth it for 3.7x speedup

Benchmark: I/O-bound Task (Web Requests)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

async def fetch_page(session, url):
    """I/O-intensive: Download web page"""
    async with session.get(url) as response:
        return await response.text()

URLS = 100 (each takes ~0.5s to fetch)
::


**Results**:

| Approach | Time | Speedup | Memory | Max Concurrent |
|----------|------|---------|---------|----------------|
| Sequential | 50.0s | 1.0x | 50 MB | 1 |
| Threading (10 threads) | 5.2s | 9.6x [[OK]] | 130 MB | 10 |
| Threading (100 threads) | 1.8s | 27.8x [[OK]] | 850 MB | 100 |
| Asyncio (100 tasks) | 1.5s | 33.3x [[OK]] | 65 MB | 100 |
| Multiprocessing (4 proc) | 13.0s | 3.8x [[FAIL]] | 200 MB | 4 |

**Analysis**:
- Threading: Good speedup, but memory grows with threads
- Asyncio: Best speedup with lowest memory
- Multiprocessing: Poor choice (high overhead, limited concurrency)

Benchmark: Mixed Workload
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

async def process_data(session, url):
    """Fetch data (I/O) then process (CPU)"""
    # I/O: Fetch data (2 seconds)
    async with session.get(url) as response:
        data = await response.json()

    # CPU: Heavy processing (1 second)
    result = complex_computation(data)
    return result

TASKS = 10
::


**Results**:

| Approach | Time | Analysis |
|----------|------|----------|
| Sequential | 30.0s | (2s I/O + 1s CPU) x 10 |
| Threading | 15.5s | I/O concurrent, CPU sequential |
| Asyncio | 15.2s | I/O concurrent, CPU sequential |
| Asyncio + ProcessPool | 5.8s | I/O concurrent, CPU parallel [[OK]] |

**Best approach for mixed workload**:
.. code-block:: python

import asyncio
from concurrent.futures import ProcessPoolExecutor

async def process_data(session, url, executor):
    # I/O: Use asyncio
    async with session.get(url) as response:
        data = await response.json()

    # CPU: Offload to process pool
    loop = asyncio.get_event*loop()
    result = await loop.run_in*executor(executor, complex_computation, data)
    return result

async def main():
    with ProcessPoolExecutor(max_workers=4) as executor:
        async with aiohttp.ClientSession() as session:
            tasks = [process_data(session, url, executor) for url in urls]
            results = await asyncio.gather(*tasks)
.. code-block:: text




---

Decision Tree
-------------

Use this decision tree to choose the right approach:

::

Start: What type of operation?
|
|- CPU-bound (math, data processing, compression)
|  |
|  |- How many tasks?
|  |  |- Single task -> Sequential (simplest)
|  |  +- Multiple tasks -> Multiprocessing
|  |
|  +- How many CPU cores available?
|     |- 1 core -> Sequential (multiprocessing won't help)
|     +- 2+ cores -> Multiprocessing (use max_workers = CPU cores)
|
+- I/O-bound (network, disk, database)
   |
   |- How many concurrent operations?
   |  |
   |  |- Few (< 50)
   |  |  |- Using blocking libraries? -> Threading
   |  |  +- Can use async libraries? -> Asyncio (preferred)
   |  |
   |  |- Many (50-1000)
   |  |  +- Asyncio (threading becomes expensive)
   |  |
   |  +- Very many (> 1000)
   |     +- Asyncio (threading impossible)
   |
   +- Do you need simplicity or scalability?
      |- Simplicity -> Threading (easier to understand)
      +- Scalability -> Asyncio (better performance)
::


Quick Reference Table
~~~~~~~~~~~~~~~~~~~~~

| Scenario | Solution | Reason |
|----------|----------|--------|
| Image processing (1000 images) | Multiprocessing | CPU-bound, benefits from parallel cores |
| Web scraping (100 pages) | Asyncio | I/O-bound, many concurrent connections |
| REST API server | Asyncio | I/O-bound, handle many simultaneous requests |
| Video encoding | Multiprocessing | CPU-intensive, utilize all cores |
| Database queries (10 concurrent) | Threading | I/O-bound, simple implementation |
| Database queries (1000 concurrent) | Asyncio | I/O-bound, need high concurrency |
| File downloads (5 files) | Threading | I/O-bound, blocking library OK |
| WebSocket server (10000 clients) | Asyncio | I/O-bound, need extreme scalability |
| Scientific computation | Multiprocessing | CPU-intensive calculations |
| Real-time chat (1000 users) | Asyncio | I/O-bound, many idle connections |

Code Templates
~~~~~~~~~~~~~~

**CPU-bound Template**:
.. code-block:: python

from concurrent.futures import ProcessPoolExecutor
import os

def cpu_intensive*task(data):
    # Your CPU-heavy computation here
    result = complex_calculation(data)
    return result

def main():
    data_items = [...]  # Your data

    # Use number of CPU cores
    max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(cpu_intensive*task, data_items)

    return list(results)
.. code-block:: text




**I/O-bound Template (Few operations, blocking library)**:
.. code-block:: python

from concurrent.futures import ThreadPoolExecutor
import requests

def io_intensive*task(url):
    response = requests.get(url)
    return process(response)

def main():
    urls = [...]  # Your URLs

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(io_intensive*task, urls)

    return list(results)
.. code-block:: text




**I/O-bound Template (Many operations, async library)**:
.. code-block:: python

import asyncio
import aiohttp

async def io_intensive*task(session, url):
    async with session.get(url) as response:
        data = await response.text()
        return process(data)

async def main():
    urls = [...]  # Your URLs

    async with aiohttp.ClientSession() as session:
        tasks = [io_intensive*task(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    return results

if __name** == '**main**':
    asyncio.run(main())
::


**Mixed Workload Template**:
.. code-block:: python

import asyncio
import aiohttp
from concurrent.futures import ProcessPoolExecutor

def cpu_intensive(data):
    # CPU-heavy work here
    return complex_calculation(data)

async def mixed_task(session, url, executor):
    # I/O part (async)
    async with session.get(url) as response:
        data = await response.json()

    # CPU part (process pool)
    loop = asyncio.get_event*loop()
    result = await loop.run_in*executor(executor, cpu_intensive, data)

    return result

async def main():
    urls = [...]

    with ProcessPoolExecutor(max_workers=4) as executor:
        async with aiohttp.ClientSession() as session:
            tasks = [mixed_task(session, url, executor) for url in urls]
            results = await asyncio.gather(*tasks)

    return results

if **name** == '**main**':
    asyncio.run(main())
.. code-block:: text




---

Summary: The Core Principles
----------------------------

1. The GIL Controls Everything
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Python's GIL:
|- Allows only ONE thread to execute Python bytecode at a time
|- Released during I/O operations (C library calls)
+- Not present in separate processes (each has own GIL)

Therefore:
|- CPU-bound + Threading = No parallelism (GIL bottleneck)
|- I/O-bound + Threading = Concurrency (GIL released during I/O)
+- CPU-bound + Multiprocessing = Parallelism (separate GILs)
::


2. Resource Usage Matters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text



Operation Type  |  Resource Bottleneck  |  Solution  |  Why?
----------------+----------------------+------------+------------------
CPU-bound       |  CPU cycles          |  Multi-    |  Bypass GIL,
                |  (computation)       |  processing|  use all cores
----------------+----------------------+------------+------------------
I/O-bound       |  Waiting for I/O     |  Asyncio/  |  GIL released,
(few ops)       |  (network, disk)     |  Threading |  work during wait
----------------+----------------------+------------+------------------
I/O-bound       |  Waiting for I/O     |  Asyncio   |  Lightweight,
(many ops)      |  + scalability       |            |  handles 1000s
.. code-block:: text




3. Trade-offs are Real
~~~~~~~~~~~~~~~~~~~~~~

**Multiprocessing**:
- Pros: True parallelism, bypasses GIL
- Cons: Memory overhead, slow startup, complex IPC
- **Use when**: CPU-bound work benefits > memory cost

**Threading**:
- Pros: Lightweight, easy data sharing, fast startup
- Cons: No speedup for CPU work, race conditions possible
- **Use when**: I/O-bound with moderate concurrency

**Asyncio**:
- Pros: Extremely lightweight, handles 100,000+ operations
- Cons: Requires async libraries, learning curve
- **Use when**: I/O-bound with high concurrency

4. Know Your Workload
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Profile your code first!
========================
import time

def profile_task(task_func):
    # CPU time (actual processing)
    cpu_start = time.process_time()
    # Wall time (including waiting)
    wall_start = time.perf_counter()

    result = task_func()

    cpu_time = time.process_time() - cpu_start
    wall_time = time.perf_counter() - wall_start

    if cpu_time / wall_time > 0.8:
        print("CPU-bound -> Use multiprocessing")
    else:
        print("I/O-bound -> Use asyncio/threading")

    return result
.. code-block:: text




---

Final Thoughts
--------------

The choice between multiprocessing, threading, and asyncio isn't about which is "better" - it's about matching the tool to the task:

- **Multiprocessing**: Powerful but heavy. Use when you need true parallel CPU computation.
- **Threading**: Simple and effective for I/O. Use when blocking libraries are needed.
- **Asyncio**: Lightweight and scalable for I/O. Use when you need to handle many concurrent operations.

Understanding the GIL and how Python interacts with the OS is key to making the right choice. Always profile your code, measure the results, and choose based on your specific requirements.
