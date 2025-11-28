Threading Basics: start() and join()

What is start()?

``start()`` **launches the thread and begins execution**. It's the method that actually creates and starts running the thread.

Key Points About start()
~~~~~~~~~~~~~~~~~~~~~~~~

- **It doesn't run the function immediately** - Instead, it schedules the thread to run
- **The main thread continues immediately** - It doesn't wait for the worker thread to finish
- **Without start(), the thread never runs** - Creating a Thread object does nothing by itself
- **Can only call start() once** - Calling it twice raises an error

Example
~~~~~~~

.. code-block:: python


def worker():
    print("Worker running")

Create thread object (doesn't start yet)
thread = threading.Thread(target=worker)

Call start() to actually begin execution
thread.start()

print("Main continues here")
::


**Output:**
::

Worker running
Main continues here
::


Notice: Both print statements execute, showing that main doesn't wait.

---

What is join()?

``join()`` **makes the main thread wait until the worker thread finishes**. It's a blocking operation.

Key Points About join()
~~~~~~~~~~~~~~~~~~~~~~~

- **The main thread blocks/pauses** at the join() call
- **Waits for the worker thread to complete** before continuing
- **Without join(), main might exit before worker finishes** (and the work is lost)
- **You can join() multiple threads** to wait for all of them

Example
~~~~~~~

.. code-block:: python

import time

def worker():
    print("Worker starting")
    time.sleep(1)
    print("Worker done")

thread = threading.Thread(target=worker)
thread.start()

print("Main: waiting for worker...")
thread.join()  # Main thread BLOCKS here
print("Main: worker finished, I can continue")
::


**Output:**
::

Worker starting
Main: waiting for worker...
Worker done
Main: worker finished, I can continue
.. code-block:: text





Comparison: With vs Without join()

WITHOUT join() - DANGER!
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

import time

def download_file():
    print("  Download starting...")
    time.sleep(2)
    print("  Download complete!")

thread = threading.Thread(target=download_file)
thread.start()

print("Download started! I'm moving on...")
print("Program exiting")
Program exits before download finishes!
Download is interrupted!
::


**Output:**
::

Download started! I'm moving on...
Program exiting
::


The download never completes because the program exits!

WITH join() - CORRECT!
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

import time

def download_file():
    print("  Download starting...")
    time.sleep(2)
    print("  Download complete!")

thread = threading.Thread(target=download_file)
thread.start()

print("Download started! Waiting...")
thread.join()  # Main blocks here until download finishes
print("Download is done! Safely exiting")
::


**Output:**
::

Download started! Waiting...
  Download starting...
  Download complete!
Download is done! Safely exiting
::


Now the download completes before the program exits.

---

Thread Lifecycle Diagram

.. code-block:: text


    | Thread Object Created                                   |
    | thread = Thread(target=worker_func)                     |
    | (Thread exists but doesn't run)                         |
    | thread.start()
    [v]
    | Thread Running (in parallel with main)                  |
    | - Main thread continues immediately                     |
    | - Worker thread executes target function               |
    | - Both run concurrently                                |
    | Worker thread finishes
    [v]
    | Thread Finished                                         |
    | (if no join(), main might exit before here)            |
    | (with join(), main waits here)                         |





---

Typical Pattern for Multiple Threads

The standard pattern for managing multiple threads:

.. code-block:: python


Step 1: Create all threads
threads = []
for task in tasks:
    thread = threading.Thread(target=do_task, args=(task,))
    thread.start()      # Start each thread
    threads.append(thread)

Step 2: Wait for all to finish
for thread in threads:
    thread.join()       # Wait for each one

print("All tasks complete!")
::


This ensures:
- All work starts in parallel
- Main waits for all work to complete
- You only continue when everything is done

---

Common Mistakes

Mistake 1: Calling the function directly instead of start()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

thread = threading.Thread(target=worker)
worker()  # This blocks main! Not parallel!

[[OK]] CORRECT - runs worker in NEW thread
thread = threading.Thread(target=worker)
thread.start()  # Non-blocking, parallel execution
::


Mistake 2: Forgetting join()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

thread = threading.Thread(target=save_to*database)
thread.start()
print("Saved!")  # Exits immediately, database never saves

[[OK]] CORRECT - wait for worker to finish
thread = threading.Thread(target=save_to*database)
thread.start()
thread.join()  # Wait for it
print("Saved!")  # Now safe to continue
::


Mistake 3: Thinking threads share data automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

counter = 0
def increment():
    global counter
    counter += 1  # Not atomic! Can have race condition

threads = [threading.Thread(target=increment) for * in range(100)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(counter)  # Might not be 100!

[[OK]] SAFE - use lock
counter = 0
lock = threading.Lock()
def increment():
    global counter
    with lock:  # Acquire lock
        counter += 1  # Now safe
        # Release lock automatically

threads = [threading.Thread(target=increment) for * in range(100)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(counter)  # Guaranteed to be 100
.. code-block:: text





Practical Example: Web Scraper

.. code-block:: python

import time

def fetch_url(url, results):
    """Fetch a URL and store result."""
    print(f"  Fetching {url}...")
    time.sleep(1)  # Simulate network request
    results.append(f"Data from {url}")
    print(f"  Done: {url}")

URLs to fetch
urls = ["http://example.com/1", "http://example.com/2", "http://example.com/3"]
results = []

Create and start threads
threads = []
for url in urls:
    thread = threading.Thread(target=fetch_url, args=(url, results))
    thread.start()      # Launch the thread
    threads.append(thread)

print(f"Started fetching {len(threads)} URLs...")

Wait for all to complete
for thread in threads:
    thread.join()       # Wait for each thread

print(f"All done! Results: {results}")
::


**Output:**
::

Started fetching 3 URLs...
  Fetching http://example.com/1...
  Fetching http://example.com/2...
  Fetching http://example.com/3...
  Done: http://example.com/1
  Done: http://example.com/2
  Done: http://example.com/3
All done! Results: ['Data from http://example.com/1', ...]
::


All three URLs fetched in parallel. Without ``join()``, the program would exit before results were ready.

---

Summary Table

| Method | What it does | When to use | Important notes |
| ``start()`` | Launches the thread | Always, right after creating Thread object | Can only call once per thread; doesn't block |
| ``join()`` | Waits for thread to finish | Always, before assuming work is done | Blocks execution; use with multiple threads for parallel work |
| Neither | Thread created but never runs | Never (waste of memory) | You MUST call start() or thread does nothing |

---

Key Takeaways

1. **start() launches**, **join() waits**
2. **Without start()** - thread never runs (dead code)
3. **Without join()** - main exits before worker finishes (lost work)
4. **Proper pattern** - start all threads, then join all threads
5. **Always use join()** before assuming work is complete
6. **Threads run in parallel** - multiple tasks execute simultaneously
7. **Be careful with shared data** - use locks to prevent race conditions
