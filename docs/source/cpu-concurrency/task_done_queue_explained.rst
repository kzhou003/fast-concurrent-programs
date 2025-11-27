task*done() and Queue Explained: Counter & Condition Variables
==============================================================

Quick Answer
------------

``task*done()`` is used to **track when you've finished processing an item**, so that ``queue.join()`` knows when ALL items have been processed.

**Yes, ``task*done()`` is on the Queue and does TWO things:**

1. **Decrements the task counter** - Tracks unfinished tasks
2. **Notifies the join() condition** - Wakes up ``queue.join()`` if counter reaches 0

---

The Problem Without task*done()
-------------------------------

Without task*done() - Can't Track Completion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from queue import Queue
from threading import Thread
import time

queue = Queue()

def producer():
    for i in range(3):
        queue.put(f"Task {i}")
        print(f"Produced Task {i}")

def consumer():
    for i in range(3):
        item = queue.get()
        print(f"Processing {item}")
        time.sleep(1)  # Simulate work
        print(f"Finished {item}")
        # NO task*done() here!

producer*thread = Thread(target=producer)
consumer*thread = Thread(target=consumer)

producer*thread.start()
consumer*thread.start()

producer*thread.join()
consumer*thread.join()

How do we know if ALL items have been processed?
================================================
WE DON'T! Consumer might still be working!
==========================================
::


With task*done() - Can Track Completion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from queue import Queue
from threading import Thread
import time

queue = Queue()

def producer():
    for i in range(3):
        queue.put(f"Task {i}")
        print(f"Produced Task {i}")

def consumer():
    for i in range(3):
        item = queue.get()
        print(f"Processing {item}")
        time.sleep(1)  # Simulate work
        print(f"Finished {item}")
        queue.task*done()  # Tell queue we're done!

producer*thread = Thread(target=producer)
consumer*thread = Thread(target=consumer)

producer*thread.start()
consumer*thread.start()

producer*thread.join()
consumer*thread.join()

queue.join()  # NOW we can safely wait for all items to be processed!
print("All tasks complete!")
::


---

How task*done() Works
---------------------

Internal Counter System
~~~~~~~~~~~~~~~~~~~~~~~

Queue maintains an **internal task counter**:

::

Initial state: task*counter = 0

put(item1)    → task*counter = 1
put(item2)    → task*counter = 2
put(item3)    → task*counter = 3

get()         → Returns item1 (counter still 3)
task*done()   → task*counter = 2 (decrement)

get()         → Returns item2 (counter still 2)
task*done()   → task*counter = 1 (decrement)

get()         → Returns item3 (counter still 1)
task*done()   → task*counter = 0 (decrement)

join()        → Returns immediately (counter = 0)
::


Visual Timeline
~~~~~~~~~~~~~~~

::

Producer         Queue           Consumer          Main
───────────────────────────────────────────────────────
put(1)          [1]
                counter=1

                              get() → gets 1
                              counter=1 (unchanged)
                              processing...
                              task*done()
                              counter=0
                                                  join()
                                                  BLOCKED
                                                  (counter=0)
                                                  CONTINUES!
::


---

What task*done() Actually Does
------------------------------

Code (Simplified)
~~~~~~~~~~~~~~~~~

.. code-block:: python

def task*done(self):
    with self.mutex:
        if self.unfinished*tasks <= 0:
            raise ValueError('task*done() called too many times')

        self.unfinished*tasks -= 1  # Decrement counter

        if self.unfinished*tasks == 0:
            self.all*tasks*done.notify*all()  # Wake up join()
::


Two Operations
~~~~~~~~~~~~~~

::

task*done() does:

1. Decrement counter
   ┌──────────────┐
   │ counter: 3   │
   └──────┬───────┘
          │ task*done()
          ▼
   ┌──────────────┐
   │ counter: 2   │ ← Just decremented
   └──────────────┘

2. Check if counter = 0, then notify all*tasks*done
   ┌──────────────┐
   │ counter: 1   │
   └──────┬───────┘
          │ task*done()
          ▼
   ┌──────────────┐
   │ counter: 0   │ ← All tasks done!
   │ Notify!      │ ← Wakes join()
   └──────────────┘
::


---

The Third Condition Variable: all*tasks*done
--------------------------------------------

Queue has **THREE** condition variables:

.. code-block:: python

class Queue:
    def _*init**(self):
        self.mutex = Lock()
        self.not*empty = Condition()      # put() signals, get() waits
        self.not*full = Condition()       # get() signals, put() waits
        self.all*tasks*done = Condition() # task*done() signals, join() waits
        self.items = []
        self.unfinished*tasks = 0
::


Timeline: All Three Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Queue Object:
┌─────────────────────────────────────┐
│ Condition: not*empty                │
│ ├─ Signaled by: put()              │
│ ├─ Waited on by: get()             │
│ └─ Purpose: "Data available"       │
│                                     │
│ Condition: not*full                 │
│ ├─ Signaled by: get()              │
│ ├─ Waited on by: put()             │
│ └─ Purpose: "Space available"      │
│                                     │
│ Condition: all*tasks*done           │
│ ├─ Signaled by: task*done()        │ ← YOUR QUESTION!
│ ├─ Waited on by: join()            │
│ └─ Purpose: "All work processed"   │
│                                     │
│ Counter: unfinished*tasks           │
│ ├─ Incremented by: put()           │
│ ├─ Decremented by: task*done()     │
│ └─ Checked by: join()              │
└─────────────────────────────────────┘
::


---

Complete Picture: All Operations
--------------------------------

put()
~~~~~
.. code-block:: python

def put(self, item):
    with self.not*full:
        while len(self.items) >= self.maxsize:
            self.not*full.wait()

        self.items.append(item)
        self.unfinished*tasks += 1  # ← Increment here!
        self.not*empty.notify()
::


get()
~~~~~
.. code-block:: python

def get(self):
    with self.not*empty:
        while not self.items:
            self.not*empty.wait()

        item = self.items.pop(0)
        self.not*full.notify()

        return item  # ← Returns just the item
::


task*done()
~~~~~~~~~~~
.. code-block:: python

def task*done(self):
    with self.mutex:
        self.unfinished*tasks -= 1  # ← Decrement here!

        if self.unfinished*tasks == 0:
            self.all*tasks*done.notify*all()  # ← Notify join()!
::


join()
~~~~~~
.. code-block:: python

def join(self):
    with self.all*tasks*done:
        while self.unfinished*tasks:
            self.all*tasks*done.wait()  # ← Waits here!
        # When counter reaches 0, task*done() wakes this up
::


---

The Complete Flow with All Three Conditions
-------------------------------------------

Scenario: 1 Producer, 1 Consumer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Time  Producer        get()           task*done()      Consumer Waits
──────────────────────────────────────────────────────────────────

0     put(1)
      ├─ items.append(1)
      ├─ unfinished=1
      └─ not*empty.notify()
                      │
                      ▼
                   get() wakes up
                   │
                   ├─ items.pop(0)
                   └─ not*full.notify()

1                                    Processing item...

2                                    task*done()
                                     ├─ unfinished=0
                                     └─ all*tasks*done.notify*all()
                                                         │
                                                         ▼
                                                    join() wakes up!
                                                    Returns!
::


---

Detailed Breakdown: task*done() with Counter
--------------------------------------------

Step 1: put() - Increment Counter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Initial state:
┌──────────────────────┐
│ unfinished*tasks: 0  │
│ items: []            │
└──────────────────────┘

put(1) called:
│
├─ Acquire mutex
├─ items.append(1)
├─ unfinished*tasks += 1  ← Counter incremented!
│
├─ State now:
│  ┌──────────────────────┐
│  │ unfinished*tasks: 1  │ ← Task added
│  │ items: [1]           │
│  └──────────────────────┘
│
├─ not*empty.notify()
└─ Release mutex
::


Step 2: get() - Item Removed, Counter Unchanged
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Before get():
┌──────────────────────┐
│ unfinished*tasks: 1  │ ← Still 1!
│ items: [1]           │
└──────────────────────┘

get() called:
│
├─ Acquire mutex
├─ while not items:
│   └─ not*empty.wait()  (not called, has items)
│
├─ item = items.pop(0)  ← Get the item
│
├─ State now:
│  ┌──────────────────────┐
│  │ unfinished*tasks: 1  │ ← STILL 1!
│  │ items: []            │   (not decremented by get!)
│  └──────────────────────┘
│
├─ not*full.notify()
└─ Release mutex & return 1
::


Step 3: task*done() - Decrement Counter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Before task*done():
┌──────────────────────┐
│ unfinished*tasks: 1  │
│ items: []            │
└──────────────────────┘

task*done() called:
│
├─ Acquire mutex
├─ if unfinished*tasks <= 0:
│   └─ raise ValueError (not true here)
│
├─ unfinished*tasks -= 1  ← Counter decremented!
│
├─ if unfinished*tasks == 0:
│   ├─ YES! (now 0)
│   └─ all*tasks*done.notify*all()  ← Signal join()!
│
├─ State now:
│  ┌──────────────────────┐
│  │ unfinished*tasks: 0  │ ← NOW 0!
│  │ items: []            │
│  └──────────────────────┘
│
└─ Release mutex
::


Step 4: join() - Wait, Then Return
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

join() called (from main thread):

├─ Acquire all*tasks*done condition
│
├─ while unfinished*tasks != 0:
│   └─ all*tasks*done.wait()  (waiting...)
│       [Paused, waiting for task*done()]
│
[task*done() is called from consumer thread]
[unfinished*tasks becomes 0]
[task*done() calls notify*all()]
│
├─ Woken up! Check condition again
│
├─ while unfinished*tasks != 0:
│   └─ False! (unfinished*tasks = 0)
│
├─ Continue past the loop
│
└─ Return from join()
   Main thread can continue!
::


---

The Counter Lifecycle
---------------------

Diagram: Tracking One Task
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Lifecycle of Task #1:

put(task1)
  ↓
unfinished*tasks = 1
  ↓
get() → task1
  ↓
unfinished*tasks = 1 (unchanged)
  ↓
[Processing task1...]
  ↓
task*done()
  ↓
unfinished*tasks = 0 ← Counter decremented!
  ↓
all*tasks*done.notify*all()
  ↓
join() wakes up (if counter = 0)
::


Diagram: Multiple Tasks
~~~~~~~~~~~~~~~~~~~~~~~

::

Multiple tasks lifecycle:

put(t1) → unfinished=1
put(t2) → unfinished=2
put(t3) → unfinished=3

get()→t1, get()→t2, get()→t3
      unfinished=3 (unchanged)

[Processing all...]

task*done()  → unfinished=2
task*done()  → unfinished=1
task*done()  → unfinished=0 ← Reached 0!
              all*tasks*done.notify*all()

join() was waiting...
Now wakes up! ← Can continue
::


---

Key Insight: Two Data Flows
---------------------------

::

Queue has TWO independent data flows:

1. ITEM FLOW:
   put() → [items list] → get()
   Signal: not*empty (consumer waits for items)
   Signal: not*full (producer waits for space)

2. COMPLETION FLOW:
   put() → [unfinished*tasks counter] → task*done()
   Signal: all*tasks*done (join() waits for completion)

These are INDEPENDENT:
- get() doesn't care if task*done() was called
- task*done() doesn't remove items from list
- They work in parallel
::


---

Why BOTH Counter AND Condition Variable?
----------------------------------------

Counter
~~~~~~~
.. code-block:: python

unfinished*tasks: int

Tracks how many items haven't been marked done yet
==================================================
Used to know WHEN to signal
===========================
Checked by join()
=================
::


Condition Variable
~~~~~~~~~~~~~~~~~~
.. code-block:: python

all*tasks*done: Condition()

Efficiently wakes up waiting threads
====================================
Avoids busy-waiting
===================
Used by join() to sleep instead of spinning
===========================================
::


Together
~~~~~~~~
.. code-block:: python

def task*done(self):
    with self.mutex:
        self.unfinished*tasks -= 1  # ← Update counter

        if self.unfinished*tasks == 0:  # ← Check counter
            self.all*tasks*done.notify*all()  # ← Use condition
::


**Counter answers "when?"**
**Condition variable answers "how to signal efficiently?"**

Without all*tasks*done Condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If Queue only had counter but NO condition variable:

.. code-block:: python

WRONG - Wastes CPU
==================
def join*without*condition(self):
    while self.unfinished*tasks > 0:
        time.sleep(0.01)  # Busy-wait, bad!
        # Keeps waking up to check, wastes CPU

RIGHT - Uses condition variable
===============================
def join*with*condition(self):
    with self.all*tasks*done:
        while self.unfinished*tasks > 0:
            self.all*tasks*done.wait()  # Sleep efficiently
            # Only wakes up when task*done() signals
::


---

Visual: Counter and Condition Together
--------------------------------------

::

Queue internals:

┌────────────────────────────────────────┐
│ unfinished*tasks Counter               │
│ ┌─────────────────────────────────────┐│
│ │ Value: 3                             ││
│ │ - Incremented by put()              ││
│ │ - Decremented by task*done()        ││
│ │ - Checked by join()                 ││
│ └─────────────────────────────────────┘│
│                                        │
│ all*tasks*done Condition Variable      │
│ ┌─────────────────────────────────────┐│
│ │ Waiting threads: [join*thread]      ││
│ │ - Signaled by task*done()           ││
│ │ - Waited on by join()               ││
│ │ - Notifies when counter = 0         ││
│ └─────────────────────────────────────┘│
└────────────────────────────────────────┘
::


---

Complete Operation Table
------------------------

| Operation | What it touches | Effect |
|-----------|-----------------|--------|
| ``put(item)`` | items list | Appends item |
| ``put(item)`` | unfinished*tasks | Increments |
| ``put(item)`` | not*empty | Notifies consumer |
| ``get()`` | items list | Pops item |
| ``get()`` | unfinished*tasks | No change! |
| ``get()`` | not*full | Notifies producer |
| ``task*done()`` | unfinished*tasks | Decrements |
| ``task*done()`` | all*tasks*done | Notifies join() |
| ``join()`` | unfinished*tasks | Checks if 0 |
| ``join()`` | all*tasks*done | Waits on it |

---

What Happens WITHOUT task*done()
--------------------------------

Queue.join() Without task*done()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

queue = Queue()

queue.put(1)
queue.put(2)
queue.put(3)

item = queue.get()  # Get all items
item = queue.get()
item = queue.get()

NO task*done() calls!
=====================

queue.join()  # BLOCKS FOREVER!
              # Queue still thinks 3 items are "unprocessed"
              # Counter never decrements to 0
::


Why It Blocks Forever
~~~~~~~~~~~~~~~~~~~~~

::

Queue state:
┌─────────────────────────────┐
│ items: []                   │  ← All removed
│ task*counter: 3             │  ← Still 3! Never decremented
└─────────────────────────────┘

join() checks: if task*counter != 0: wait()
              3 != 0, so WAIT FOREVER
::


---

What Happens WITH task*done()
-----------------------------

Queue.join() With task*done()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

queue = Queue()

queue.put(1)      # task*counter = 1
queue.put(2)      # task*counter = 2
queue.put(3)      # task*counter = 3

item = queue.get()  # task*counter = 3 (unchanged)
queue.task*done()   # task*counter = 2 (decremented!)

item = queue.get()  # task*counter = 2 (unchanged)
queue.task*done()   # task*counter = 1 (decremented!)

item = queue.get()  # task*counter = 1 (unchanged)
queue.task*done()   # task*counter = 0 (decremented!)

queue.join()  # task*counter = 0, so continues immediately!
print("Done!")
::


Why It Works
~~~~~~~~~~~~

::

Queue state after all task*done():
┌─────────────────────────────┐
│ items: []                   │  ← All removed
│ task*counter: 0             │  ← All decremented to 0
└─────────────────────────────┘

join() checks: if task*counter != 0: wait()
              0 == 0, so CONTINUE (don't wait)
::


---

The Three-Step Cycle
--------------------

Step 1: Put Item
~~~~~~~~~~~~~~~~
.. code-block:: python

queue.put(item)
task*counter increments
=======================
::


Step 2: Get and Process
~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

item = queue.get()  # Removes from queue but...
task*counter STAYS the same
===========================
(item is "in flight", being processed)
======================================
process(item)       # Do actual work
::


Step 3: Mark Done
~~~~~~~~~~~~~~~~~
.. code-block:: python

queue.task*done()
task*counter decrements
=======================
Tells queue: "I'm done with this item"
======================================
::


Flow Diagram
~~~~~~~~~~~~

::

put(A)       get()      task*done()    queue state
  ↓            ↓            ↓
[A]          (processing)   ✓          counter: 1→0
  ↓            ↓            ↓
counter=1   counter=1   counter=0
::


---

Real Example: Work Queue Pattern
--------------------------------

Without task*done() - PROBLEMATIC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from queue import Queue
from threading import Thread

tasks = Queue()

Add tasks
=========
for i in range(5):
    tasks.put(f"Task {i}")

def worker():
    while True:
        task = tasks.get()
        if task is None:
            break
        print(f"Working on {task}")
        # Do work here...
        # NO task*done()!

Start worker
============
t = Thread(target=worker)
t.start()

Main thread wants to know when all work is done
===============================================
tasks.join()  # BLOCKS FOREVER!
print("All done!")  # Never prints!

Send poison pill to stop worker
===============================
tasks.put(None)
t.join()
::


**Output:**
::

Working on Task 0
Working on Task 1
Working on Task 2
Working on Task 3
Working on Task 4
[Program hangs, never says "All done!"]
::


With task*done() - CORRECT
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from queue import Queue
from threading import Thread

tasks = Queue()

Add tasks
=========
for i in range(5):
    tasks.put(f"Task {i}")

def worker():
    while True:
        task = tasks.get()
        if task is None:
            break
        print(f"Working on {task}")
        # Do work here...
        tasks.task*done()  # IMPORTANT!

Start worker
============
t = Thread(target=worker)
t.start()

Main thread waits for all work to be done
=========================================
tasks.join()  # Returns when all task*done() called
print("All done!")  # Now prints!

Send poison pill to stop worker
===============================
tasks.put(None)
t.join()
::


**Output:**
::

Working on Task 0
Working on Task 1
Working on Task 2
Working on Task 3
Working on Task 4
All done!
::


---

Why Is This Useful?
-------------------

Use Case 1: Verify All Work Complete
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Main thread
===========
for task in tasks:
    queue.put(task)

Wait for workers to finish ALL tasks
====================================
queue.join()

NOW we know all work is done
============================
print("All tasks processed successfully!")
save*results()
shutdown*workers()
::


Use Case 2: Track Progress
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

queue = Queue()
for i in range(100):
    queue.put(i)

def worker():
    while True:
        task = queue.get()
        if task is None:
            break
        process(task)
        queue.task*done()

workers = [Thread(target=worker) for * in range(4)]
for w in workers:
    w.start()

Wait for completion
===================
queue.join()
print(f"All 100 tasks completed!")
::


Use Case 3: Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

batch*queue = Queue()

def add*to*batch(item):
    batch*queue.put(item)

def process*batch():
    batch = []
    while True:
        item = batch*queue.get()
        if item is None:
            process*batch*now(batch)
            break

        batch.append(item)
        batch*queue.task*done()

        if len(batch) == 10:
            process*batch*now(batch)
            batch = []

Main
====
for i in range(50):
    add*to*batch(i)

Wait until all batches processed
================================
batch*queue.join()
print("All batches processed!")
::


---

Comparison: With vs Without task*done()
---------------------------------------

Scenario: Main thread needs to know when workers finish
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Without task*done() | With task*done() |
|-------------------|-----------------|
| ``queue.join()`` blocks forever | ``queue.join()`` returns when done |
| Can't track progress | Can track completion |
| Unpredictable behavior | Predictable, reliable |
| No way to know if work finished | Clear completion signal |

---

The Flow in Your Code (05*queue.py)
-----------------------------------

Looking back at the code:

.. code-block:: python

class Consumer(Thread):
    def _*init**(self, queue):
        Thread.**init**(self)
        self.queue = queue

    def run(self):
        while True:
            item = self.queue.get()           # Step 1: Remove from queue
            print(f"consumer: got {item}")    # Step 2: Do work
            self.queue.task*done()            # Step 3: Mark as done
::


Timeline
~~~~~~~~

::

Producer puts 5 items in 5 seconds
├─ 0s: put(42)   → counter = 1
├─ 1s: put(7)    → counter = 2
├─ 2s: put(199)  → counter = 3
├─ 3s: put(88)   → counter = 4
└─ 4s: put(12)   → counter = 5

Meanwhile, 3 consumers get and process:
Consumer1: get(42) → task*done() → counter = 4
Consumer2: get(7) → task*done() → counter = 3
Consumer3: get(199) → task*done() → counter = 2
Consumer1: get(88) → task*done() → counter = 1
Consumer2: get(12) → task*done() → counter = 0

Main thread: producer.join() completes at 4s
Main thread: But consumers are in infinite loop (while True)
Main thread: Tries to join consumers (would block forever)
::


---

The Missing Piece in the Code
-----------------------------

The code has an issue - consumers have ``while True`` (infinite loop):

.. code-block:: python

def run(self):
    while True:  # Never exits!
        item = self.queue.get()
        print(f"consumer notify: item popped from queue by {item, self.name}")
        self.queue.task*done()
::


Better Code Pattern
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

def run(self):
    while True:
        item = self.queue.get()

        if item is None:  # Poison pill - signal to exit
            self.queue.task*done()
            break

        print(f"consumer: processing {item}")
        self.queue.task*done()

In main
=======
producer.join()
queue.join()  # All items processed

Stop consumers
==============
for * in range(3):
    queue.put(None)  # Poison pill

consumer1.join()
consumer2.join()
consumer3.join()
::


---

Summary: What task*done() Does
------------------------------

**YES, task*done() is on Queue and does:**

1. **Decrements unfinished*tasks counter** by 1
   - Tracks how many items still need processing

2. **Checks if counter reached 0**
   - If yes: ALL items processed

3. **Notifies all*tasks*done condition variable** (if counter = 0)
   - Wakes up any threads waiting on ``join()``
   - Uses ``notify*all()`` to wake ALL waiters

**The counter tracks WHAT, the condition signals HOW.**

Why We Need task*done()
~~~~~~~~~~~~~~~~~~~~~~~

1. **Tracks Completion** - Queue knows when items are processed
2. **Enables join()** - ``queue.join()`` knows when to return
3. **Prevents Hanging** - Without it, ``join()`` blocks forever
4. **Progress Tracking** - Count which items are done
5. **Synchronization** - Coordinate main thread with workers

The Pattern
~~~~~~~~~~~

::

put(item)      → "Item added to queue"
       ↓
get(item)      → "Someone took the item"
       ↓
task*done()    → "Item was PROCESSED"
       ↓
join()         → "All items PROCESSED?" → Yes → Continue
::


**Without ``task*done()``, ``join()`` can never verify all items are truly processed!**

---

Real Code Example
-----------------

.. code-block:: python

from queue import Queue
from threading import Thread
import time

q = Queue()

def producer():
    for i in range(3):
        q.put(f"Task {i}")
        print(f"Put Task {i}, unfinished*tasks now = {q.unfinished*tasks}")

def consumer():
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Got {item}, unfinished*tasks still = {q.unfinished*tasks}")
        time.sleep(0.5)
        q.task*done()
        print(f"Marked {item} done, unfinished*tasks now = {q.unfinished*tasks}")

t1 = Thread(target=producer)
t2 = Thread(target=consumer)

t1.start()
t2.start()

t1.join()

Now wait for consumer to finish processing ALL items
====================================================
print("Waiting for all tasks to be processed...")
q.join()  # Waits here until counter reaches 0
print("All tasks processed!")

q.put(None)  # Stop consumer
t2.join()
::


**Output:**
::

Put Task 0, unfinished*tasks now = 1
Put Task 1, unfinished*tasks now = 2
Put Task 2, unfinished*tasks now = 3
Got Task 0, unfinished*tasks still = 3
Marked Task 0 done, unfinished*tasks now = 2
Got Task 1, unfinished*tasks still = 2
Marked Task 1 done, unfinished*tasks now = 1
Got Task 2, unfinished*tasks still = 1
Marked Task 2 done, unfinished*tasks now = 0
Waiting for all tasks to be processed...
All tasks processed!
::


Notice how ``task_done()`` decrements the counter, and when it reaches 0, ``join()`` returns!
