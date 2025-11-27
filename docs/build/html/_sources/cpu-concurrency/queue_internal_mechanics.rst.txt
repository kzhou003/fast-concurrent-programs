Queue Internal Mechanics: Condition Variables & Events
======================================================

The Answer
----------

**No, the item itself doesn't get a condition/event with it.**

Instead, the **Queue object maintains condition variables** that signal **all waiting threads** when items are added or removed.

---

Queue's Internal Structure
--------------------------

When you create a Queue, here's what's inside:

.. code-block:: python

from queue import Queue

q = Queue()

Inside the Queue object:
========================
┌─────────────────────────────────────────┐
===========================================
│ Queue Internal State                    │
===========================================
├─────────────────────────────────────────┤
===========================================
│ mutex: Lock()                           │
===========================================
│   - Protects all modifications          │
===========================================
│                                         │
===========================================
│ not*empty: Condition()                  │
===========================================
│   - Signals when items added            │
===========================================
│   - Consumers wait on this              │
===========================================
│                                         │
===========================================
│ not*full: Condition()                   │
===========================================
│   - Signals when items removed          │
===========================================
│   - Producers wait on this              │
===========================================
│                                         │
===========================================
│ items: []                               │
===========================================
│   - Actual items (without conditions)   │
===========================================
│                                         │
===========================================
│ task*counter: 0                         │
===========================================
│   - Tracks unfinished tasks             │
===========================================
└─────────────────────────────────────────┘
===========================================
::


The Items Don't Have Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

When you put an item
====================
q.put(42)

Inside queue:
=============
items = [42]  ← Just the number, no condition attached
======================================================
not*empty condition wakes up consumers
======================================
not*full condition might wake up producers
==========================================
::


The **item** is just data. The **conditions** belong to the **Queue**, not the item.

---

How Conditions Work in Queue
----------------------------

Condition Variables Explained
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A **Condition variable** is like a **notification system**:
- Threads can **wait** on a condition
- Threads can **notify** waiting threads

::

Condition Variable (not*empty):
┌────────────────────────────────┐
│ Waiting Threads Queue:         │
│ [Consumer1] [Consumer2] [C3]   │ ← Sleeping, waiting for items
└────────────────────────────────┘
         │
         │ when put() is called
         │ notify() is called
         ▼
One waiting thread wakes up!
::


---

put() - What Actually Happens
-----------------------------

Step-by-Step Execution
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

q.put(item)
::


**Internal Queue code:**

.. code-block:: python

def put(self, item):
    # 1. Acquire the mutex (lock)
    with self.mutex:

        # 2. Wait if queue is full
        while len(self.items) >= self.maxsize:
            self.not*full.wait(self.mutex)
            # Thread blocks here, waiting for space
            # Releases mutex while waiting

        # 3. Queue has space now, add item
        self.items.append(item)
        # items = [42]  ← Item added (no condition on item)

        # 4. Notify ONE consumer that item is available
        self.not*empty.notify()
        # Wakes one consumer waiting on not*empty

    # 5. Mutex released here (with block ends)
::


Timeline: put() with Condition Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Time  Queue                    Condition Variables    Consumers
────────────────────────────────────────────────────────────────
0     put(42)                                         waiting...
      acquire mutex

1                              not*empty signal       WAKE UP!
      items.append(42)

2     notify()

3                                                     get() continues
      release mutex

4     (done)
::


---

get() - What Actually Happens
-----------------------------

Step-by-Step Execution
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

item = q.get()
::


**Internal Queue code:**

.. code-block:: python

def get(self):
    # 1. Acquire the mutex (lock)
    with self.mutex:

        # 2. Wait if queue is empty
        while len(self.items) == 0:
            self.not*empty.wait(self.mutex)
            # Thread blocks here, waiting for items
            # Releases mutex while waiting
            # When notified, reacquires mutex and continues

        # 3. Queue has items, remove one
        item = self.items.pop(0)
        # Just the value, no condition with it

        # 4. Notify ONE producer that space is available
        self.not*full.notify()
        # Wakes one producer waiting on not*full

        # 5. Return the item (just the value)
        return item

    # 6. Mutex released here (with block ends)
::


Timeline: get() with Condition Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Time  Queue                    Condition Variables    Producers
────────────────────────────────────────────────────────────────
0     get()                                           put() blocked
      acquire mutex                                   (queue full)

1     items = [1,2,3]
      items.pop(0) → returns 1  not*full signal       WAKE UP!

2     notify()

3     release mutex                                   put() continues
      return 1

4     (done)
::


---

Visual: No Condition On Items
-----------------------------

::

Queue Structure:

┌──────────────────────────────────────────────────┐
│ Queue                                            │
│                                                  │
│  not*empty Condition Variable                   │
│  ┌─────────────────────────────────────────┐   │
│  │ Waiting Consumers: [C1] [C2] [C3]       │   │
│  │ (sleeping, blocked on not*empty.wait()) │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
│  Items List (just data, no conditions):         │
│  ┌─────────────────────────────────────────┐   │
│  │ [42]  [7]  [199]  [88]                  │   │
│  │  ↑                                       │   │
│  │  └─ Just numbers, no events attached    │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
│  not*full Condition Variable                    │
│  ┌─────────────────────────────────────────┐   │
│  │ Waiting Producers: [P1]                 │   │
│  │ (sleeping, blocked on not*full.wait())  │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
└──────────────────────────────────────────────────┘

KEY: The conditions are properties of the QUEUE,
     not attached to individual items!
::


---

Complete put() and get() Timeline
---------------------------------

Scenario: Queue with maxsize=2, multiple producers/consumers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Initial:
Queue: []
not*empty: [Consumer1, Consumer2]  ← Waiting for items
not*full: []                       ← No one waiting

Time 0: Producer1.put(42)
├─ Acquire mutex
├─ Queue not full, add item
├─ items = [42]
├─ Call not*empty.notify()
│  └─ Consumer1 wakes up!
├─ Release mutex
└─ (put done)

Queue: [42]
not*empty: [Consumer2]  ← Consumer1 woke up
not*full: []

Time 1: Producer2.put(7)
├─ Acquire mutex
├─ Queue not full, add item
├─ items = [42, 7]
├─ Call not*empty.notify()
│  └─ Consumer2 wakes up!
├─ Release mutex
└─ (put done)

Queue: [42, 7]  ← FULL (maxsize=2)
not*empty: []   ← All consumers woke
not*full: []

Time 2: Producer3.put(199)
├─ Acquire mutex
├─ Queue IS FULL! len(items)=2 >= maxsize=2
├─ Call not*full.wait(mutex)
│  └─ Producer3 BLOCKS here
│  └─ Releases mutex
└─ (waiting for space)

Queue: [42, 7]
not*empty: []
not*full: [Producer3]  ← Waiting for space

Time 3: Consumer1.get()
├─ Acquire mutex
├─ items = [42, 7], not empty
├─ item = items.pop(0) = 42  ← Returns just 42, no condition
├─ Call not*full.notify()
│  └─ Producer3 wakes up!
├─ Release mutex
└─ Return 42

Queue: [7]
not*empty: []
not*full: []  ← Producer3 woke up

Time 4: Producer3 continues from where it blocked
├─ Acquire mutex (had released it, now reacquires)
├─ Queue not full anymore, add item
├─ items = [7, 199]
├─ Call not*empty.notify() (no one waiting)
├─ Release mutex
└─ (put done)

Queue: [7, 199]  ← FULL again
::


---

Key Insight: Conditions Are On Queue, Not Items
-----------------------------------------------

What People Might Think (WRONG):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

❌ WRONG MENTAL MODEL
====================
q.put(42)  # Does the item 42 carry an event/condition with it?
item = q.get()  # Does the returned item have a condition?
::


The Actual Truth:
~~~~~~~~~~~~~~~~~

.. code-block:: python

✓ CORRECT MENTAL MODEL
======================
q.put(42)  # 42 goes into queue
           # Queue's not*empty condition is signaled
           # The number 42 itself has NO condition

item = q.get()  # 42 is returned as plain data
                # Queue's not*full condition is signaled
                # The number 42 returned has NO condition
::


---

Why Queue Uses Shared Conditions
--------------------------------

Instead of Per-Item Events
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

❌ This would be inefficient (if items had events)
=================================================
class BadQueue:
    def put(self, item):
        event = Event()
        self.items.append((item, event))
        # Every item carries its own event - wastes memory!

    def get(self):
        item, event = self.items.pop(0)
        # What do we do with the event?
        # Consumer doesn't care about one item's event
        return item

✓ This is efficient (what Queue actually does)
==============================================
class GoodQueue:
    def **init**(self):
        self.items = []
        self.not*empty = Condition()  # Single condition for ALL items

    def put(self, item):
        self.items.append(item)  # Item is just data
        self.not*empty.notify()  # Signal all waiting consumers

    def get(self):
        item = self.items.pop(0)  # Item is just data
        return item
::


---

How Conditions Work: The Mechanism
----------------------------------

not*empty.notify() - What It Does
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

When put() is called
====================
q.put(42)

Inside put():
=============
self.not*empty.notify()

This tells the condition variable:
==================================
"Someone just added an item"
============================
#
The condition variable wakes ONE waiting thread
===============================================
(the one sleeping on not*empty.wait())
======================================
#
That thread checks: "Is there data now?"
========================================
If yes, it continues
====================
If no, it goes back to sleep
============================
::


Diagram: Condition Variable Notification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

BEFORE put():
┌─────────────────────────┐
│ not*empty Condition     │
│ Waiting threads:        │
│ [Consumer1]             │ ← Sleeping, waiting for items
│ [Consumer2]             │
│ [Consumer3]             │
│ Items: []               │
└─────────────────────────┘

put(42) called:
│
├─ items.append(42)
│
├─ not*empty.notify()
│  └─ Sends signal!
│
└─ (mutex released)

AFTER notify():
┌─────────────────────────┐
│ not*empty Condition     │
│ Waiting threads:        │
│ [Consumer1] ← WOKEN UP! │
│ [Consumer2]             │
│ [Consumer3]             │
│ Items: [42]             │
└─────────────────────────┘

Consumer1 wakes up:
├─ Reacquires mutex
├─ Checks: is items not empty? YES!
├─ Proceeds to pop item
└─ (gets the 42)
::


---

Back to Your Original Question
------------------------------

"Does each item get a condition?"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**No.** Here's why:

1. **Items are just data** - 42, "hello", [1,2,3], etc.
2. **Queue has conditions** - not*empty, not*full
3. **Conditions signal all threads** - not individual items
4. **More efficient** - One pair of conditions for all items
5. **Correct semantics** - Consumer doesn't care which item, just needs one

The Flow
~~~~~~~~

::

put(42)
  │
  └─ Add 42 to items list
  └─ Signal not*empty
     └─ Wakes one consumer

get()
  │
  └─ Wait on not*empty (if empty)
  └─ Remove 42 from items
  └─ Return 42 (just the value, no condition)
  └─ Signal not*full
     └─ Wakes one producer
::


---

Real Code Example
-----------------

What Queue.put() Actually Does
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from threading import Lock, Condition

class MyQueue:
    def _*init**(self, maxsize=0):
        self.mutex = Lock()
        self.not*empty = Condition(self.mutex)
        self.not*full = Condition(self.mutex)
        self.items = []
        self.maxsize = maxsize
        self.unfinished*tasks = 0

    def put(self, item):
        with self.not*full:  # Acquire lock via condition
            # Wait if full
            while len(self.items) >= self.maxsize and self.maxsize:
                self.not*full.wait()

            # Add item (just the value, NO condition)
            self.items.append(item)
            self.unfinished*tasks += 1

            # Notify one consumer
            self.not*empty.notify()

    def get(self):
        with self.not*empty:  # Acquire lock via condition
            # Wait if empty
            while not self.items:
                self.not*empty.wait()

            # Remove and return (just the value)
            item = self.items.pop(0)

            # Notify one producer
            self.not*full.notify()

            return item

    def task*done(self):
        with self.mutex:
            self.unfinished*tasks -= 1
            if self.unfinished*tasks == 0:
                # All tasks done, could notify here
                pass
::


---

Summary
-------

| Question | Answer |
|----------|--------|
| **Do items have conditions?** | No, items are just data |
| **Where are conditions?** | In the Queue object itself |
| **How many conditions?** | Two: not*empty and not*full |
| **What do conditions do?** | Signal all waiting threads when state changes |
| **Is condition per-item?** | No, shared for all items |
| **When does notification happen?** | Every put() and get() |
| **Who receives notification?** | One waiting thread (if any) |

---

Key Takeaways
-------------

1. **Items are just data** - No conditions attached
2. **Queue owns the conditions** - not*empty and not*full
3. **Conditions are shared** - Signal all consumers/producers
4. **Efficient design** - One pair of conditions for unlimited items
5. **put() signals not*empty** - "Item added, consumers can proceed"
6. **get() signals not*full** - "Space freed, producers can proceed"
7. **Thread-safe** - Mutex protects all modifications
8. **No per-item overhead** - Conditions don't travel with items

The magic is that Queue uses **shared conditions** at the Queue level, not individual conditions per item. This is more efficient and achieves perfect synchronization!
