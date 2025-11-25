# Queue Explained: Thread-Safe Data Structure & Locking

## What is a Queue?

A **queue** is a **First-In-First-Out (FIFO)** data structure with **built-in thread safety**. It handles synchronization automatically, so you don't have to manage locks yourself.

### Key Concept
```
put(item)      get()
   │             │
   ▼             ▼
┌─────────────────────────┐
│  [1] [2] [3] [4] [5]    │
│   ▲                      │
│  Item added here    Item removed here
```

- Items added to the **back**
- Items removed from the **front**
- **FIFO**: First item in is first item out

---

## How Queue Works Internally

### Internal Structure

```python
from queue import Queue

q = Queue()
```

Queue internally looks like this:

```
Queue object:
┌─────────────────────────────────────┐
│ items: [1, 2, 3, 4, 5]             │ ← Internal list
│ mutex: Lock()                       │ ← Built-in lock!
│ not_empty: Condition()              │ ← Signals when items available
│ not_full: Condition()               │ ← Signals when space available
│ maxsize: 0 (unlimited)              │
└─────────────────────────────────────┘
```

### Automatic Locking

When you call `put()` or `get()`, the Queue **automatically**:
1. **Acquires the lock** - Only one thread can access at a time
2. **Modifies the list** - Safely adds/removes items
3. **Releases the lock** - Other threads can now access
4. **Signals waiting threads** - Wakes up threads waiting for data/space

---

## Queue Operations

### `put(item)`
- **Adds item to the back**
- **Blocks if queue is full** (maxsize reached)
- **Thread-safe** - internally uses lock

```python
q = Queue(maxsize=3)

q.put(1)  # Adds 1, queue: [1]
q.put(2)  # Adds 2, queue: [1, 2]
q.put(3)  # Adds 3, queue: [1, 2, 3]
q.put(4)  # BLOCKS! Queue is full
          # Waits until something is removed
```

### `get()`
- **Removes and returns item from the front**
- **Blocks if queue is empty**
- **Thread-safe** - internally uses lock

```python
q = Queue()
q.put(1)
q.put(2)
q.put(3)

item = q.get()  # Returns 1, queue: [2, 3]
item = q.get()  # Returns 2, queue: [3]
item = q.get()  # Returns 3, queue: []
item = q.get()  # BLOCKS! Queue is empty
                # Waits until something is added
```

### `task_done()`
- **Signals that you've finished processing an item**
- **Used with `join()`** to wait for all items to be processed
- **Decrements internal counter**

```python
q.put(1)
item = q.get()      # Remove item from queue
# Process item...
q.task_done()       # Tell queue you're done with this item
```

### `join()`
- **Waits until all items have been processed**
- **Blocks until every `get()` has a matching `task_done()`**

```python
q.put(1)
q.put(2)
q.put(3)

thread.get()  # Got item
thread.get()  # Got item
thread.get()  # Got item
thread.task_done()  # Done with first
thread.task_done()  # Done with second
thread.task_done()  # Done with third

q.join()  # Now returns (all items processed)
```

---

## Why Queue is Best Practice

### Problem with Manual Locks

**Without Queue - Manual Locking (Error-prone):**
```python
import threading

data = []
lock = threading.Lock()

def producer():
    item = produce()
    with lock:
        data.append(item)  # Must remember to use lock!

def consumer():
    with lock:
        if not data:
            # What do we do here? Busy wait? Return None?
            return None
        item = data.pop(0)
    process(item)

# Problems:
# 1. Easy to forget lock somewhere
# 2. Busy-waiting wastes CPU
# 3. No synchronization between threads
# 4. Race conditions possible
```

**With Queue - Automatic Locking (Correct):**
```python
from queue import Queue

q = Queue()

def producer():
    item = produce()
    q.put(item)  # Automatically thread-safe!

def consumer():
    item = q.get()  # Blocks if empty, no busy-wait
    process(item)   # No lock management needed!

# Benefits:
# 1. Synchronization automatic
# 2. No busy-waiting
# 3. Blocks correctly when empty/full
# 4. Task tracking with task_done()
```

---

## Queue as Locking Mechanism

Queue provides **implicit locking** without you having to manage locks directly.

### How Queue Does Locking

Inside Queue:

```python
class Queue:
    def __init__(self):
        self.mutex = Lock()           # Internal lock
        self.not_empty = Condition()  # Wait for items
        self.not_full = Condition()   # Wait for space
        self.items = []

    def put(self, item):
        with self.mutex:              # AUTO-LOCK HERE
            if self.maxsize > 0 and len(self.items) >= self.maxsize:
                self.not_full.wait()  # Wait for space

            self.items.append(item)
            self.not_empty.notify()   # Wake consumer waiting for items

    def get(self):
        with self.mutex:              # AUTO-LOCK HERE
            while not self.items:
                self.not_empty.wait() # Wait for items

            item = self.items.pop(0)
            self.not_full.notify()    # Wake producer if waiting
            return item
```

### Locking Benefits

1. **Automatic** - No manual lock management
2. **Condition variables** - Threads wait efficiently (no busy-wait)
3. **Atomic operations** - Add/remove is indivisible
4. **Fair** - FIFO order for waiting threads

---

## The Code Explained: Producer-Consumer

```python
from queue import Queue
from threading import Thread

class Producer(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        for i in range(5):
            item = random.randint(0, 256)
            self.queue.put(item)        # Safely add to queue
            print(f"producer: added {item}")
            time.sleep(1)

class Consumer(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            item = self.queue.get()     # Safely remove from queue
            print(f"consumer: got {item}")
            self.queue.task_done()      # Mark as processed

# Create queue
queue = Queue()

# Create threads
producer = Producer(queue)   # 1 producer
consumer1 = Consumer(queue)  # 3 consumers
consumer2 = Consumer(queue)
consumer3 = Consumer(queue)

# Start all
producer.start()
consumer1.start()
consumer2.start()
consumer3.start()

producer.join()   # Wait for producer to finish
consumer1.join()  # Wait for consumers (infinite loop!)
consumer2.join()
consumer3.join()
```

### Execution Flow

```
Time  Producer               Queue              Consumer1/2/3
────────────────────────────────────────────────────────────
0     produces item 42      []                 waiting...
      put(42)               [42]               get() wakes up!

0     put(42)               [42]               consumer1: got 42
                                               task_done()

1     produces item 7       []                 get() blocks (empty)
      put(7)                [7]                consumer2 wakes up

1     put(7)                [7]                consumer2: got 7
                                               task_done()

2     produces item 199     []                 get() blocks (empty)
      put(199)              [199]              consumer3 wakes up

2     put(199)              [199]              consumer3: got 199
                                               task_done()
...
5     Finishes (join done)   [waiting...]      Still running (while True)
```

---

## Queue vs Lock vs Semaphore vs RLock

| Feature | Lock | RLock | Semaphore | Queue |
|---------|------|-------|-----------|-------|
| **Purpose** | Mutual exclusion | Reentrant locking | Signaling/Resource pool | Data passing |
| **Thread-safe data** | ✗ | ✗ | ✗ | ✓ (built-in) |
| **Built-in blocking** | ✗ | ✗ | ✓ | ✓ |
| **FIFO ordering** | ✗ | ✗ | ✗ | ✓ |
| **Task tracking** | ✗ | ✗ | ✗ | ✓ (task_done) |
| **Condition vars** | ✗ | ✗ | ✓ | ✓ (internal) |
| **Best for** | Critical sections | Nested locks | Signaling | Producer-consumer |

---

## Queue Benefits Summary

### 1. **Thread-Safe Data Structure**
```python
# Without Queue - Need to manage locking
data = []
lock = Lock()
with lock:
    data.append(item)  # Manual lock

# With Queue - Automatic
q = Queue()
q.put(item)  # No lock needed
```

### 2. **Blocks Correctly**
```python
# Without Queue - Need condition variables
if not data:
    # How do we block here? Spin? Sleep?
    while not data:
        time.sleep(0.01)  # Busy-wait, wastes CPU

# With Queue - Blocks automatically
item = q.get()  # Blocks if empty, efficient
```

### 3. **No Busy-Waiting**
```python
# Bad - Busy-wait (wastes CPU)
while queue_is_empty:
    time.sleep(0.01)
item = get_item()

# Good - Queue blocks efficiently
item = q.get()  # No CPU waste, thread sleeps
```

### 4. **Task Tracking**
```python
q.put(item1)
q.put(item2)
q.put(item3)

# Process items
for _ in range(3):
    item = q.get()
    process(item)
    q.task_done()

q.join()  # Wait until all items processed
```

### 5. **Safe for Multiple Producers/Consumers**
```python
q = Queue()

# Multiple producers
producer1 = Producer(q)
producer2 = Producer(q)
producer3 = Producer(q)

# Multiple consumers
consumer1 = Consumer(q)
consumer2 = Consumer(q)
consumer3 = Consumer(q)

# All safe! Queue handles synchronization
```

---

## Internal Queue Locking Mechanism

### Step-by-Step: What Happens in `put()`

```python
q.put(42)

# Internally in Queue:
# 1. Acquire mutex (lock)
#    with self.mutex:
#        ↓
# 2. Check if full
#    if queue is full:
#        wait(not_full)  # Sleep here until there's space
#        ↓
# 3. Add item
#    self.items.append(42)
#        ↓
# 4. Notify consumers
#    not_empty.notify()  # Wake one sleeping consumer
#        ↓
# 5. Release mutex
#    (with block ends)
```

### Step-by-Step: What Happens in `get()`

```python
item = q.get()

# Internally in Queue:
# 1. Acquire mutex (lock)
#    with self.mutex:
#        ↓
# 2. Check if empty
#    while not self.items:
#        wait(not_empty)  # Sleep here until items available
#        ↓
# 3. Remove item
#    item = self.items.pop(0)
#        ↓
# 4. Notify producers
#    not_full.notify()  # Wake one sleeping producer
#        ↓
# 5. Release mutex
#    return item
```

---

## Why Queue is Safer Than Manual Locking

### Mistake 1: Forgetting Lock

```python
# ❌ WRONG - No lock on append
data = []

def producer():
    data.append(item)  # No lock! Race condition!

def consumer():
    item = data.pop(0)  # Not synchronized!

# ✅ CORRECT - Queue is always locked
q = Queue()

def producer():
    q.put(item)  # Always thread-safe

def consumer():
    item = q.get()  # Always thread-safe
```

### Mistake 2: Busy-Waiting

```python
# ❌ WRONG - Wastes CPU
while not data:
    time.sleep(0.01)  # Spin loop, bad!
item = data.pop(0)

# ✅ CORRECT - Queue blocks efficiently
item = q.get()  # Sleeps without spinning
```

### Mistake 3: Race Condition

```python
# ❌ WRONG - Check and pop not atomic
with lock:
    if len(data) > 0:  # Check
        # Lock released here!
        item = data.pop(0)  # Another thread might have removed it!

# ✅ CORRECT - Queue makes check and pop atomic
item = q.get()  # Entire operation is atomic
```

---

## Practical Example: Work Queue

```python
from queue import Queue
from threading import Thread
import time

task_queue = Queue()
results = []

def worker(worker_id):
    while True:
        task = task_queue.get()  # Blocks if no tasks

        if task is None:  # Poison pill (signal to exit)
            break

        # Do work
        print(f"Worker {worker_id} processing {task}")
        time.sleep(1)
        results.append(f"Completed {task}")

        task_queue.task_done()

# Create workers
workers = [Thread(target=worker, args=(i,)) for i in range(3)]
for w in workers:
    w.start()

# Add tasks
for i in range(10):
    task_queue.put(f"Task {i}")

# Wait for all tasks to be processed
task_queue.join()

# Stop workers
for _ in range(3):
    task_queue.put(None)  # Poison pill

for w in workers:
    w.join()

print(f"All done! Results: {results}")
```

---

## Key Takeaways

1. **Queue = Thread-Safe Data Structure**
   - Built-in locks and condition variables
   - No manual synchronization needed

2. **Automatic Locking**
   - `put()` and `get()` handle all locking
   - Prevents race conditions automatically

3. **Efficient Blocking**
   - No busy-waiting
   - Threads sleep efficiently until data available

4. **Task Tracking**
   - `task_done()` marks completion
   - `join()` waits for all tasks

5. **Best Practice for Producer-Consumer**
   - Cleaner than manual locks
   - Less error-prone
   - Better performance

6. **Multiple Threads**
   - Works safely with many producers/consumers
   - FIFO ordering guaranteed
   - All synchronization automatic

7. **In the Code**
   - Producer calls `put()` to add items
   - Consumers call `get()` to remove items
   - Queue handles all locking internally
   - `task_done()` signals completion
   - All thread-safe with zero manual locking!
