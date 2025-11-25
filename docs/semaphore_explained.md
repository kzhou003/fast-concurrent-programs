# Semaphore Explained

## What is a Semaphore?

A **semaphore** is a synchronization primitive that uses an **internal counter** to control access to a shared resource. It's like having a certain number of "permits" or "tokens" that threads need to acquire.

### Key Concept
- Semaphore has a **counter** (starts at some number)
- `acquire()` - decrements the counter, if counter = 0, thread waits
- `release()` - increments the counter, wakes up waiting threads

---

## Two Types of Semaphores

### 1. Counting Semaphore (Counter > 1)
Controls access to a **pool of resources**. Example: 3 parking spots.

```python
semaphore = threading.Semaphore(3)  # 3 "permits"

semaphore.acquire()  # Counter: 3 → 2 (one car parks)
semaphore.acquire()  # Counter: 2 → 1 (another parks)
semaphore.acquire()  # Counter: 1 → 0 (another parks)
semaphore.acquire()  # Counter: 0 - WAIT! (parking full, blocked)

semaphore.release()  # Counter: 0 → 1 (one car leaves, waiting car parks)
```

### 2. Binary Semaphore (Counter = 0 or 1)
Acts like a **signal/notification**. Example: "data is ready".

```python
semaphore = threading.Semaphore(0)  # Counter starts at 0

semaphore.acquire()  # Counter: 0 - WAIT! (blocked, waiting for signal)
# ... thread blocks until...
semaphore.release()  # Counter: 0 → 1 (signal sent! waiting thread wakes up)
```

---

## The Code Explained: Producer-Consumer Pattern

Looking at `/basics/04_semaphore.py`:

```python
semaphore = threading.Semaphore(0)  # Binary semaphore, counter starts at 0
item = 0

def consumer():
    logging.info("Consumer waiting")
    semaphore.acquire()              # BLOCKS HERE! Waiting for producer
    logging.info(f"Consumer notify: item number {item}")

def producer():
    global item
    time.sleep(3)                    # Simulate work
    item = random.randint(0, 1000)   # Produce data
    logging.info(f"Producer notify: item number {item}")
    semaphore.release()              # SIGNAL! Unblock the consumer
```

### Execution Timeline

```
Time  Consumer                 Semaphore Counter    Producer
──────────────────────────────────────────────────────────────
0     waiting...              0

0     acquire()               BLOCKED!             (sleeping 3s)
      (blocked here)          0

3                             0                    wakes up
                              0                    item = 123
                              0                    release()

      UNBLOCKED!             1 → 0                acquired from 0
      prints item 123        0
```

---

## Semaphore Operations

### `acquire()`
- **Decrements counter by 1**
- **If counter becomes 0 or was 0**, thread **blocks/waits**
- Thread resumes when another thread calls `release()`

```python
semaphore = threading.Semaphore(2)

semaphore.acquire()  # Counter: 2 → 1 (continues)
semaphore.acquire()  # Counter: 1 → 0 (continues)
semaphore.acquire()  # Counter: 0 - BLOCKS HERE (waits for release)
```

### `release()`
- **Increments counter by 1**
- **Wakes up one waiting thread** (if any)
- The woken thread can now `acquire()`

```python
semaphore.acquire()  # Counter: 0 - BLOCKED
# ... from another thread ...
semaphore.release()  # Counter: 0 → 1 (wakes up blocked thread)
                     # Blocked thread can now continue
```

---

## Visual Representation

### Counting Semaphore (3 spots available)

```
Initial: Semaphore(3)
┌─────────────────────────────────┐
│ Available Spots: 3              │
└─────────────────────────────────┘

Thread A calls acquire():
┌─────────────────────────────────┐
│ Available Spots: 2              │
│ Thread A: ACQUIRED ✓            │
└─────────────────────────────────┘

Thread B calls acquire():
┌─────────────────────────────────┐
│ Available Spots: 1              │
│ Thread A: ACQUIRED              │
│ Thread B: ACQUIRED ✓            │
└─────────────────────────────────┘

Thread C calls acquire():
┌─────────────────────────────────┐
│ Available Spots: 0              │
│ Thread A: ACQUIRED              │
│ Thread B: ACQUIRED              │
│ Thread C: ACQUIRED ✓            │
└─────────────────────────────────┘

Thread D calls acquire():
┌─────────────────────────────────┐
│ Available Spots: 0              │
│ Thread A: ACQUIRED              │
│ Thread B: ACQUIRED              │
│ Thread C: ACQUIRED              │
│ Thread D: WAITING (blocked)     │ ← Can't proceed
└─────────────────────────────────┘

Thread A calls release():
┌─────────────────────────────────┐
│ Available Spots: 1              │
│ Thread A: RELEASED              │
│ Thread B: ACQUIRED              │
│ Thread C: ACQUIRED              │
│ Thread D: ACQUIRED (now!) ✓     │ ← Unblocked!
└─────────────────────────────────┘
```

### Binary Semaphore (Producer-Consumer)

```
Initial: Semaphore(0)
┌──────────────────────┐
│ Counter: 0           │
│ Signal: NOT SET      │
└──────────────────────┘

Consumer calls acquire():
┌──────────────────────┐
│ Counter: 0           │
│ Consumer: WAITING    │ ← BLOCKED here
│ Signal: NOT SET      │
└──────────────────────┘

Producer (after work) calls release():
┌──────────────────────┐
│ Counter: 1           │
│ Consumer: PROCEED ✓  │ ← UNBLOCKED!
│ Signal: SET          │
└──────────────────────┘
```

---

## Semaphore vs Lock

| Aspect | Lock | Semaphore |
|--------|------|-----------|
| **Purpose** | Mutual exclusion | Signaling / Resource pooling |
| **Counter** | 0 or 1 (locked/unlocked) | Any number |
| **Multiple resources** | 1 | Many |
| **Waiting threads** | Any thread can unlock | Only thread that acquired can release |
| **Use case** | Protect critical section | Signal events or manage pool |

---

## Real-World Examples

### Example 1: Swimming Pool with Limited Capacity

```python
import threading
import time

pool_capacity = threading.Semaphore(3)  # Only 3 people can swim at once

def swimmer(name):
    print(f"{name}: trying to enter pool...")
    pool_capacity.acquire()              # Wait if full

    print(f"{name}: SWIMMING!")
    time.sleep(2)                        # Swim for 2 seconds

    pool_capacity.release()              # Leave, let someone else in
    print(f"{name}: LEFT the pool")

threads = [threading.Thread(target=swimmer, args=(f"Swimmer {i}",))
           for i in range(5)]

for t in threads:
    t.start()

for t in threads:
    t.join()
```

**Output:**
```
Swimmer 0: trying to enter pool...
Swimmer 0: SWIMMING!
Swimmer 1: trying to enter pool...
Swimmer 1: SWIMMING!
Swimmer 2: trying to enter pool...
Swimmer 2: SWIMMING!
Swimmer 3: trying to enter pool...
Swimmer 3: WAITING (blocked - pool full)
Swimmer 4: trying to enter pool...
Swimmer 4: WAITING (blocked - pool full)

[After 2 seconds]
Swimmer 0: LEFT the pool
Swimmer 3: SWIMMING! (now can enter)
Swimmer 1: LEFT the pool
Swimmer 4: SWIMMING! (now can enter)
...
```

### Example 2: Producer-Consumer (Like the Code)

```python
import threading
import time
import random

queue_semaphore = threading.Semaphore(0)  # 0 items initially
item_produced = None

def producer():
    global item_produced
    time.sleep(2)
    item_produced = random.randint(1, 100)
    print(f"Producer: created item {item_produced}")
    queue_semaphore.release()  # Signal: item ready!

def consumer():
    print("Consumer: waiting for item...")
    queue_semaphore.acquire()  # Wait for item
    print(f"Consumer: received item {item_produced}")

producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)

consumer_thread.start()    # Consumer waits first
producer_thread.start()    # Producer produces

consumer_thread.join()
producer_thread.join()
```

**Output:**
```
Consumer: waiting for item...
[2 seconds pass]
Producer: created item 42
Consumer: received item 42
```

---

## How the Code Works Step by Step

```python
# Initial state
semaphore = threading.Semaphore(0)  # Counter = 0
item = 0

# Consumer thread starts
def consumer():
    logging.info("Consumer waiting")
    semaphore.acquire()              # Counter is 0, BLOCKS HERE
                                     # Consumer thread is waiting
    logging.info(f"Consumer notify: item number {item}")  # Not reached yet

# Producer thread starts (10 times in the loop)
def producer():
    global item
    time.sleep(3)                    # Simulate work for 3 seconds
    item = random.randint(0, 1000)   # Generate item
    logging.info(f"Producer notify: item number {item}")
    semaphore.release()              # Counter: 0 → 1
                                     # This WAKES UP the consumer

# Execution timeline:
# Time 0:   consumer() starts, calls acquire(), blocks (counter=0)
# Time 0:   producer() starts, sleeps for 3 seconds
# Time 3:   producer() wakes up, generates item
# Time 3:   producer() calls release() - counter becomes 1
# Time 3:   consumer() wakes up from acquire(), can continue
# Time 3:   consumer() prints the item
# Time 3:   both threads finish (join completes)
#
# Loop repeats 10 times
```

---

## Semaphore States and Transitions

```
Binary Semaphore (0 or 1) - Producer-Consumer

INITIAL STATE:
┌─────────────┐
│ Counter: 0  │
│ No signal   │
└──────┬──────┘
       │
       │ Consumer acquire()
       │ (counter = 0, so block)
       ▼
┌──────────────────┐
│ Counter: 0       │
│ Consumer WAITING │
└──────┬───────────┘
       │
       │ Producer release()
       │ (counter 0 → 1, wake consumer)
       ▼
┌──────────────────────┐
│ Counter: 1           │
│ Consumer UNBLOCKED   │
│ Can now acquire() → 0│
└──────────────────────┘
```

---

## Counting Semaphore States

```
Counting Semaphore(3) - Resource Pool

INITIAL:
┌─────────────┐
│ Counter: 3  │
│ 3 available │
└──────┬──────┘
       │
       │ Thread A acquire()
       ▼
┌──────────────────┐
│ Counter: 2       │
│ A acquired       │
└──────┬───────────┘
       │
       │ Thread B acquire()
       ▼
┌──────────────────┐
│ Counter: 1       │
│ B acquired       │
└──────┬───────────┘
       │
       │ Thread C acquire()
       ▼
┌──────────────────┐
│ Counter: 0       │
│ C acquired       │
└──────┬───────────┘
       │
       │ Thread D acquire() - BLOCKS!
       ▼
┌──────────────────┐
│ Counter: 0       │
│ D WAITING        │
└──────┬───────────┘
       │
       │ Thread A release()
       │ (counter 0 → 1, wake D)
       ▼
┌──────────────────┐
│ Counter: 0       │
│ D acquired!      │
└──────────────────┘
```

---

## Common Use Cases

### 1. **Limiting Concurrent Access**
```python
# Only 5 threads can access database at once
db_access = threading.Semaphore(5)

def query_database(query):
    with db_access:  # Use like a lock
        # Access database
        pass
```

### 2. **Producer-Consumer Communication**
```python
# Sender signals receiver
data_ready = threading.Semaphore(0)

def sender():
    prepare_data()
    data_ready.release()  # Signal: data ready!

def receiver():
    data_ready.acquire()  # Wait for signal
    process_data()
```

### 3. **Synchronizing Multiple Threads**
```python
# Wait for all workers to finish
workers_done = threading.Semaphore(0)
num_workers = 5

def worker():
    do_work()
    workers_done.release()

# Main waits for all
for _ in range(num_workers):
    workers_done.acquire()
print("All workers done!")
```

---

## Key Differences from Lock

### Lock (`threading.Lock`)
```python
lock = threading.Lock()

with lock:  # Only ONE thread can be here at a time
    shared_data += 1
```

### Semaphore (Counting)
```python
semaphore = threading.Semaphore(3)

with semaphore:  # Up to 3 threads can be here at a time
    access_resource()
```

### Semaphore (Binary - Used as Signal)
```python
signal = threading.Semaphore(0)

# Thread A waits for signal
signal.acquire()
print("I've been signaled!")

# Thread B sends signal
signal.release()
```

---

## Summary Table

| Operation | Effect | When |
|-----------|--------|------|
| `acquire()` | Counter - 1, blocks if 0 | When entering critical section |
| `release()` | Counter + 1, wakes 1 thread | When leaving critical section |
| Multiple resources | ✓ Semaphore(n) | Manage pool |
| Signaling | ✓ Semaphore(0) | Event notification |

---

## Key Takeaways

1. **Semaphore = Counter + Wait Queue**
   - `acquire()` decrements, blocks if 0
   - `release()` increments, wakes waiting threads

2. **Binary Semaphore (0/1)** - Used for signaling
   - Start at 0: consumer waits, producer signals
   - Start at 1: resource available

3. **Counting Semaphore (>1)** - Used for resource pools
   - Start at N: N threads can access resource
   - Thread blocks if counter = 0

4. **Difference from Lock**
   - Lock: binary (locked/unlocked)
   - Semaphore: can have many resources

5. **In the Code**
   - `Semaphore(0)`: Consumer waits, producer signals
   - `acquire()`: Consumer blocks until data ready
   - `release()`: Producer unblocks consumer
   - This is the **producer-consumer pattern**

The code is a classic **producer-consumer synchronization** where the semaphore ensures the consumer doesn't process data before the producer has created it.
