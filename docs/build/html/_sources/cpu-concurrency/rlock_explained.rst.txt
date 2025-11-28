RLock (Reentrant Lock) Explained

What is RLock?

**RLock** stands for **Reentrant Lock**. It's a special type of lock that allows the **same thread to acquire it multiple times** without deadlocking.

Regular Lock vs RLock
~~~~~~~~~~~~~~~~~~~~~

**Regular Lock (threading.Lock):**
.. code-block:: python


with lock:           # Thread acquires lock
    with lock:       # Same thread tries to acquire again
        pass         # DEADLOCK! Thread waits for itself!
::


**RLock (threading.RLock):**
.. code-block:: python


with rlock:          # Thread acquires lock (count = 1)
    with rlock:      # Same thread acquires again (count = 2)
        pass         # No deadlock! Just increments counter
                     # Releases when count goes back to 0
.. code-block:: text





How RLock Works Internally

RLock uses a **counter system**:

1. **First acquire** - Counter = 1, lock is held
2. **Same thread acquires again** - Counter = 2, lock still held
3. **Same thread acquires again** - Counter = 3, lock still held
4. **First release** - Counter = 2, lock still held
5. **Second release** - Counter = 1, lock still held
6. **Third release** - Counter = 0, lock released

::

Thread A acquires:    Counter = 1 [OK]
Thread A acquires:    Counter = 2 [OK]
Thread A acquires:    Counter = 3 [OK]
(other threads blocked)
Thread A releases:    Counter = 2 (still locked)
Thread A releases:    Counter = 1 (still locked)
Thread A releases:    Counter = 0 (NOW unlocked)
Thread B can now acquire
.. code-block:: text





The Code Explained

Looking at ``/basics/03*rlock.py``:

.. code-block:: python

    def **init**(self):
        self.lock = threading.RLock()  # Reentrant lock
        self.total_items = 0

    def execute(self, value):
        with self.lock:                # Acquire lock
            self.total_items += value  # Modify shared data
                                       # Release lock

    def add(self):
        with self.lock:                # Acquire lock (1st time)
            self.execute(1)            # Calls execute...
                                       # execute() tries to acquire same lock (2nd time)
                                       # RLock allows this! Regular Lock would deadlock

    def remove(self):
        with self.lock:                # Acquire lock
            self.execute(-1)           # Same lock acquisition happens here
::


Why RLock is Needed Here
~~~~~~~~~~~~~~~~~~~~~~~~

Without RLock, this would happen:

.. code-block:: python

lock = threading.Lock()

def execute(self, value):
    with lock:  # Acquires lock
        self.total_items += value

def add(self):
    with lock:  # Acquires lock (1st time)
        self.execute(1)  # Tries to acquire lock (2nd time)
                         # DEADLOCK! Same thread waiting for itself!
::


With RLock, it works fine:

.. code-block:: python

rlock = threading.RLock()

def execute(self, value):
    with rlock:  # Acquires lock, counter = 1
        self.total_items += value

def add(self):
    with rlock:  # Acquires lock, counter = 1
        self.execute(1)  # Acquires same lock, counter = 2
                         # No deadlock! Just increments counter
                         # After execute() returns, counter = 1
                         # After add() ends, counter = 0
.. code-block:: text





When to Use RLock vs Regular Lock

Use Regular Lock When:
~~~~~~~~~~~~~~~~~~~~~~
- Simple cases where you don't have nested lock acquisitions
- Different methods don't call each other
- Performance is critical (RLock is slightly slower)

.. code-block:: python

    def **init**(self):
        self.lock = threading.Lock()  # Sufficient here
        self.value = 0

    def increment(self):
        with self.lock:
            self.value += 1  # Simple, no nested calls
::


Use RLock When:
~~~~~~~~~~~~~~~
- Methods call other methods that also need the lock
- You have nested function calls
- Complex class hierarchies
- You're unsure if nesting will happen

.. code-block:: python

    def **init**(self):
        self.lock = threading.RLock()  # Needed here!
        self.items = 0

    def add(self):
        with self.lock:
            self.execute(1)  # Calls another method

    def execute(self, value):
        with self.lock:      # Same lock needed here too
            self.items += value
.. code-block:: text





Execution Flow of the Code

::

Thread 1 (adder):              Thread 2 (remover):
  add()
    with lock (count=1)
      execute(1)
        with lock (count=2)    (blocked - same thread can go)
          items += 1
        (count=1)
      (count=0)                remove()
                                 with lock (count=1)
                                   execute(-1)
                                     with lock (count=2)
                                       items -= 1
                                     (count=1)
                                   (count=0)
::


**Key Point:** The same thread can acquire the same RLock multiple times without blocking. With a regular Lock, this would deadlock.

---

Reentrant Lock Counter Animation

::

Time  Thread A              RLock Counter    Thread B Status
1     acquire()            1                blocked (waiting)
2     acquire()            2                blocked
3     release()            1                blocked
4     release()            0                Can now acquire!
5     (out of lock)                        acquire() succeeds
::


With a regular Lock, Thread B would wait forever because Thread A keeps acquiring without releasing.

---

Code Flow with Comments

.. code-block:: python

    def **init**(self):
        self.lock = threading.RLock()  # Counter starts at 0
        self.total_items = 0

    def execute(self, value):
        with self.lock:                # Counter: 0->1
            self.total_items += value
        # Counter: 1->0 (lock released)

    def add(self):
        with self.lock:                # Counter: 0->1
            self.execute(1)
            # Inside execute():
            # Counter: 1->2 (REENTRANT - same thread!)
            # Modify items
            # Counter: 2->1
        # Counter: 1->0 (lock fully released)

    def remove(self):
        with self.lock:                # Counter: 0->1
            self.execute(-1)
            # Inside execute():
            # Counter: 1->2 (REENTRANT)
            # Modify items
            # Counter: 2->1
        # Counter: 1->0
.. code-block:: text





Visual Comparison

Regular Lock - Would Deadlock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text




         |   Thread A acquires     |
         |   Lock acquired [OK]       |
         +------------[v]-------------+
         | Thread A tries to        |
         | acquire again            |
         | WAITING... (deadlock!)   | <- Blocked forever
         | (waiting for itself)     |
.. code-block:: text




~~~~~~~~~~~~~~~~~~~

.. code-block:: text




    | Thread A acquires (1)    |
    | Lock count = 1 [OK]         |
    +-------------[v]-------------+
    | Thread A acquires again   |
    | Lock count = 2 [OK]          | <- Allowed!
    | (same thread, no block)   |
    +-------------[v]-------------+
    | Thread A releases (2->1)   |
    | Lock still held [OK]         |
    +-------------[v]-------------+
    | Thread A releases (1->0)   |
    | Lock released [OK]           | <- Now others can acquire
.. code-block:: text





Real-World Example

.. code-block:: python

    def **init**(self, balance=0):
        self.lock = threading.RLock()
        self.balance = balance

    def withdraw(self, amount):
        """Helper method - acquires lock"""
        with self.lock:
            if self.balance >= amount:
                self.balance -= amount
                return True
            return False

    def transfer(self, other, amount):
        """Main method - also needs lock"""
        with self.lock:
            # Need to ensure consistent state while transferring
            if self.withdraw(amount):  # withdraw() also wants lock!
                with other.lock:
                    other.balance += amount
                    return True
            return False

Without RLock:
transfer() acquires self.lock
withdraw() tries to acquire self.lock again
DEADLOCK!
=========

With RLock:
transfer() acquires self.lock (count=1)
withdraw() acquires self.lock (count=2) - allowed!
No deadlock!
.. code-block:: text





Summary Table

| Aspect | Regular Lock | RLock |
| **Same thread re-acquire** | [[FAIL]] Deadlock | [OK] Allowed |
| **Counter system** | No | Yes (0, 1, 2, ...) |
| **Nested calls** | Dangerous | Safe |
| **Performance** | Faster | Slightly slower |
| **Use case** | Simple locking | Complex methods |

---

Key Takeaways

1. **RLock = Reentrant Lock** - allows same thread to acquire multiple times
2. **Uses a counter** - increments on acquire, decrements on release
3. **Only fully released when counter = 0** - previous locks stay held
4. **Prevents deadlock** - same thread can safely call nested locked methods
5. **Slightly slower** - use regular Lock if you don't need reentrancy
6. **When in doubt, use RLock** - it's safer for complex code with nested calls

The code in ``03*rlock.py`` uses RLock because:
- ``add()`` calls ``execute()`` (both want the lock)
- ``remove()`` calls ``execute()`` (both want the lock)
- Without RLock, these would deadlock
- With RLock, the same thread can safely reacquire the lock
