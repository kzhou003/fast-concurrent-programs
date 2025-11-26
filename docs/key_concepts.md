# Key Concepts in Concurrent Programming

This document explains the fundamental concepts used across the concurrent programming examples (scripts 06-10).

## Table of Contents
1. [Concurrency vs Parallelism](#concurrency-vs-parallelism)
2. [Threading vs Multiprocessing](#threading-vs-multiprocessing)
3. [The Global Interpreter Lock (GIL)](#the-global-interpreter-lock-gil)
4. [Asyncio and Event Loops](#asyncio-and-event-loops)
5. [Coroutines](#coroutines)
6. [Tasks and Futures](#tasks-and-futures)
7. [CPU-bound vs I/O-bound Operations](#cpu-bound-vs-io-bound-operations)
8. [Python 3.12 Migration Changes](#python-312-migration-changes)

---

## Concurrency vs Parallelism

### Concurrency
**Definition**: Multiple tasks making progress by sharing time on the same resource.

**Characteristics**:
- Tasks interleave execution
- Single CPU core can run concurrent tasks
- One task pauses, another runs
- Like juggling - one ball in hand at a time, but all balls are in play

**Python Examples**:
- Threading (limited by GIL)
- Asyncio (scripts 07-10)

**Use Cases**:
- I/O-bound operations
- Network requests
- File operations
- User interfaces

### Parallelism
**Definition**: Multiple tasks executing simultaneously on different resources.

**Characteristics**:
- True simultaneous execution
- Requires multiple CPU cores
- Each task runs on its own core
- Like having multiple jugglers

**Python Examples**:
- Multiprocessing (script 06)
- ProcessPoolExecutor

**Use Cases**:
- CPU-intensive computations
- Data processing
- Scientific computing
- Video encoding

### Visual Comparison
```
Concurrency (Threading/Asyncio):
Core 1: [Task A][Task B][Task A][Task C][Task B][Task A]
Time:   ────────────────────────────────────────────────→

Parallelism (Multiprocessing):
Core 1: [Task A──────────────────────────────]
Core 2: [Task B──────────────────────────────]
Core 3: [Task C──────────────────────────────]
Time:   ────────────────────────────────────────────────→
```

---

## Threading vs Multiprocessing

### Threading (concurrent.futures.ThreadPoolExecutor)

**How it Works**:
- Multiple threads in a single process
- Share the same memory space
- Limited by GIL in CPython

**Advantages**:
- Low memory overhead
- Fast context switching
- Easy data sharing between threads
- Good for I/O-bound tasks

**Disadvantages**:
- Cannot achieve true parallelism for CPU-bound tasks (GIL limitation)
- Race conditions possible with shared memory
- Limited by single CPU core for CPU-intensive work

**Example** (from script 06):
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    for item in number_list:
        executor.submit(evaluate, item)
```

### Multiprocessing (concurrent.futures.ProcessPoolExecutor)

**How it Works**:
- Separate Python processes
- Each has its own memory space
- Each has its own Python interpreter
- Bypasses the GIL

**Advantages**:
- True parallelism for CPU-bound tasks
- No GIL limitations
- Can fully utilize multiple CPU cores
- Process isolation (crash in one doesn't affect others)

**Disadvantages**:
- Higher memory overhead
- Slower process creation
- Inter-process communication overhead
- More complex data sharing

**Example** (from script 06):
```python
with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    for item in number_list:
        executor.submit(evaluate, item)
```

### When to Use What

| Scenario | Use Threading | Use Multiprocessing |
|----------|--------------|---------------------|
| I/O-bound tasks (network, files) | ✅ Yes | ❌ Overkill |
| CPU-bound tasks | ❌ No (GIL) | ✅ Yes |
| Need shared memory | ✅ Yes | ❌ Complex |
| Need true parallelism | ❌ No | ✅ Yes |
| Low memory available | ✅ Yes | ❌ No |

---

## The Global Interpreter Lock (GIL)

### What is the GIL?

The GIL is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecode simultaneously.

### Key Points:
1. Only one thread can execute Python bytecode at a time
2. Protects Python's memory management (reference counting)
3. Makes CPython memory-safe but limits parallelism
4. Released during I/O operations
5. Not present in Jython, IronPython, or PyPy (with STM)

### Impact on Performance:

**CPU-bound tasks with threading**:
```python
# With GIL, threads don't help CPU-bound tasks:
Sequential:        ████████████ (10 seconds)
Threading (4):     ████████████ (10 seconds) ← No improvement!
Multiprocessing:   ███ (2.5 seconds) ← True speedup!
```

**I/O-bound tasks with threading**:
```python
# GIL is released during I/O, so threading helps:
Sequential:        ████████████ (10 seconds)
Threading (4):     ███ (2.5 seconds) ← Good improvement!
Asyncio:           ██ (2 seconds) ← Even better!
```

### Observing the GIL (from script 06):
```
Sequential Execution:     3.47 seconds
Thread Pool Execution:    3.31 seconds  (minimal improvement)
Process Pool Execution:   1.23 seconds  (3x speedup!)
```

The thread pool shows minimal improvement because the CPU-bound task is limited by the GIL.

---

## Asyncio and Event Loops

### Event Loop

The event loop is the core of asyncio. It:
- Manages and schedules coroutines
- Handles I/O operations
- Coordinates concurrent tasks
- Runs in a single thread

**Conceptual Model**:
```
Event Loop:
  ┌────────────────────────────────┐
  │  1. Check for ready tasks      │
  │  2. Execute task until await   │
  │  3. Switch to next ready task  │
  │  4. Handle I/O operations      │
  │  5. Repeat                     │
  └────────────────────────────────┘
```

### How It Works (from script 07):
```python
async def main():
    loop = asyncio.get_event_loop()
    end_loop = loop.time() + 60
    await task_A(end_loop)

if __name__ == '__main__':
    asyncio.run(main())  # Creates and runs event loop
```

### Event Loop Lifecycle:
1. `asyncio.run()` creates a new event loop
2. Schedules the main coroutine
3. Runs the loop until main completes
4. Closes and cleans up the loop

### Modern vs Old Patterns:

**Old (Deprecated)**:
```python
loop = asyncio.get_event_loop()
loop.run_until_complete(coro())
loop.close()
```

**Modern (Python 3.7+)**:
```python
asyncio.run(coro())  # Handles everything automatically
```

---

## Coroutines

### What are Coroutines?

Coroutines are functions that can pause and resume execution, yielding control back to the event loop.

### Defining Coroutines:

**Old Syntax (Deprecated)**:
```python
@asyncio.coroutine
def my_coroutine():
    result = yield from other_coroutine()
    return result
```

**Modern Syntax (Python 3.5+)**:
```python
async def my_coroutine():
    result = await other_coroutine()
    return result
```

### Key Features:
1. **Non-blocking**: Can pause execution without blocking the thread
2. **Cooperative**: Explicitly yield control with `await`
3. **Chainable**: Can call other coroutines
4. **Return Values**: Can return results like regular functions

### Example from Script 08:
```python
async def state1(transition_value):
    output_value = f'State 1 with transition value = {transition_value}\n'
    await asyncio.sleep(1)  # Yields control

    if input_value == 0:
        result = await state3(input_value)  # Calls another coroutine

    return output_value + f'State 1 calling {result}'  # Returns value
```

### Execution Flow:
```
state1() starts
  ↓
Executes synchronous code
  ↓
await asyncio.sleep(1)  ← Yields control to event loop
  ↓                       ← Other tasks can run here
Resumes after sleep
  ↓
await state3()          ← Calls another coroutine
  ↓
Returns result
```

### Important Rules:
1. **Must use `await`** inside async functions (can't use `yield from`)
2. **Can only `await`** awaitable objects (coroutines, tasks, futures)
3. **Must be called** with `await` or scheduled as a task
4. **Use `asyncio.sleep()`** never `time.sleep()` in async code

---

## Tasks and Futures

### Tasks

Tasks are wrappers around coroutines that schedule them for execution.

**Creating Tasks**:

**Old (Deprecated)**:
```python
task = asyncio.Task(my_coroutine())
```

**Modern**:
```python
task = asyncio.create_task(my_coroutine())
```

### Task Characteristics:
- Automatically scheduled when created
- Run concurrently with other tasks
- Can be cancelled
- Can be awaited for results

### Example from Script 09:
```python
async def main():
    tasks = [
        asyncio.create_task(factorial(10)),
        asyncio.create_task(fibonacci(10)),
        asyncio.create_task(binomial_coefficient(20, 10))
    ]
    await asyncio.gather(*tasks)  # Wait for all tasks
```

### Futures

Futures represent the eventual result of an asynchronous operation.

**Modern Approach** (from script 10):
- Rarely needed in application code
- Tasks return values directly
- Use `await` instead of callbacks

**Old Pattern (Deprecated)**:
```python
future = asyncio.Future()
future.add_done_callback(callback_function)
future.set_result(value)
result = future.result()
```

**Modern Pattern**:
```python
result = await my_coroutine()  # Direct return value
```

### Waiting for Multiple Tasks:

**asyncio.gather()** (Recommended):
```python
results = await asyncio.gather(task1, task2, task3)
# Returns results in order
# Raises exception if any task fails
```

**asyncio.wait()** (Old pattern):
```python
done, pending = await asyncio.wait([task1, task2, task3])
# Returns sets of done and pending tasks
# More complex to use
```

---

## CPU-bound vs I/O-bound Operations

### CPU-bound Operations

Operations limited by CPU processing speed.

**Characteristics**:
- Spend most time computing
- Use CPU intensively
- Little to no waiting for external resources
- Performance scales with CPU power

**Examples**:
- Mathematical calculations (factorial, fibonacci)
- Data processing and transformations
- Compression/decompression
- Cryptography
- Image/video processing

**Best Solution**:
- **Multiprocessing** (ProcessPoolExecutor)
- Utilizes multiple CPU cores
- Bypasses GIL limitations

**Example** (from script 06):
```python
def count(number):
    for i in range(0, 10000000):  # CPU-intensive loop
        i += 1
    return i * number

# Use ProcessPoolExecutor for CPU-bound tasks
with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    for item in number_list:
        executor.submit(evaluate, item)
```

### I/O-bound Operations

Operations limited by input/output speed (waiting for data).

**Characteristics**:
- Spend most time waiting
- CPU is mostly idle
- Depend on external resources
- Performance limited by I/O speed

**Examples**:
- Network requests (HTTP, API calls)
- Database queries
- File read/write operations
- User input
- Sleep/delays

**Best Solutions**:
1. **Asyncio** (best for many I/O operations)
2. **Threading** (good for blocking I/O)

**Example** (from scripts 07-10):
```python
async def fetch_data():
    await asyncio.sleep(2)  # Simulates I/O wait
    # During this wait, other tasks can run
    return data
```

### Comparison Table:

| Aspect | CPU-bound | I/O-bound |
|--------|-----------|-----------|
| **Bottleneck** | CPU processing | Waiting for I/O |
| **CPU Usage** | High (100%) | Low (often idle) |
| **Best Solution** | Multiprocessing | Asyncio / Threading |
| **Scales With** | More CPU cores | Concurrent operations |
| **GIL Impact** | High (blocks parallelism) | Low (released during I/O) |
| **Example** | Video encoding | API requests |

### Hybrid Workloads:

Some applications have both:
```python
# Combine approaches:
async def process_data():
    # I/O: Fetch data asynchronously
    data = await fetch_from_api()

    # CPU: Process in separate process
    with ProcessPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, cpu_intensive_func, data)

    return result
```

---

## Python 3.12 Migration Changes

This section summarizes all deprecated features replaced during migration.

### 1. Time Measurement (`time.clock()` → `time.perf_counter()`)

**Issue**: `time.clock()` was deprecated in Python 3.3 and removed in Python 3.8.

**Before**:
```python
start = time.clock()
# ... work ...
elapsed = time.clock() - start
```

**After**:
```python
start = time.perf_counter()
# ... work ...
elapsed = time.perf_counter() - start
```

**Why**: `perf_counter()` provides:
- Higher resolution timing
- System-wide consistency
- Better precision for benchmarking

### 2. Coroutine Syntax (`@asyncio.coroutine` → `async def`)

**Issue**: `@asyncio.coroutine` decorator deprecated in Python 3.8.

**Before**:
```python
@asyncio.coroutine
def my_coroutine():
    result = yield from other_coroutine()
    return result
```

**After**:
```python
async def my_coroutine():
    result = await other_coroutine()
    return result
```

**Why**: `async/await` is:
- More explicit and readable
- Native language syntax
- Better error messages
- Type checker friendly

### 3. Task Creation (`asyncio.Task()` → `asyncio.create_task()`)

**Issue**: Direct `asyncio.Task()` constructor deprecated.

**Before**:
```python
task = asyncio.Task(my_coroutine())
```

**After**:
```python
task = asyncio.create_task(my_coroutine())
```

**Why**: `create_task()` is:
- The official API
- Handles edge cases better
- Works with custom event loops
- More future-proof

### 4. Event Loop Management

**Issue**: Manual loop management is verbose and error-prone.

**Before**:
```python
loop = asyncio.get_event_loop()
loop.run_until_complete(my_coroutine())
loop.close()
```

**After**:
```python
asyncio.run(my_coroutine())
```

**Why**: `asyncio.run()`:
- Handles loop creation and cleanup
- Properly closes the loop
- Cleaner and safer
- Recommended since Python 3.7

### 5. Future Callbacks (Callbacks → `await`)

**Issue**: Callback-based code is harder to read and maintain.

**Before**:
```python
future = asyncio.Future()

def callback(future):
    print(future.result())

future.add_done_callback(callback)
await some_coroutine(future)
future.set_result(value)
```

**After**:
```python
result = await some_coroutine()
print(result)
```

**Why**: Direct await:
- More readable (linear flow)
- Better error handling
- No callback hell
- Exception propagation works naturally

### 6. Blocking Calls in Async Code

**Issue**: Blocking calls freeze the entire event loop.

**Before**:
```python
async def bad_function():
    time.sleep(5)  # ❌ Blocks entire event loop!
```

**After**:
```python
async def good_function():
    await asyncio.sleep(5)  # ✅ Yields control
```

**Why**: `asyncio.sleep()`:
- Doesn't block the event loop
- Allows other tasks to run
- Proper async behavior

### 7. String Formatting (`%` → f-strings)

**Issue**: Old-style formatting is less readable.

**Before**:
```python
print('Value: %s' % value)
print('X: %s, Y: %s' % (x, y))
```

**After**:
```python
print(f'Value: {value}')
print(f'X: {x}, Y: {y}')
```

**Why**: f-strings are:
- More readable
- Faster (evaluated at runtime)
- Support expressions: `f'{x + y}'`
- Standard since Python 3.6

### Migration Checklist

- [x] Replace `time.clock()` with `time.perf_counter()`
- [x] Replace `@asyncio.coroutine` with `async def`
- [x] Replace `yield from` with `await`
- [x] Replace `asyncio.Task()` with `asyncio.create_task()`
- [x] Replace manual loop management with `asyncio.run()`
- [x] Replace Future callbacks with direct `await`
- [x] Replace `time.sleep()` with `asyncio.sleep()` in async code
- [x] Replace `%` formatting with f-strings
- [x] Test all scripts for compatibility

### Compatibility

All migrated scripts are compatible with:
- ✅ Python 3.12 (tested)
- ✅ Python 3.11
- ✅ Python 3.10
- ✅ Python 3.9
- ✅ Python 3.8
- ✅ Python 3.7 (minimum for `asyncio.run()`)

---

## Summary

### Quick Reference Guide

| Task Type | Best Solution | Example Script |
|-----------|--------------|----------------|
| CPU-intensive | ProcessPoolExecutor | Script 06 |
| I/O-intensive | Asyncio | Scripts 07-10 |
| Mixed workload | Asyncio + ProcessPool | - |
| Simple concurrency | ThreadPoolExecutor | - |
| Event scheduling | Asyncio event loop | Script 07 |
| State machines | Asyncio coroutines | Script 08 |
| Parallel tasks | asyncio.gather() | Script 09 |
| With CLI args | asyncio.run() | Script 10 |

### Key Takeaways

1. **Understand your workload**: CPU-bound vs I/O-bound determines the solution
2. **Respect the GIL**: Use multiprocessing for CPU-bound tasks
3. **Use modern syntax**: async/await is clearer than callbacks
4. **Never block the loop**: Use asyncio.sleep(), not time.sleep()
5. **Leverage asyncio.run()**: Simplest way to run async code
6. **Create tasks properly**: Use asyncio.create_task()
7. **Gather results**: Use asyncio.gather() for multiple tasks
8. **Profile first**: Measure before optimizing

### Further Reading

- [Python asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [concurrent.futures documentation](https://docs.python.org/3/library/concurrent.futures.html)
- [Understanding the GIL](https://docs.python.org/3/glossary.html#term-global-interpreter-lock)
- [PEP 492 - Coroutines with async/await](https://www.python.org/dev/peps/pep-0492/)
