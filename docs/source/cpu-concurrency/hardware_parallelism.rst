Hardware Parallelism: Cores, Hyperthreading, and GPUs

A comprehensive guide to understanding physical cores, hyperthreading, and why GPUs excel at compute-intensive parallel tasks.

Table of Contents
1. `Physical Cores vs Logical Cores <#physical-cores-vs-logical-cores>`_
2. `What is Hyperthreading? <#what-is-hyperthreading>`_
3. `How Threading is Bounded by Physical Cores <#how-threading-is-bounded-by-physical-cores>`_
4. `Why GPUs Excel at Parallel Computing <#why-gpus-excel-at-parallel-computing>`_
5. `CPU vs GPU Architecture <#cpu-vs-gpu-architecture>`_
6. `When to Use What <#when-to-use-what>`_
7. `Practical Examples <#practical-examples>`_

---

Physical Cores vs Logical Cores

Physical Cores
~~~~~~~~~~~~~~

**Definition**: An actual, independent processing unit on the CPU chip with its own:
- Arithmetic Logic Unit (ALU)
- Floating Point Unit (FPU)
- L1 and L2 cache
- Execution units

::

Physical CPU Chip:
|  |  Core 0  |  |  Core 1  |  |  Core 2  |   ...   |
|  | | ALU  | |  | | ALU  | |  | | ALU  | |         |
|  | | FPU  | |  | | FPU  | |  | | FPU  | |         |
|  | | L1/L2| |  | | L1/L2| |  | | L1/L2| |         |
|              Shared L3 Cache                        |


**Characteristics**:
- [[OK]] True parallel execution
- [[OK]] Independent computation streams
- [[OK]] Each can execute different instructions simultaneously
- [[OK]] Maximum performance for CPU-bound tasks

Logical Cores
~~~~~~~~~~~~~

**Definition**: Virtual cores created by technologies like Intel's Hyperthreading or AMD's Simultaneous Multithreading (SMT).

**Example**:
- CPU: Intel Core i7 with 4 physical cores
- With Hyperthreading: Shows as 8 logical cores
- Ratio: 2 logical cores per 1 physical core

.. code-block:: python


Check on your system
logical_cores = os.cpu_count()  # e.g., 8
Physical cores require platform-specific code:
On Linux: check /proc/cpuinfo
On macOS: sysctl hw.physicalcpu
Typically: physical_cores = logical_cores / 2 (if HT enabled)
::


---

What is Hyperthreading?

The Concept
~~~~~~~~~~~

**Hyperthreading (HT)** allows a single physical core to execute two instruction streams (threads) simultaneously by sharing the core's execution resources.

How It Works
~~~~~~~~~~~~

A CPU core has multiple execution units but they're not always all in use:

::

Without Hyperthreading (one thread per core):
Time ->
Core execution units: [ALU][FPU][Load][Store][Branch]
                       down    down     down     down      down
Thread A:             [[#]]  [ ]   [[#]]   [ ]    [[#]]   <- Only 60% utilized
                       up         up            up
                    Used units  (unused)   Used units

Wasted capacity: 40%
::


::

With Hyperthreading (two threads per core):
Time ->
Core execution units: [ALU][FPU][Load][Store][Branch]
                       down    down     down     down      down
Thread A:             [[#]]  [ ]   [[#]]   [ ]    [[#]]
Thread B:             [ ]  [[#]]   [ ]   [[#]]    [ ]   <- Fills the gaps!
                       up    up     up     up      up
Combined utilization: [[#]]  [[#]]   [[#]]   [[#]]    [[#]]   <- ~85% utilized

Better resource usage!
::


Technical Implementation
~~~~~~~~~~~~~~~~~~~~~~~~

Each physical core with HT has:

::

Physical Core with Hyperthreading:
|  Duplicated (per thread):              |
|  | Thread 1 |      | Thread 2 |        |
|  | * PC     |      | * PC     |        |  PC = Program Counter
|  | * Regs   |      | * Regs   |        |  Regs = Registers
|  | * State  |      | * State  |        |
|  Shared (between both threads):        |
|  | * ALU (Arithmetic Logic Unit)    |  |
|  | * FPU (Floating Point Unit)      |  |
|  | * L1/L2 Cache                    |  |
|  | * Execution Units                |  |
|  | * Load/Store Units               |  |


**Key Insight**: Two threads share the same execution hardware but have separate architectural state (registers, program counter).

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best Case** (threads use different execution units):
::

Thread A: Integer operations (uses ALU)
Thread B: Floating-point operations (uses FPU)
Result: ~70-80% better performance than single thread
::


**Worst Case** (threads compete for same resources):
::

Thread A: Integer operations (needs ALU)
Thread B: Integer operations (also needs ALU)
Result: ~10-20% better performance (mostly from hiding latency)
::


**Reality** (typical workloads):
::

Hyperthreading improvement: 20-40% on average
Still much less than true dual-core: 100% improvement
::


Hyperthreading Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Not True Parallelism**
   ::

   1 physical core + HT = 1.3x performance (not 2x)
   2 physical cores = 2x performance
   ::


2. **Shared Resources Create Contention**
   ::

   Both threads need cache -> cache thrashing
   Both threads need FPU -> one waits
   Both threads need memory -> bandwidth split
   ::


3. **Can Hurt Performance in Some Cases**
   .. code-block:: python

   # 4 physical cores, 8 logical cores

   # Using 4 workers (physical cores): 3.8x speedup [[OK]]
   # Using 8 workers (logical cores): 3.2x speedup [[FAIL]] (worse!)

   # Why? OS scheduling overhead + resource contention
   ::


Checking Hyperthreading Status
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linux**:
.. code-block:: bash

lscpu | grep "Thread(s) per core"
Output: Thread(s) per core: 2  <- HT enabled
Output: Thread(s) per core: 1  <- HT disabled

Or check CPU info
grep -E "siblings|cpu cores" /proc/cpuinfo | head -2
siblings = logical cores per physical CPU
cpu cores = physical cores per physical CPU
::


**macOS**:
.. code-block:: bash

sysctl hw.logicalcpu
Output: hw.logicalcpu: 8

Physical cores
sysctl hw.physicalcpu
Output: hw.physicalcpu: 4

If logical > physical, HT is enabled
::


**Python**:
.. code-block:: python

import subprocess

logical_cores = os.cpu_count()

Platform-specific physical core detection
import platform
if platform.system() == 'Darwin':  # macOS
    result = subprocess.run(['sysctl', '-n', 'hw.physicalcpu'],
                          capture_output=True, text=True)
    physical_cores = int(result.stdout.strip())
elif platform.system() == 'Linux':
    # Count unique physical IDs
    with open('/proc/cpuinfo') as f:
        physical_cores = len(set(
            line.split(':')[1].strip()
            for line in f
            if line.startswith('physical id')
        ))
else:  # Windows
    physical_cores = logical_cores // 2  # Approximation

print(f"Logical cores: {logical_cores}")
print(f"Physical cores: {physical_cores}")
print(f"Hyperthreading: {'Enabled' if logical_cores > physical_cores else 'Disabled'}")
::


---

How Threading is Bounded by Physical Cores

The Fundamental Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~

**No matter how many threads you create, true parallel execution is limited by physical cores.**

.. code-block:: text

    System: 4 physical cores (8 logical with HT)

    Scenario 1: 4 CPU-intensive threads
    |Thread 1 | |Thread 2 | |Thread 3 | |Thread 4 |
    |  100%   | |  100%   | |  100%   | |  100%   |
         down           down           down           down
    | Core 0  | | Core 1  | | Core 2  | | Core 3  |
    |  100%   | |  100%   | |  100%   | |  100%   |
    Result: Perfect utilization, 4x speedup [OK]

    Scenario 2: 8 CPU-intensive threads
    | T1 || T2 || T3 || T4 || T5 || T6 || T7 || T8 |
         down      down      down      down      down      down      down
    | Core 0  | | Core 1  | | Core 2  | | Core 3  |
    | T1 + T5 | | T2 + T6 | | T3 + T7 | | T4 + T8 |
    | compete | | compete | | compete | | compete |
    Result: ~4.5x speedup (not 8x!) [warning]

    Scenario 3: 16 CPU-intensive threads
    |T1||T2||T3||T4||T5||T6||T7||T8||T9||10||11||12||13||14||15||16|
                              down
    | Core 0  | | Core 1  | | Core 2  | | Core 3  |
    | 4 threads| | 4 threads| | 4 threads| | 4 threads|
    |time-slice| |time-slice| |time-slice| |time-slice|
    Result: ~4x speedup (same as 4 threads!) + overhead [FAIL]


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CPU-bound tasks** are limited by actual computation capacity:

.. code-block:: python


import time
from concurrent.futures import ProcessPoolExecutor

def compute(n):
    return sum(i_i for i in range(n))

def benchmark(num_workers):
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = [10*000*000] * num_workers
        list(executor.map(compute, tasks))
    return time.perf_counter() - start

Results on 4-core CPU:
1 worker:  10.0s  (baseline)
2 workers:  5.1s  (1.96x speedup) [[OK]]
4 workers:  2.6s  (3.85x speedup) [[OK]]
8 workers:  2.8s  (3.57x speedup) [warning] (worse than 4!)
16 workers: 3.2s  (3.13x speedup) [[FAIL]] (much worse!)
::


**Why performance degrades**:

1. **Context Switching Overhead**
   ::

   OS must constantly switch between threads:
   - Save thread state (registers, PC, stack pointer)
   - Load next thread state
   - Flush CPU caches
   - Update memory mappings

   Cost: ~1-10 microseconds per switch
   With many threads: Spends more time switching than computing!
   ::


2. **Cache Thrashing**
   ::

   Each thread loads its data into cache:
   Thread A: Loads data -> Evicts Thread B's cache
   Thread B: Loads data -> Evicts Thread C's cache
   Thread C: Loads data -> Evicts Thread A's cache
   Thread A: Needs data again -> Cache miss! (must reload)

   Result: More memory access, slower execution
   ::


3. **Resource Contention**
   ::

   Multiple threads compete for:
   - Memory bandwidth
   - Cache space
   - TLB entries
   - Execution units

   More threads = More contention = Slower per-thread progress
   ::


Optimal Worker Count
~~~~~~~~~~~~~~~~~~~~

**For CPU-bound tasks**:

.. code-block:: python


Best practice:
physical_cores = os.cpu_count() // 2  # Approximate physical cores
optimal_workers = physical_cores

Conservative (recommended for production):
optimal_workers = max(1, physical_cores - 1)  # Leave one core for OS

Or detect actual physical cores:
import psutil  # pip install psutil
optimal_workers = psutil.cpu_count(logical=False)
::


**Rule of thumb**:
::

CPU-bound tasks:

I/O-bound tasks:


Real-World Example
~~~~~~~~~~~~~~~~~~

.. code-block:: python


import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time

def multiply_matrices(size):
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    return np.dot(A, B)

def benchmark(num_workers, num_tasks=12):
    """
    Multiply 12 matrices of size 1000x1000
    Each multiplication takes ~1 second on one core
    """
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(multiply_matrices, 1000)
                   for * in range(num_tasks)]
        results = [f.result() for f in futures]

    elapsed = time.perf_counter() - start
    speedup = (num_tasks * 1.0) / elapsed
    efficiency = speedup / num_workers * 100

    return elapsed, speedup, efficiency

Test on 4-core CPU:
print("Workers | Time  | Speedup | Efficiency")
print("--------|-------|---------|------------")
for workers in [1, 2, 4, 6, 8]:
    time, speedup, eff = benchmark(workers)
    print(f"{workers:7} | {time:5.1f}s | {speedup:5.2f}x  | {eff:6.1f}%")

Typical output:
Workers | Time  | Speedup | Efficiency
      1 | 12.0s |  1.00x  |  100.0%
      2 |  6.1s |  1.97x  |   98.5%  <- Near perfect
      4 |  3.1s |  3.87x  |   96.8%  <- Near perfect
      6 |  2.8s |  4.29x  |   71.5%  <- Diminishing returns
      8 |  2.9s |  4.14x  |   51.8%  <- Getting worse
::


**Analysis**:
- **1-4 workers**: Nearly linear speedup (limited by 4 physical cores)
- **6 workers**: Still faster, but efficiency drops (HT helps a bit)
- **8 workers**: Performance plateau or degradation (overhead dominates)

---

Why GPUs Excel at Parallel Computing

CPU vs GPU: Different Design Philosophies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

CPU (Latency-Optimized):
Goal: Execute single thread as fast as possible


::

    |  Few cores (4-64), each very powerful  |
    |  | Core 0  |  | Core 1  |             |
    |  |  |ALU|  |  |  |ALU|  |  ...        |
    |  |  |FPU|  |  |  |FPU|  |             |
    |  |  |Big|  |  |  |Big|  |             |
    |  |  |L1 |  |  |  |L1 |  |             |
    |  |  |L2 |  |  |  |L2 |  |             |
    |  | Complex |  | Complex |             |
    |  | Control |  | Control |             |
    |  |Out-order|  |Out-order|             |
    |  | Exec    |  | Exec    |             |
    |      Huge L3 Cache (32+ MB)            |
    |      Complex Branch Prediction         |
    |      Speculative Execution             |


GPU (Throughput-Optimized):
Goal: Execute many threads simultaneously


::

    |  Thousands of tiny cores               |
    | |C||C||C||C||C||C||C||C||C||C|        |
    | |C||C||C||C||C||C||C||C||C||C|  ...   |
    | |C||C||C||C||C||C||C||C||C||C|        |
    |  ... (thousands more) ...              |
    |  Tiny L1 caches                        |
    |  Simple control logic                  |
    |  No branch prediction                  |
    |  No out-of-order execution             |
    |  SIMD: Same instruction, all cores     |

::


Key Differences
~~~~~~~~~~~~~~~

| Aspect | CPU | GPU |
| **Core Count** | 4-64 cores | 1,000s-10,000s of cores |
| **Core Speed** | 3-5 GHz | 1-2 GHz |
| **Core Complexity** | Very complex | Very simple |
| **Cache per Core** | 256KB-2MB L2 | 16-128KB L1 |
| **Control Logic** | 40% of die | 5% of die |
| **Compute Units** | 60% of die | 95% of die |
| **Best For** | Complex, branching code | Simple, repetitive operations |
| **Parallelism** | 4-64 tasks | 1,000-10,000+ tasks |

SIMD and GPU Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~

**SIMD**: Single Instruction, Multiple Data

::

CPU executing 4 additions sequentially:
Time ->
Core 1: [A+B] -> [C+D] -> [E+F] -> [G+H]  (4 time units)

GPU executing 4 additions in parallel (SIMD):
Time ->
Core 1: [A+B]
Core 2: [C+D]  } All execute simultaneously
Core 3: [E+F]    (1 time unit)
Core 4: [G+H]

Speedup: 4x for this simple case
::


**GPU Organization** (NVIDIA example):

::

GPU (e.g., NVIDIA RTX 4090):
|  Streaming Multiprocessors (SMs): 128 units         |
|  Each SM contains:                                  |
|  |- 128 CUDA cores (simple ALUs)                   |
|  |- 4 Tensor cores (matrix operations)             |
|  |- Shared memory (64-128 KB)                      |
|  +- Warp scheduler                                 |
|  Total: 128 SMs x 128 cores = 16,384 CUDA cores    |
|  All cores execute in lockstep (SIMD):             |
|  One instruction broadcast to 32 cores (1 warp)    |


Why GPUs Excel at Compute-Intensive Tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Massive Parallelism
^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Add 1 million numbers

.. code-block:: python

Split into 4 chunks of 250,000 each
Each core processes 250,000 additions sequentially

Core 1: [sum 250,000 numbers]
Core 2: [sum 250,000 numbers]  } Parallel
Core 3: [sum 250,000 numbers]
Core 4: [sum 250,000 numbers]

Time: 250,000 additions / core_speed
::


.. code-block:: python

Split into 16,384 chunks of ~61 each
Each core processes ~61 additions

Cores 1-16,384: [sum ~61 numbers each]  } All parallel

Time: 61 additions / core_speed

Speedup: 250,000 / 61 ~ 4,000x faster!
::


2. Perfect for Data-Parallel Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Data-parallel**: Same operation on different data elements

::

Example: Image processing - apply filter to each pixel

Image: 1920x1080 = 2,073,600 pixels
Operation: Apply blur filter to each pixel

CPU (4 cores):
Time: 518,400 pixels per core

GPU (8,192 cores):
Time: ~253 pixels per core

Speedup: 518,400 / 253 ~ 2,048x faster!
::


3. High Memory Bandwidth
^^^^^^^^^^^^^^^^^^^^^^^^

::

CPU Memory Bandwidth:

GPU Memory Bandwidth:

Ratio: GPU has 10-20x more memory bandwidth!
::


**Why it matters**:
::

Compute-intensive task often needs:
1. Read data from memory
2. Perform computation
3. Write result to memory

With 16,384 cores all reading/writing:


What GPUs Are Good At
~~~~~~~~~~~~~~~~~~~~~

**Perfect for GPUs** (embarrassingly parallel):

1. **Matrix Operations**
   .. code-block:: python

   # Each element of C can be computed independently

   C[i,j] = sum(A[i,k] * B[k,j] for k in range(n))

   # For 1000x1000 matrices:
   # 1,000,000 independent calculations
   # Perfect for 16,384 GPU cores!
   ::


2. **Image/Video Processing**
   .. code-block:: python

   for each pixel in image:
       output[pixel] = transform(input[pixel])

   # 4K video frame: 8,294,400 pixels
   # All can be processed in parallel on GPU
   ::


3. **Deep Learning**
   .. code-block:: python

   layer_output = activation(weights @ inputs + bias)

   # Millions of multiply-add operations
   # All can run in parallel
   ::


4. **Scientific Simulations**
   .. code-block:: python

   for particle_i in particles:
       force = sum(calculate_force(particle_i, particle_j)
                   for particle_j in other_particles)

   # Each particle's force is independent
   # GPU can calculate all simultaneously
   ::


**Poor for GPUs** (lots of branching/dependencies):

1. **Complex Control Flow**
   .. code-block:: python

   def complex_logic(data):
       if data.type == 'A':
           if data.value > threshold_1:
               return process_A1(data)
           else:
               return process_A2(data)
       elif data.type == 'B':
           # ... more branching

   # Different threads take different paths
   # GPU cores must wait for slowest path (divergence)
   ::


2. **Recursive Algorithms**
   .. code-block:: python

       if len(arr) <= 1:
           return arr
       pivot = arr[0]
       left = quicksort([x for x in arr[1:] if x < pivot])
       right = quicksort([x for x in arr[1:] if x >= pivot])
       return left + [pivot] + right

   # Highly sequential, data-dependent
   # Cannot parallelize effectively on GPU
   ::


3. **Database Queries with Joins**
   .. code-block:: sql

   SELECT * FROM table1
   JOIN table2 ON table1.id = table2.foreign_id
   WHERE complex_condition(table1.data)

   -- Irregular memory access
   -- Branch-heavy logic
   -- CPU better suited
   ::


---

CPU vs GPU Architecture

Silicon Real Estate Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

CPU Die Layout (approximate):
|  ############ Control Logic (40%)           |
|  Fetch, decode, branch predict, OoO, etc.  |
|  ######## Cache (30%)                       |
|  L1, L2, L3 caches                          |
|  ###### Compute Units (20%)                 |
|  ALUs, FPUs, actual computation             |
|  ## Other (10%)                             |
|  Memory controller, I/O, etc.               |

GPU Die Layout (approximate):
|  #####################################      |
|  #####################################      |
|  #####################################      |
|  #####################################      |
|  #####################################      |
|  Compute Units (80-85%)                     |
|  Thousands of simple ALUs                   |
|  #####################################      |
|  #####################################      |
|  #####################################      |
|  #####################################      |
|  ## Cache (5-10%)                           |
|  ## Control (5%)                            |
|  # Other (5%)                               |


Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

**Example: Matrix Multiplication (2048x2048)**

.. code-block:: python

import time

Generate random matrices
A = np.random.rand(2048, 2048)
B = np.random.rand(2048, 2048)

CPU (NumPy with optimized BLAS)
start = time.time()
C_cpu = np.dot(A, B)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.3f}s")

GPU (using CuPy - CUDA arrays)
import cupy as cp
A_gpu = cp.array(A)
B_gpu = cp.array(B)

start = time.time()
C_gpu = cp.dot(A_gpu, B_gpu)
cp.cuda.Stream.null.synchronize()  # Wait for GPU
gpu_time = time.time() - start
print(f"GPU time: {gpu_time:.3f}s")

print(f"Speedup: {cpu_time / gpu_time:.1f}x")

Typical results:
CPU time: 0.850s (Intel i9, 8 cores)
GPU time: 0.012s (NVIDIA RTX 4090)
Speedup: 70.8x
::


Detailed Benchmark Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

| Task | CPU (8-core i9) | GPU (RTX 4090) | Speedup |
| Matrix Multiply (2048^2) | 850 ms | 12 ms | 70x |
| Image Convolution (4K) | 1200 ms | 8 ms | 150x |
| FFT (16M points) | 2500 ms | 15 ms | 166x |
| Neural Net Forward Pass | 5000 ms | 25 ms | 200x |
| Ray Tracing (1080p frame) | 45000 ms | 16 ms | 2800x |

---

When to Use What

Decision Matrix
~~~~~~~~~~~~~~~

::


::

    |                     DECISION TREE                           |


Is the problem parallelizable?
            Reason: Massive parallelism advantage
::


Practical Guidelines
~~~~~~~~~~~~~~~~~~~~

Use CPU When:
^^^^^^^^^^^^^

::

[[OK]] Complex branching logic
   if/else trees, switch statements, dynamic dispatch

[[OK]] Irregular memory access patterns
   Hash tables, tree structures, linked lists

[[OK]] Small datasets (< 10K elements)
   GPU transfer overhead > computation time

[[OK]] Frequent host-device communication
   Need to move data between CPU/GPU often

[[OK]] Sequential dependencies
   Each step depends on previous result

[[OK]] Debugging and development
   CPU tools more mature, easier to debug
::


Use GPU When:
^^^^^^^^^^^^^

::

[[OK]] Massive data parallelism
   Same operation on millions of data points

[[OK]] Matrix operations
   Linear algebra, transformations

[[OK]] Regular memory access patterns
   Arrays, grids, uniform data structures

[[OK]] Minimal branching
   Straight-line code, vectorizable operations

[[OK]] Large datasets (> 100K elements)
   Enough work to saturate GPU cores

[[OK]] Can keep data on GPU
   Minimize CPU<->GPU transfers
::


Use Hyperthreading When:
^^^^^^^^^^^^^^^^^^^^^^^^

::

[[OK]] Workload has varied resource usage
   Some threads use ALU, others use FPU

[[OK]] Latency hiding
   Memory-bound code with cache misses

[[OK]] Server workloads
   Many small tasks with idle time

[[FAIL]] DON'T use for CPU-intensive Python
   GIL + context switching = slower
::


---

Practical Examples

Example 1: Image Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python


Approach 1: Sequential (CPU, 1 core)
import cv2
for img_path in image_paths:
    img = cv2.imread(img_path)
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    cv2.imwrite(output_path, blurred)
Time: ~300 seconds

Approach 2: CPU Multiprocessing (4 physical cores)
from concurrent.futures import ProcessPoolExecutor

def process_image(img_path):
    img = cv2.imread(img_path)
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    cv2.imwrite(output_path, blurred)

with ProcessPoolExecutor(max_workers=4) as executor:
    executor.map(process_image, image_paths)
Time: ~80 seconds (3.75x speedup)

Approach 3: GPU (CUDA)
import cupy as cp
import cupyx.scipy.ndimage as ndimage

def process_image*gpu(img_path):
    img = cv2.imread(img_path)
    img_gpu = cp.array(img)
    blurred_gpu = ndimage.gaussian_filter(img_gpu, sigma=3)
    blurred = cp.asnumpy(blurred_gpu)
    cv2.imwrite(output_path, blurred)

for img_path in image_paths:
    process_image*gpu(img_path)
Time: ~4 seconds (75x speedup!)
::


Example 2: Monte Carlo Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python


import numpy as np
import time

Sequential CPU
def estimate_pi*cpu(n):
    inside = 0
    for * in range(n):
        x, y = np.random.random(), np.random.random()
        if x_x + y_y <= 1:
            inside += 1
    return 4 * inside / n

start = time.time()
pi_estimate = estimate_pi*cpu(1*000*000*000)
print(f"CPU time: {time.time() - start:.2f}s")
Output: CPU time: 180.00s

CPU Multiprocessing (4 cores)
from concurrent.futures import ProcessPoolExecutor

def estimate_pi*parallel(n, num_workers=4):
    chunk_size = n // num_workers
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(estimate_pi*cpu, [chunk_size] * num_workers)
    return sum(results) / num_workers

start = time.time()
pi_estimate = estimate_pi*parallel(1*000*000*000, 4)
print(f"CPU parallel time: {time.time() - start:.2f}s")
Output: CPU parallel time: 47.00s (3.8x speedup)

GPU (CUDA)
==========
import cupy as cp

def estimate_pi*gpu(n):
    # Generate all random numbers at once on GPU
    points = cp.random.random((n, 2))
    # Vectorized distance calculation
    inside = cp.sum(cp.sum(points**2, axis=1) <= 1)
    return 4 * float(inside) / n

start = time.time()
pi_estimate = estimate_pi*gpu(1*000*000*000)
cp.cuda.Stream.null.synchronize()
print(f"GPU time: {time.time() - start:.2f}s")
Output: GPU time: 1.50s (120x speedup!)
::


Example 3: Neural Network Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python


import torch
import torch.nn as nn
import time

Define simple network
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

CPU Training
model_cpu = model.to('cpu')
optimizer = torch.optim.Adam(model_cpu.parameters())

start = time.time()
for epoch in range(10):
    for batch in train_loader:
        data, target = batch
        data, target = data.to('cpu'), target.to('cpu')

        optimizer.zero_grad()
        output = model_cpu(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

cpu_time = time.time() - start
print(f"CPU training time: {cpu_time:.2f}s")
Output: CPU training time: 450.00s

GPU Training
model_gpu = model.to('cuda')
optimizer = torch.optim.Adam(model_gpu.parameters())

start = time.time()
for epoch in range(10):
    for batch in train_loader:
        data, target = batch
        data, target = data.to('cuda'), target.to('cuda')

        optimizer.zero_grad()
        output = model_gpu(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

torch.cuda.synchronize()
gpu_time = time.time() - start
print(f"GPU training time: {gpu_time:.2f}s")
print(f"Speedup: {cpu_time / gpu_time:.1f}x")
Output: GPU training time: 15.00s
        Speedup: 30.0x
::


---

Summary
-------

Core Principles
~~~~~~~~~~~~~~~

1. **Physical Cores = True Parallelism**
   - Only physical cores provide true simultaneous execution
   - Thread count beyond physical cores = diminishing returns
   - For CPU-bound: optimal workers = physical cores

2. **Hyperthreading = Resource Sharing**
   - 2 threads share 1 physical core's execution units
   - ~20-40% improvement in best case (not 2x)
   - Can hurt performance for CPU-intensive tasks
   - Best for varied workloads with different resource needs

3. **GPUs = Massive Parallelism**
   - 1000s of simple cores vs few complex cores
   - Optimized for throughput, not latency
   - Perfect for data-parallel workloads
   - Trade-off: Simple operations, massive scale

4. **Architecture Determines Use Case**
   ::

   CPU: Complex tasks, few parallel streams
   GPU: Simple tasks, massive parallel streams
   HT:  Resource sharing within a core
   ::


Quick Reference
~~~~~~~~~~~~~~~

| Workload | Best Solution | Why |
| Web server (1000s connections) | CPU + Asyncio | I/O-bound, need async |
| Video encoding (1 file) | CPU + multiprocessing | CPU-bound, need all cores |
| Deep learning training | GPU | Data-parallel, matrix ops |
| Database query | CPU | Complex branching, irregular access |
| Image batch processing | GPU | Same operation, many images |
| Code compilation | CPU + multiprocessing | Complex, parallelizable |
| Scientific simulation | GPU | Numerical computation, data-parallel |
| Game physics | GPU | Many objects, same equations |

Key Takeaways
~~~~~~~~~~~~~

- **Know your hardware**: Physical cores determine max CPU parallelism
- **Match tool to task**: CPU for complex, GPU for simple-but-massive
- **Hyperthreading helps** when threads use different resources
- **Profile first**: Measure before choosing architecture
- **Consider transfer costs**: CPU<->GPU data movement is expensive
