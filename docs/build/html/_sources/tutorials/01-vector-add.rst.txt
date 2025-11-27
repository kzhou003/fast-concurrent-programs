Vector Addition in Triton
=========================

Overview
--------
Vector addition is the fundamental "Hello World" of GPU programming. This tutorial demonstrates how to write a simple element-wise vector addition kernel in Triton and introduces core GPU programming concepts.

What You'll Learn
-----------------
- The basic programming model of GPU parallel computing
- How Triton kernels are structured using ``@triton.jit``
- The SPMD (Single Program, Multiple Data) execution model
- Memory access patterns and coalescing
- How to benchmark GPU kernels

GPU/CUDA Concepts
-----------------

SPMD (Single Program, Multiple Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unlike traditional CPU programming where you write a loop to process all elements, GPU programming uses the SPMD model:
- You write code that processes **one block** of data
- The GPU launches **thousands of copies** of this program in parallel
- Each copy (called a "program" in Triton, or "thread block" in CUDA) processes a different portion of the data

Program ID and Block Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the kernel:
.. code-block:: python

pid = tl.program*id(axis=0)  # Get this program's unique ID
block*start = pid * BLOCK*SIZE
offsets = block*start + tl.arange(0, BLOCK*SIZE)
::


If you have a vector of 256 elements and ``BLOCK*SIZE=64``:
- Program 0 processes elements [0:64]
- Program 1 processes elements [64:128]
- Program 2 processes elements [128:192]
- Program 3 processes elements [192:256]

Grid Size
~~~~~~~~~
The "grid" determines how many program instances run in parallel:
.. code-block:: python

grid = lambda meta: (triton.cdiv(n*elements, meta['BLOCK*SIZE']), )
::


This calculates: ``num*programs = ceil(n*elements / BLOCK*SIZE)``

Memory Hierarchy
~~~~~~~~~~~~~~~~
GPUs have a multi-level memory hierarchy:
1. **Global Memory (DRAM)** - Large but slow (hundreds of GB/s bandwidth)
2. **L2 Cache** - Medium size, faster (TBs/s)
3. **L1 Cache/Shared Memory (SRAM)** - Small but very fast (10+ TB/s)
4. **Registers** - Fastest, per-thread storage

Memory Coalescing
~~~~~~~~~~~~~~~~~
When loading data from DRAM:
.. code-block:: python

x = tl.load(x*ptr + offsets, mask=mask)
::


Triton automatically coalesces these loads. In CUDA terms, this means:
- Adjacent threads load adjacent memory addresses
- The memory controller combines multiple loads into fewer transactions
- This is crucial for achieving high bandwidth

Masking for Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
What if your vector size isn't a perfect multiple of BLOCK*SIZE?
.. code-block:: python

mask = offsets < n*elements
x = tl.load(x*ptr + offsets, mask=mask)
::


- The mask ensures out-of-bounds elements aren't loaded
- This prevents memory access violations
- In CUDA, you'd typically use conditional statements for this

How the Kernel Works
--------------------

Step-by-Step Execution
~~~~~~~~~~~~~~~~~~~~~~

1. **Get Program ID**
   .. code-block:: python

   pid = tl.program*id(axis=0)
   ::

   Each parallel program instance gets a unique ID (0, 1, 2, ...).

2. **Calculate Block Offsets**
   .. code-block:: python

   block*start = pid * BLOCK*SIZE
   offsets = block*start + tl.arange(0, BLOCK*SIZE)
   ::

   Determines which elements this program will process.

3. **Create Boundary Mask**
   .. code-block:: python

   mask = offsets < n*elements
   ::

   Handles cases where input size isn't a multiple of BLOCK*SIZE.

4. **Load Data from DRAM to Registers**
   .. code-block:: python

   x = tl.load(x*ptr + offsets, mask=mask)
   y = tl.load(y*ptr + offsets, mask=mask)
   ::

   Brings data from slow global memory to fast registers.

5. **Compute Result**
   .. code-block:: python

   output = x + y
   ::

   Performs the actual addition in registers (extremely fast).

6. **Store Result Back to DRAM**
   .. code-block:: python

   tl.store(output*ptr + offsets, output, mask=mask)
   ::

   Writes the result back to global memory.

Performance Characteristics
---------------------------

Bandwidth-Bound Operation
~~~~~~~~~~~~~~~~~~~~~~~~~
Vector addition is **memory-bound**, not **compute-bound**:
- **Compute**: Just one addition per element (trivial)
- **Memory**: Must read 2 values and write 1 value (3x data movement)

Arithmetic Intensity
~~~~~~~~~~~~~~~~~~~~
Arithmetic Intensity = FLOPs / Bytes Transferred
- For vector add: 1 FLOP / 12 bytes (assuming float32) = 0.083 FLOPs/byte
- Very low! Most time is spent waiting for memory.

Theoretical Performance
~~~~~~~~~~~~~~~~~~~~~~~
For a GPU with 1 TB/s memory bandwidth:
::

Max GB/s = Memory Bandwidth = 1000 GB/s
For vector add: Need to move 3 * 4 bytes per element = 12 bytes
Max elements/s = 1000 GB/s / 12 bytes ≈ 83 billion elements/s
::


Triton-Specific Features
------------------------

``@triton.jit`` Decorator
~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

@triton.jit
def add*kernel(...):
::

- Marks the function as a Triton kernel
- Triton JIT-compiles this to optimized GPU code
- Automatically handles type inference and optimization

``constexpr`` for Compile-Time Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

BLOCK*SIZE: tl.constexpr
::

- Tells Triton this value is known at compile time
- Allows better optimizations (loop unrolling, etc.)
- In CUDA, this is like template parameters

Launch Grid Syntax
~~~~~~~~~~~~~~~~~~
.. code-block:: python

add*kernel`grid <x, y, output, n*elements, BLOCK*SIZE=1024>`_
::

- ``[grid]`` specifies how many program instances to launch
- Triton automatically converts torch tensors to pointers
- Meta-parameters (like BLOCK*SIZE) are passed as keywords

Benchmarking Insights
---------------------

The benchmark code measures **GB/s (Gigabytes per second)**:
.. code-block:: python

gbps = lambda ms: 3 * x.numel() * x.element*size() * 1e-9 / (ms * 1e-3)
::


Why multiply by 3?
- Read x: ``n*elements * 4 bytes``
- Read y: ``n*elements * 4 bytes``
- Write output: ``n*elements * 4 bytes``
- Total: ``3 * n*elements * 4 bytes``

Expected Results
~~~~~~~~~~~~~~~~
For modern GPUs:
- **Peak memory bandwidth**: 500-2000 GB/s
- **Typical achieved bandwidth**: 60-80% of peak
- Both PyTorch and Triton should achieve similar performance (both are memory-bound)

Common Patterns You'll See
--------------------------

1. Pointer Arithmetic
~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

x*ptr + offsets
::

In Triton, pointers are just memory addresses, and you can add offsets to them.

2. Vectorized Operations
~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

output = x + y
::

This single line actually adds BLOCK*SIZE elements in parallel!

3. Masked Memory Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

tl.load(ptr, mask=mask)
tl.store(ptr, value, mask=mask)
::

Essential for handling non-uniform sizes safely.

Key Takeaways
-------------

1. **GPU programming uses SPMD**: Write code for one block, GPU runs many copies in parallel
2. **Block size matters**: Too small = overhead, too large = wasted resources
3. **Memory is the bottleneck**: For simple operations like vector add, memory bandwidth limits performance
4. **Masking is essential**: Handle boundary conditions properly to avoid crashes
5. **Triton abstracts complexity**: You don't need to write raw CUDA to get good performance

Comparison to CUDA
------------------

If you were to write this in CUDA, you'd need:
- Explicit block/thread indexing (``blockIdx.x``, ``threadIdx.x``)
- Manual memory management (cudaMalloc, cudaMemcpy)
- Kernel launch syntax (``kernel<<<grid, block>>>``)
- Error checking boilerplate

Triton handles most of this automatically while still achieving similar performance!

Next Steps
----------

- Try different BLOCK_SIZE values (powers of 2 work best)
- Experiment with different input sizes
- Look at the generated assembly with ``triton.tools.disasm()``
- Profile with ``nvprof`` or ``nsys`` to see memory bandwidth utilization
