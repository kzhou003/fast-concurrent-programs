Tutorial 08: Grouped GEMM
=========================

Overview
--------

**Grouped GEMM** (General Matrix Multiply) allows you to compute multiple independent matrix multiplications in a single kernel launch. This is essential for applications like:

- **Mixture of Experts (MoE) models** - Different experts process different tokens
- **Batched inference** - Multiple requests with different shapes
- **Sparse computations** - Non-uniform workload distribution
- **Multi-task learning** - Different tasks with different matrix sizes

Instead of launching separate kernels for each GEMM, grouped GEMM uses a **fixed number of CTAs (Cooperative Thread Arrays)** that statically schedule work across all problems.

Key Concepts
------------

Static On-Device Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

NUM*SM: tl.constexpr  # Fixed number of streaming multiprocessors
::


- Launch a fixed number of CTAs (typically equal to GPU SM count)
- Each CTA iterates through tiles across **all** GEMM problems
- Scheduling is done **on-device** at runtime
- More efficient than launching multiple separate kernels

Group Problem Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

group*a*ptrs  # Device tensor of pointers to A matrices
group*b*ptrs  # Device tensor of pointers to B matrices
group*c*ptrs  # Device tensor of pointers to C matrices
group*gemm*sizes  # Shape [group*size, 3] storing [M, N, K] for each GEMM
g*lds  # Leading dimensions [lda, ldb, ldc] for each GEMM
::


**Why device tensors?**
- Kernels can't directly access Python lists
- All metadata must be in GPU memory
- Allows dynamic problem lookup during execution

Code Walkthrough
----------------

1. Grouped GEMM Kernel (Basic Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.jit
def grouped*matmul*kernel(
    group*a*ptrs, group*b*ptrs, group*c*ptrs,
    group*gemm*sizes, g*lds, group*size,
    NUM*SM: tl.constexpr,
    BLOCK*SIZE*M: tl.constexpr,
    BLOCK*SIZE*N: tl.constexpr,
    BLOCK*SIZE*K: tl.constexpr,
):
    tile*idx = tl.program*id(0)
    last*problem*end = 0

    # Iterate through all problems
    for g in range(group*size):
        # Load problem dimensions
        gm = tl.load(group*gemm*sizes + g * 3)      # M dimension
        gn = tl.load(group*gemm*sizes + g * 3 + 1)  # N dimension
        gk = tl.load(group*gemm*sizes + g * 3 + 2)  # K dimension

        num*m*tiles = tl.cdiv(gm, BLOCK*SIZE*M)
        num*n*tiles = tl.cdiv(gn, BLOCK*SIZE*N)
        num*tiles = num*m*tiles * num*n*tiles

        # Process tiles belonging to current problem
        while (tile*idx >= last*problem*end and
               tile*idx < last*problem*end + num*tiles):
            # Get matrix pointers for this problem
            a*ptr = tl.load(group*a*ptrs + g).to(tl.pointer*type(tl.float16))
            b*ptr = tl.load(group*b*ptrs + g).to(tl.pointer*type(tl.float16))
            c*ptr = tl.load(group*c*ptrs + g).to(tl.pointer*type(tl.float16))

            # Figure out tile coordinates within this problem
            tile*idx*in*gemm = tile*idx - last*problem*end
            tile*m*idx = tile*idx*in*gemm // num*n*tiles
            tile*n*idx = tile*idx*in*gemm % num*n*tiles

            # Standard matmul computation for this tile
            # ... (compute accumulator) ...

            # Jump to next tile assigned to this CTA
            tile*idx += NUM*SM

        # Move to next problem
        last*problem*end = last*problem*end + num*tiles
::


**Scheduling Logic:**
1. Each CTA starts with ``tile*idx = program*id(0)`` (0, 1, 2, ..., NUM*SM-1)
2. Process tiles at indices: ``tile*idx``, ``tile*idx + NUM*SM``, ``tile*idx + 2*NUM*SM``, ...
3. This distributes work evenly across all SMs

2. TMA (Tensor Memory Accelerator) Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GPUs with compute capability ≥ 9.0 (Hopper and beyond):

.. code-block:: python

@triton.jit
def grouped*matmul*tma*kernel(...):
    # Create TMA descriptors for each problem
    a*desc = tl.make*tensor*descriptor(
        a*ptr,
        shape=[gm, gk],
        strides=[lda, 1],
        block*shape=[BLOCK*SIZE*M, BLOCK*SIZE*K],
    )

    # Load using TMA
    a = a*desc.load([offs*am, offs*k])
    b = b*desc.load([offs*bn, offs*k])
    accumulator = tl.dot(a, b.T, accumulator)
::


**TMA Benefits:**
- Hardware-accelerated memory transfers
- Better memory coalescing
- Reduced register pressure
- Automatic boundary handling

3. Host Function Setup
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

def group*gemm*fn(group*A, group*B):
    group*size = len(group*A)

    # Collect matrix metadata
    A*addrs = []
    B*addrs = []
    C*addrs = []
    g*sizes = []
    g*lds = []

    for i in range(group*size):
        A = group*A[i]
        B = group*B[i]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=device, dtype=A.dtype)

        # Store pointers as Python integers
        A*addrs.append(A.data*ptr())
        B*addrs.append(B.data*ptr())
        C*addrs.append(C.data*ptr())

        # Store dimensions and strides
        g*sizes += [M, N, K]
        g*lds += [A.stride(0), B.stride(0), C.stride(0)]

    # Convert to device tensors
    d*a*ptrs = torch.tensor(A*addrs, device=device)
    d*b*ptrs = torch.tensor(B*addrs, device=device)
    d*c*ptrs = torch.tensor(C*addrs, device=device)
    d*g*sizes = torch.tensor(g*sizes, dtype=torch.int32, device=device)
    d*g*lds = torch.tensor(g*lds, dtype=torch.int32, device=device)

    # Fixed grid size
    grid = lambda META: (META['NUM*SM'], )
    grouped*matmul*kernel`grid <
        d*a*ptrs, d*b*ptrs, d*c*ptrs,
        d*g*sizes, d*g*lds, group*size,
    >`_

    return group*C
::


**Key Points:**
- Grid size is **fixed** (NUM*SM CTAs), not dependent on problem size
- All metadata lives in GPU memory
- Python list of output matrices returned

Auto-tuning Configurations
--------------------------

.. code-block:: python

@triton.autotune(
    configs=[
        triton.Config({'BLOCK*SIZE*M': 128, 'BLOCK*SIZE*N': 128,
                       'BLOCK*SIZE*K': 32, 'NUM*SM': 84}),
        triton.Config({'BLOCK*SIZE*M': 64, 'BLOCK*SIZE*N': 64,
                       'BLOCK*SIZE*K': 32, 'NUM*SM': 128}),
        triton.Config({'BLOCK*SIZE*M': 128, 'BLOCK*SIZE*N': 128,
                       'BLOCK*SIZE*K': 64, 'NUM*SM': num*sms()}),
    ],
    key=['group*size'],
)
::


**Configuration Choices:**
- ``NUM*SM``: Number of CTAs to launch (84, 128, or actual SM count)
- Block sizes: Smaller blocks = better load balancing, larger = better throughput
- Auto-tuning finds optimal configuration for your GPU

Memory Layout Considerations
----------------------------

Pointer Indirection
~~~~~~~~~~~~~~~~~~~
::

group*a*ptrs (on GPU) → [ptr0, ptr1, ptr2, ...] → A matrices (on GPU)
                         ↓     ↓     ↓
                        A0    A1    A2
::


- Two-level indirection: array of pointers, then actual data
- Overhead is minimal for large matrices
- Allows arbitrary matrix shapes and layouts

Leading Dimension (Stride)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

For row-major matrix A[M, K]:
=============================
lda = A.stride(0)  # Usually equals K for contiguous tensors

Element at (i, j) is at: base*ptr + i * lda + j
===============================================
::


Performance Characteristics
---------------------------

When Grouped GEMM Wins
~~~~~~~~~~~~~~~~~~~~~~

✅ **Good cases:**
- Multiple small-to-medium GEMMs (hundreds to thousands of elements)
- Variable problem sizes (not batched)
- GPU utilization is low with separate launches
- Total compute justifies kernel overhead

When to Use Separate Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

❌ **Bad cases:**
- Very few problems (< 10) with large matrices
- All matrices are the same size (use batched GEMM instead)
- Problems are too small (< 64×64)

Benchmarking Results
~~~~~~~~~~~~~~~~~~~~

From the script:
.. code-block:: python

@triton.testing.perf*report(
    x*names=['N'],
    x*vals=[2**i for i in range(7, 11)],  # 128 to 1024
    line*vals=['cublas', 'triton', 'triton-tma'],
)
::


**Typical results:**
- Grouped GEMM competitive with cuBLAS for moderate sizes
- TMA version faster on Hopper (compute capability 9.0+)
- Overhead becomes negligible for N ≥ 256

GPU Architecture Insights
-------------------------

Why Fixed Number of CTAs?
~~~~~~~~~~~~~~~~~~~~~~~~~

::

Traditional approach: Launch one CTA per tile
- Problem 1: 10 tiles → 10 CTAs
- Problem 2: 20 tiles → 20 CTAs
- Total: 30 kernel launches or 30 CTAs

Grouped GEMM: Launch NUM*SM CTAs total
- 84 CTAs process all 30 tiles
- Single kernel launch
- Better SM utilization
::


Work Distribution
~~~~~~~~~~~~~~~~~

::

SM 0: Tiles 0, 84, 168, ...
SM 1: Tiles 1, 85, 169, ...
SM 2: Tiles 2, 86, 170, ...
...
SM 83: Tiles 83, 167, ...
::


- Round-robin distribution
- Automatic load balancing
- Handles variable problem sizes gracefully

Advanced: TMA Descriptor Creation
---------------------------------

For TMA version:
.. code-block:: python

a*desc = tl.make*tensor*descriptor(
    a*ptr,                    # Base pointer
    shape=[gm, gk],          # Logical tensor shape
    strides=[lda, 1],        # Row-major stride
    block*shape=[BM, BK],    # Block to load
)

Load a block starting at (offs*am, offs*k)
==========================================
a = a*desc.load([offs*am, offs*k])
::


**Benefits over manual loads:**
- Hardware manages bounds checking
- Better memory coalescing
- Reduced register usage
- Simplified kernel code

Common Pitfalls
---------------

1. Forgetting Contiguity
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Bad: B might not be contiguous after transpose
==============================================
B = B.T
grouped*matmul*kernel`grid <...>`_

Good: Ensure contiguity
=======================
B*T = B.T.contiguous()
::


2. Incorrect Leading Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Wrong: Assuming contiguous
==========================
lda = K

Correct: Use actual stride
==========================
lda = A.stride(0)
::


3. Mixed Precision Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

FP8 requires compute capability >= 9.0
======================================
if dtype == torch.float8*e4m3fn:
    assert torch.cuda.get*device*capability()[0] >= 9
::


Practical Example
-----------------

.. code-block:: python

Define different problem sizes
==============================
group*m = [1024, 512, 256, 128]
group*n = [1024, 512, 256, 128]
group*k = [1024, 512, 256, 128]

Create random matrices
======================
group*A = []
group*B = []
for i in range(len(group*m)):
    M, N, K = group*m[i], group*n[i], group*k[i]
    A = torch.rand((M, K), device='cuda', dtype=torch.float16)
    B = torch.rand((K, N), device='cuda', dtype=torch.float16)
    group*A.append(A)
    group*B.append(B)

Compute all GEMMs in one kernel launch
======================================
tri*out = group*gemm*fn(group*A, group*B)

Verify against PyTorch
======================
ref*out = [torch.matmul(a, b) for a, b in zip(group*A, group*B)]
for i in range(len(group*m)):
    assert torch.allclose(ref*out[i], tri_out[i], atol=1e-2)
::


Summary
-------

**Grouped GEMM** is a powerful technique for computing multiple independent matrix multiplications efficiently:

- **Static scheduling** on device avoids multiple kernel launches
- **Fixed number of CTAs** improves SM utilization
- **TMA support** for Hopper+ GPUs provides additional speedup
- **Auto-tuning** finds optimal block sizes
- **Competitive with cuBLAS** for moderate problem sizes

**Use cases:**
- Mixture of Experts models
- Variable-size batched inference
- Sparse neural networks
- Multi-task learning

**Performance tips:**
- Use TMA version on Hopper+ GPUs
- Auto-tune for your specific problem sizes
- Ensure matrices are contiguous
- Profile to verify GPU utilization

This technique is essential for modern ML workloads where you need to process many different-sized operations efficiently!
