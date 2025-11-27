Matrix Multiplication in Triton
===============================

Overview
--------
Matrix multiplication (GEMM: General Matrix Multiply) is the cornerstone of deep learning. This tutorial shows how to write a high-performance matmul kernel that rivals cuBLAS/rocBLAS. You'll learn advanced GPU optimization techniques including **tiling**, **cache optimization**, and **auto-tuning**.

What You'll Learn
-----------------
- **Block-level tiling** for matrix multiplication
- **Multi-dimensional pointer arithmetic**
- **L2 cache optimization** through program reordering
- **Auto-tuning** for performance optimization
- Why matmul is **compute-bound** (not memory-bound)
- **Tensor Cores** and specialized hardware

The Matrix Multiplication Problem
---------------------------------

Basic Algorithm
~~~~~~~~~~~~~~~
Compute C = A × B where:
- A has shape (M, K)
- B has shape (K, N)
- C has shape (M, N)

.. code-block:: python

for i in range(M):
    for j in range(N):
        C[i, j] = sum(A[i, k] * B[k, j] for k in range(K))
::


Time complexity: **O(M × N × K)**
For 1024×1024 matrices: ~1 billion operations!

Why It's Hard to Optimize
~~~~~~~~~~~~~~~~~~~~~~~~~

**Naive implementation problems**:
1. Poor memory locality
2. Inefficient use of cache
3. Not utilizing GPU parallelism
4. Missing out on specialized hardware (Tensor Cores)

**Our goal**: Achieve 10+ TFLOPS (Tera-FLOPs, trillions of operations per second)!

GPU Matrix Multiplication Strategy
----------------------------------

Blocked Algorithm
~~~~~~~~~~~~~~~~~
Instead of computing one element at a time, we compute **blocks** of C:

.. code-block:: python

Pseudocode for our kernel
=========================
for m in range(0, M, BLOCK*M):           # Parallel on GPU
    for n in range(0, N, BLOCK*N):       # Parallel on GPU
        accumulator = zeros(BLOCK*M, BLOCK*N)
        for k in range(0, K, BLOCK*K):   # Sequential within each block
            A*block = load A[m:m+BLOCK*M, k:k+BLOCK*K]
            B*block = load B[k:k+BLOCK*K, n:n+BLOCK*N]
            accumulator += dot(A*block, B*block)
        store accumulator to C[m:m+BLOCK*M, n:n+BLOCK*N]
::


**Key insight**: The outer two loops are parallelized across GPU programs, while the K loop is sequential within each program.

Why Tiling Works
~~~~~~~~~~~~~~~~

**Memory hierarchy**:
::

Registers (fastest, ~1 cycle)
    ↓
L1/Shared Memory (~10 cycles, 192 KB)
    ↓
L2 Cache (~100 cycles, 40 MB)
    ↓
HBM/Global Memory (slowest, ~400 cycles, 40-80 GB)
::


**With tiling**:
1. Load A*block into shared memory (reused BLOCK*N times)
2. Load B*block into shared memory (reused BLOCK*M times)
3. Accumulate into registers (reused K/BLOCK*K times)
4. Write result once to global memory

**Data reuse** = fewer slow memory accesses!

Pointer Arithmetic in 2D
------------------------

Understanding Strides
~~~~~~~~~~~~~~~~~~~~~

For a row-major matrix A[M, K]:
.. code-block:: python

A[i, j] = *(A*ptr + i*stride*M + j*stride*K)
::


Where:
- ``stride*M`` = number of elements to next row = K
- ``stride*K`` = number of elements to next column = 1

Example for A[4, 3]:
::

Elements in memory: [a00, a01, a02, a10, a11, a12, a20, a21, a22, a30, a31, a32]
A[2, 1] = A*ptr + 2*3 + 1*1 = A*ptr[7] = a21 ✓
::


Block Pointers
~~~~~~~~~~~~~~

To get pointers to a block A[m:m+BLOCK*M, k:k+BLOCK*K]:

.. code-block:: python

offs*am = (pid*m * BLOCK*M + tl.arange(0, BLOCK*M)) % M
offs*k = tl.arange(0, BLOCK*K)
a*ptrs = a*ptr + offs*am[:, None]*stride*am + offs*k[None, :]*stride*ak
::


**Broadcasting**:
- ``offs*am[:, None]`` has shape (BLOCK*M, 1)
- ``offs*k[None, :]`` has shape (1, BLOCK*K)
- Result ``a*ptrs`` has shape (BLOCK*M, BLOCK*K)

**Example** (BLOCK*M=2, BLOCK*K=2, pid*m=0):
.. code-block:: python

offs*am = [0, 1]        # shape (2,)
offs*k = [0, 1]          # shape (2,)

offs*am[:, None] = [[0],   # shape (2, 1)
                    [1]]

offs*k[None, :] = [[0, 1]]  # shape (1, 2)

Broadcasting multiplication:
============================
offs*am[:, None] * stride*am + offs*k[None, :] * stride*ak
= [[0*stride*am + 0*stride*ak, 0*stride*am + 1*stride*ak],
   [1*stride*am + 0*stride*ak, 1*stride*am + 1*stride*ak]]
::


This gives pointers to a 2×2 block!

Advancing Pointers
~~~~~~~~~~~~~~~~~~

To move to next K block:
.. code-block:: python

a*ptrs += BLOCK*K * stride*ak
b*ptrs += BLOCK*K * stride*bk
::


This shifts all pointers in the block by BLOCK*K positions in the K dimension.

L2 Cache Optimization
---------------------

The Problem with Row-Major Ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we process blocks in simple row-major order:
::

Program 0 → Block C[0, 0]  (needs A[0, :] and B[:, 0])
Program 1 → Block C[0, 1]  (needs A[0, :] and B[:, 1])
...
Program 9 → Block C[1, 0]  (needs A[1, :] and B[:, 0])
::


**Issue**: By the time we compute C[1, 0], B[:, 0] might be evicted from L2 cache!

Grouped Ordering (Swizzling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead, process blocks in groups:
.. code-block:: python

num*pid*in*group = GROUP*SIZE*M * num*pid*n
group*id = pid // num*pid*in*group
first*pid*m = group*id * GROUP*SIZE*M
group*size*m = min(num*pid*m - first*pid*m, GROUP*SIZE*M)
pid*m = first*pid*m + ((pid % num*pid*in*group) % group*size*m)
pid*n = (pid % num*pid*in*group) // group*size*m
::


**Visual Example** (GROUP*SIZE*M=3, 9×9 blocks):

Row-major order:
::

0→ 1→ 2→ 3→ 4→ 5→ 6→ 7→ 8→
9→ 10→ ...
::

Needs to load: 90 unique blocks into SRAM

Grouped order:
::

0↓ 3↓ 6↓ 1↓ 4↓ 7↓ 2↓ 5↓ 8↓
9↓ 12↓ 15↓ 10↓ 13↓ 16↓ 11↓ 14↓ 17↓
...
::

Needs to load: 54 unique blocks into SRAM

**Savings**: 40% fewer loads from HBM! This can improve performance by 10-20%.

Auto-Tuning
-----------

The Configuration Space
~~~~~~~~~~~~~~~~~~~~~~~

Many parameters affect performance:
- ``BLOCK*SIZE*M``: 32, 64, 128, 256
- ``BLOCK*SIZE*N``: 32, 64, 128, 256
- ``BLOCK*SIZE*K``: 16, 32, 64, 128
- ``GROUP*SIZE*M``: 4, 8, 16
- ``num*warps``: 2, 4, 8, 16
- ``num*stages``: 2, 3, 4, 5

Total combinations: ~1000+

**Problem**: Optimal configuration depends on:
- Matrix sizes (M, N, K)
- GPU architecture (compute capability, SRAM size, etc.)
- Data types (fp16, fp32, int8, etc.)

Triton's Auto-Tuner
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.autotune(
    configs=get*autotune*config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul*kernel(...):
::


**How it works**:
1. Define list of candidate configurations
2. Specify key parameters (M, N, K)
3. At runtime, Triton:
   - Tries each configuration
   - Measures performance
   - Caches best config for this (M, N, K)
   - Uses cached result for future calls

**Example config**:
.. code-block:: python

triton.Config(
    {'BLOCK*SIZE*M': 128, 'BLOCK*SIZE*N': 256, 'BLOCK*SIZE*K': 64, 'GROUP*SIZE*M': 8},
    num*stages=3,
    num*warps=8
)
::


Good Configurations
~~~~~~~~~~~~~~~~~~~

**For NVIDIA (CUDA)**:
- Large blocks (128×256) for large matrices
- More stages (3-4) to hide latency
- 8 warps for good occupancy

**For AMD (HIP)**:
- Medium blocks (64×64, 128×128)
- Fewer stages (2) due to different architecture
- Different thread group sizes

**FP8 (8-bit floating point)**:
- Even larger blocks (256×256)
- Can fit more data in SRAM
- Tensor Cores process 4x more data per cycle

The Kernel Implementation
-------------------------

Step 1: Compute Program IDs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

pid = tl.program*id(axis=0)
num*pid*m = tl.cdiv(M, BLOCK*SIZE*M)
num*pid*n = tl.cdiv(N, BLOCK*SIZE*N)
::


Map linear program ID to 2D (pid*m, pid*n) using grouped ordering.

Step 2: Initialize Pointers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

offs*am = (pid*m * BLOCK*SIZE*M + tl.arange(0, BLOCK*SIZE*M)) % M
offs*bn = (pid*n * BLOCK*SIZE*N + tl.arange(0, BLOCK*SIZE*N)) % N
offs*k = tl.arange(0, BLOCK*SIZE*K)

a*ptrs = a*ptr + (offs*am[:, None]*stride*am + offs*k[None, :]*stride*ak)
b*ptrs = b*ptr + (offs*k[:, None]*stride*bk + offs*bn[None, :]*stride*bn)
::


Create pointers for first blocks of A and B.

**Note the modulo**:
- ``offs*am % M`` handles M not being multiple of BLOCK*SIZE*M
- Wraps around, so we load valid (though repeated) data
- Doesn't matter because we mask the output store

Step 3: Accumulation Loop
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

accumulator = tl.zeros((BLOCK*SIZE*M, BLOCK*SIZE*N), dtype=tl.float32)

for k in range(0, tl.cdiv(K, BLOCK*SIZE*K)):
    a = tl.load(a*ptrs, mask=offs*k[None, :] < K - k*BLOCK*SIZE*K, other=0.0)
    b = tl.load(b*ptrs, mask=offs*k[:, None] < K - k*BLOCK*SIZE*K, other=0.0)

    accumulator = tl.dot(a, b, accumulator)

    a*ptrs += BLOCK*SIZE*K * stride*ak
    b*ptrs += BLOCK*SIZE*K * stride*bk
::


**Key points**:
- Accumulate in ``float32`` for numerical accuracy
- Mask loads when K not multiple of BLOCK*SIZE*K
- ``tl.dot()`` uses hardware accelerators (Tensor Cores on NVIDIA)

Step 4: Apply Activation (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

if ACTIVATION == "leaky*relu":
    accumulator = leaky*relu(accumulator)
c = accumulator.to(tl.float16)
::


**Kernel fusion**: Apply activation while data is in registers (fast)!

Step 5: Store Result
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

offs*cm = pid*m * BLOCK*SIZE*M + tl.arange(0, BLOCK*SIZE*M)
offs*cn = pid*n * BLOCK*SIZE*N + tl.arange(0, BLOCK*SIZE*N)
c*ptrs = c*ptr + stride*cm*offs*cm[:, None] + stride*cn*offs*cn[None, :]
c*mask = (offs*cm[:, None] < M) & (offs*cn[None, :] < N)
tl.store(c*ptrs, c, mask=c*mask)
::


**Masking** ensures we don't write out of bounds.

Tensor Cores
------------

What Are Tensor Cores?
~~~~~~~~~~~~~~~~~~~~~~

Special hardware units on modern GPUs for matrix multiplication:
- **NVIDIA**: Tensor Cores (Volta, Turing, Ampere, Hopper)
- **AMD**: Matrix Cores (CDNA2, CDNA3)

**Performance**:
- Regular CUDA cores: 1 FP16 multiply-add per cycle per core
- Tensor Cores: 64-256 FP16 multiply-adds per cycle per core
- **Speedup**: 10-100x for matmul!

How Tensor Cores Work
~~~~~~~~~~~~~~~~~~~~~

Operate on small matrices (e.g., 16×16):
::

D = A × B + C
::

Where A, B, C, D are 16×16 matrices.

**Single instruction**, massive computation!

Triton and Tensor Cores
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

accumulator = tl.dot(a, b, accumulator)
::


When you use ``tl.dot()`` with appropriate types and sizes, Triton automatically:
1. Detects Tensor Core availability
2. Arranges data in Tensor Core format
3. Emits Tensor Core instructions (e.g., ``mma.sync``)

**Requirements for Tensor Cores**:
- FP16, BF16, TF32, FP8, or INT8 inputs
- Block sizes that are multiples of Tensor Core dimensions (usually 16)

Performance Analysis
--------------------

Arithmetic Intensity
~~~~~~~~~~~~~~~~~~~~

For matmul C = A × B:
- **FLOPs**: 2MNK (each output element: K multiplies + K adds)
- **Memory**: 2(MK + KN + MN) bytes (read A, B, write C)

Arithmetic Intensity = 2MNK / (2(MK + KN + MN))

For square matrices (M=N=K):
::

AI = 2N³ / (4N²) = N/2
::


For N=1024: AI = 512 FLOPs/byte

**This is very high!** Matmul is **compute-bound**, not memory-bound.

Roofline Model
~~~~~~~~~~~~~~

Performance is limited by:
::

min(Peak*Compute, Peak*Memory*BW * AI)
::


For A100:
- Peak FP16 Tensor Core: 312 TFLOPS
- Peak Memory BW: 2 TB/s
- For N=1024: 2 TB/s * 512 = 1024 TFLOPS

**Bottleneck**: Compute (312 TFLOPS), not memory!

For small matrices (N=64): AI = 32
- Memory bound: 2 TB/s * 32 = 64 TFLOPS
- Can achieve much less than peak compute

Expected Performance
~~~~~~~~~~~~~~~~~~~~

**Theoretical Peak** (A100, FP16):
- 312 TFLOPS with Tensor Cores

**Achievable**:
- cuBLAS: ~280-300 TFLOPS (90-95% peak)
- Triton (optimized): ~250-280 TFLOPS (80-90% peak)
- Naive implementation: ~10-50 TFLOPS (<20% peak)

**Why not 100%?**
- Launch overhead
- Imperfect tiling (boundaries)
- L2 cache misses
- Pipeline stalls

Advanced Optimizations
----------------------

Software Pipelining
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

num*stages = 3
::


**Idea**: Overlap memory loads with computation

Without pipelining:
::

Load A, B → Wait → Compute → Load A, B → Wait → Compute
::


With 3-stage pipelining:
::

Stage 0: Load A₀, B₀
Stage 1: Load A₁, B₁ | Compute with A₀, B₀
Stage 2: Load A₂, B₂ | Compute with A₁, B₁
Stage 3: Load A₃, B₃ | Compute with A₂, B₂
...
::


**Benefit**: Computation and memory loads happen simultaneously!

**Cost**: Need more registers and SRAM to hold multiple stages.

Loop Unrolling
~~~~~~~~~~~~~~

Triton automatically unrolls the K loop when possible:
.. code-block:: python

for k in range(0, tl.cdiv(K, BLOCK*SIZE*K)):
    accumulator = tl.dot(a, b, accumulator)
::


Becomes (if K/BLOCK*K = 4):
.. code-block:: python

accumulator = tl.dot(a0, b0, accumulator)
accumulator = tl.dot(a1, b1, accumulator)
accumulator = tl.dot(a2, b2, accumulator)
accumulator = tl.dot(a3, b3, accumulator)
::


**Benefits**:
- Eliminates loop overhead
- Better instruction-level parallelism
- Easier for compiler to optimize

Register Pressure
~~~~~~~~~~~~~~~~~

Each thread needs registers for:
- A block elements
- B block elements
- Accumulator elements
- Temporary variables

**Example** (BLOCK*M=128, BLOCK*N=128, BLOCK*K=32, 8 warps):
- Threads per block: 8 * 32 = 256
- Elements per thread: (128*128) / 256 = 64 for accumulator
- Registers needed: ~100-150 per thread

**Limited supply**: 65536 registers per SM on A100

If too many registers → fewer blocks per SM → lower occupancy → lower performance.

**Balance**: Larger blocks = more reuse but more register pressure.

Common Pitfalls
---------------

1. Non-Contiguous Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

assert a.is*contiguous(), "Matrix A must be contiguous"
::


Non-contiguous tensors have unexpected strides → wrong pointer arithmetic → incorrect results.

**Solution**: Call ``.contiguous()`` or handle arbitrary strides.

2. Wrong Stride Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

a.stride(0), a.stride(1)  # Correct
::


Don't hardcode strides! Transposed matrices have different strides.

3. Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

offs*am = (...) % M  # Modulo for safety
c*mask = (offs*cm[:, None] < M) & (offs*cn[None, :] < N)  # Mask stores
::


Forgetting these → out-of-bounds accesses → crashes or wrong results.

4. Numerical Precision
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

accumulator = tl.zeros((BLOCK*SIZE*M, BLOCK*SIZE*N), dtype=tl.float32)
::


Using FP16 for accumulation → loss of precision → degraded accuracy.

**Best practice**: Accumulate in FP32, cast to FP16 for storage.

Benchmarking Tips
-----------------

1. Warm-up
~~~~~~~~~~
.. code-block:: python

triton.testing.do*bench(fn)  # Automatically does warm-up
::


First few kernel launches are slow (compilation, cache loading).

2. Measure TFLOPS
~~~~~~~~~~~~~~~~~
.. code-block:: python

perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
::


**Why 2MNK?**
- Each of MN output elements: K multiply-adds
- Multiply-add = 2 FLOPs (1 multiply + 1 add)
- Total: 2MNK FLOPs

3. Compare Against cuBLAS/rocBLAS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

torch.matmul(a, b)  # Uses cuBLAS on NVIDIA, rocBLAS on AMD
::


These are **highly optimized** by vendors. Matching them is a huge achievement!

4. Test Different Sizes
~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

x*vals=[128 * i for i in range(2, 33)]  # 256, 384, ..., 4096
::


Performance varies with size:
- Small: Memory-bound, lower TFLOPS
- Medium: Transition zone
- Large: Compute-bound, approaching peak

Key Takeaways
-------------

1. **Tiling/Blocking is essential**: Reuse data in fast SRAM
2. **Pointer arithmetic**: Understanding strides is crucial for multi-dimensional arrays
3. **L2 cache matters**: Grouped ordering can give 10-20% speedup
4. **Auto-tuning is powerful**: Optimal configs vary with size and hardware
5. **Tensor Cores are game-changers**: 10-100x speedup for matmul
6. **Matmul is compute-bound**: Unlike vector add (memory-bound)
7. **Accumulate in FP32**: Maintain numerical accuracy
8. **Triton simplifies complex optimizations**: Achieves near-cuBLAS performance with readable code

Matrix multiplication is the foundation of deep learning. Mastering these concepts will help you understand and optimize transformers, CNNs, and other neural architectures!
