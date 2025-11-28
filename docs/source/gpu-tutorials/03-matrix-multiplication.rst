Matrix Multiplication in Triton

Overview
--------
Matrix multiplication (GEMM: General Matrix Multiply) is the cornerstone of deep learning. This tutorial shows how to write a high-performance matmul kernel that rivals cuBLAS/rocBLAS. You'll learn advanced GPU optimization techniques including **tiling**, **cache optimization**, and **auto-tuning**.

What You'll Learn
- **Block-level tiling** for matrix multiplication
- **Multi-dimensional pointer arithmetic**
- **L2 cache optimization** through program reordering
- **Auto-tuning** for performance optimization
- Why matmul is **compute-bound** (not memory-bound)
- **Tensor Cores** and specialized hardware

The Matrix Multiplication Problem

Basic Algorithm
~~~~~~~~~~~~~~~
Compute C = A x B where:
- A has shape (M, K)
- B has shape (K, N)
- C has shape (M, N)

.. code-block:: python

for i in range(M):
    for j in range(N):
        C[i, j] = sum(A[i, k] * B[k, j] for k in range(K))
::


Time complexity: **O(M x N x K)**
For 1024x1024 matrices: ~1 billion operations!

Why It's Hard to Optimize
~~~~~~~~~~~~~~~~~~~~~~~~~

**Naive implementation problems**:
1. Poor memory locality
2. Inefficient use of cache
3. Not utilizing GPU parallelism
4. Missing out on specialized hardware (Tensor Cores)

**Our goal**: Achieve 10+ TFLOPS (Tera-FLOPs, trillions of operations per second)!

GPU Matrix Multiplication Strategy

Blocked Algorithm
~~~~~~~~~~~~~~~~~
Instead of computing one element at a time, we compute **blocks** of C:

.. code-block:: python

Pseudocode for our kernel
for m in range(0, M, BLOCK_M):           # Parallel on GPU
    for n in range(0, N, BLOCK_N):       # Parallel on GPU
        accumulator = zeros(BLOCK_M, BLOCK_N)
        for k in range(0, K, BLOCK_K):   # Sequential within each block
            A_block = load A[m:m+BLOCK_M, k:k+BLOCK_K]
            B_block = load B[k:k+BLOCK_K, n:n+BLOCK_N]
            accumulator += dot(A_block, B_block)
        store accumulator to C[m:m+BLOCK_M, n:n+BLOCK_N]
::


**Key insight**: The outer two loops are parallelized across GPU programs, while the K loop is sequential within each program.

Why Tiling Works
~~~~~~~~~~~~~~~~

**Memory hierarchy**:
::

Registers (fastest, ~1 cycle)
    down
L1/Shared Memory (~10 cycles, 192 KB)
    down
L2 Cache (~100 cycles, 40 MB)
    down
HBM/Global Memory (slowest, ~400 cycles, 40-80 GB)
::


**With tiling**:
1. Load A_block into shared memory (reused BLOCK_N times)
2. Load B_block into shared memory (reused BLOCK_M times)
3. Accumulate into registers (reused K/BLOCK_K times)
4. Write result once to global memory

**Data reuse** = fewer slow memory accesses!

Pointer Arithmetic in 2D

Understanding Strides
~~~~~~~~~~~~~~~~~~~~~

For a row-major matrix A[M, K]:
.. code-block:: python

A[i, j] = *(A_ptr + i_stride*M + j_stride*K)
::


Where:
- ``stride_M`` = number of elements to next row = K
- ``stride_K`` = number of elements to next column = 1

Example for A[4, 3]:
::

Elements in memory: [a00, a01, a02, a10, a11, a12, a20, a21, a22, a30, a31, a32]
A[2, 1] = A_ptr + 2*3 + 1*1 = A_ptr[7] = a21 [OK]
::


Block Pointers
~~~~~~~~~~~~~~

To get pointers to a block A[m:m+BLOCK_M, k:k+BLOCK_K]:

.. code-block:: python

offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
offs_k = tl.arange(0, BLOCK_K)
a_ptrs = a_ptr + offs_am[:, None]*stride_am + offs_k[None, :]*stride_ak
::


**Broadcasting**:
- ``offs_am[:, None]`` has shape (BLOCK_M, 1)
- ``offs_k[None, :]`` has shape (1, BLOCK_K)
- Result ``a_ptrs`` has shape (BLOCK_M, BLOCK_K)

**Example** (BLOCK_M=2, BLOCK_K=2, pid_m=0):
.. code-block:: python

offs_am = [0, 1]        # shape (2,)
offs_k = [0, 1]          # shape (2,)

offs_am[:, None] = [[0],   # shape (2, 1)
                    [1]]

offs_k[None, :] = [[0, 1]]  # shape (1, 2)

Broadcasting multiplication:
offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
= [[0*stride_am + 0*stride_ak, 0*stride_am + 1*stride_ak],
   [1*stride_am + 0*stride_ak, 1*stride_am + 1*stride_ak]]
::


This gives pointers to a 2x2 block!

Advancing Pointers
~~~~~~~~~~~~~~~~~~

To move to next K block:
.. code-block:: python

a_ptrs += BLOCK_K * stride_ak
b_ptrs += BLOCK_K * stride_bk
::


This shifts all pointers in the block by BLOCK_K positions in the K dimension.

L2 Cache Optimization

The Problem with Row-Major Ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we process blocks in simple row-major order:
::

Program 0 -> Block C[0, 0]  (needs A[0, :] and B[:, 0])
Program 1 -> Block C[0, 1]  (needs A[0, :] and B[:, 1])
...
Program 9 -> Block C[1, 0]  (needs A[1, :] and B[:, 0])
::


**Issue**: By the time we compute C[1, 0], B[:, 0] might be evicted from L2 cache!

Grouped Ordering (Swizzling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead, process blocks in groups:
.. code-block:: python

num_pid*in_group = GROUP_SIZE*M * num_pid*n
group_id = pid // num_pid*in_group
first_pid*m = group_id * GROUP_SIZE*M
group_size*m = min(num_pid*m - first_pid*m, GROUP_SIZE*M)
pid_m = first_pid*m + ((pid % num_pid*in_group) % group_size*m)
pid_n = (pid % num_pid*in_group) // group_size*m
::


**Visual Example** (GROUP_SIZE*M=3, 9x9 blocks):

Row-major order:
::

0-> 1-> 2-> 3-> 4-> 5-> 6-> 7-> 8->
9-> 10-> ...
::

Needs to load: 90 unique blocks into SRAM

Grouped order:
::

0down 3down 6down 1down 4down 7down 2down 5down 8down
9down 12down 15down 10down 13down 16down 11down 14down 17down
...
::

Needs to load: 54 unique blocks into SRAM

**Savings**: 40% fewer loads from HBM! This can improve performance by 10-20%.

Auto-Tuning

The Configuration Space
~~~~~~~~~~~~~~~~~~~~~~~

Many parameters affect performance:
- ``BLOCK_SIZE*M``: 32, 64, 128, 256
- ``BLOCK_SIZE*N``: 32, 64, 128, 256
- ``BLOCK_SIZE*K``: 16, 32, 64, 128
- ``GROUP_SIZE*M``: 4, 8, 16
- ``num_warps``: 2, 4, 8, 16
- ``num_stages``: 2, 3, 4, 5

Total combinations: ~1000+

**Problem**: Optimal configuration depends on:
- Matrix sizes (M, N, K)
- GPU architecture (compute capability, SRAM size, etc.)
- Data types (fp16, fp32, int8, etc.)

Triton's Auto-Tuner
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.autotune(
    configs=get_autotune*config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
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
    {'BLOCK_SIZE*M': 128, 'BLOCK_SIZE*N': 256, 'BLOCK_SIZE*K': 64, 'GROUP_SIZE*M': 8},
    num_stages=3,
    num_warps=8
)
::


Good Configurations
~~~~~~~~~~~~~~~~~~~

**For NVIDIA (CUDA)**:
- Large blocks (128x256) for large matrices
- More stages (3-4) to hide latency
- 8 warps for good occupancy

**For AMD (HIP)**:
- Medium blocks (64x64, 128x128)
- Fewer stages (2) due to different architecture
- Different thread group sizes

**FP8 (8-bit floating point)**:
- Even larger blocks (256x256)
- Can fit more data in SRAM
- Tensor Cores process 4x more data per cycle

The Kernel Implementation

Step 1: Compute Program IDs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

pid = tl.program_id(axis=0)
num_pid*m = tl.cdiv(M, BLOCK_SIZE*M)
num_pid*n = tl.cdiv(N, BLOCK_SIZE*N)
::


Map linear program ID to 2D (pid_m, pid_n) using grouped ordering.

Step 2: Initialize Pointers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

offs_am = (pid_m * BLOCK_SIZE*M + tl.arange(0, BLOCK_SIZE*M)) % M
offs_bn = (pid_n * BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE*N)) % N
offs_k = tl.arange(0, BLOCK_SIZE*K)

a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k[None, :]*stride_ak)
b_ptrs = b_ptr + (offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn)
::


Create pointers for first blocks of A and B.

**Note the modulo**:
- ``offs_am % M`` handles M not being multiple of BLOCK_SIZE*M
- Wraps around, so we load valid (though repeated) data
- Doesn't matter because we mask the output store

Step 3: Accumulation Loop
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

accumulator = tl.zeros((BLOCK_SIZE*M, BLOCK_SIZE*N), dtype=tl.float32)

for k in range(0, tl.cdiv(K, BLOCK_SIZE*K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_BLOCK*SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_BLOCK*SIZE_K, other=0.0)

    accumulator = tl.dot(a, b, accumulator)

    a_ptrs += BLOCK_SIZE*K * stride_ak
    b_ptrs += BLOCK_SIZE*K * stride_bk
::


**Key points**:
- Accumulate in ``float32`` for numerical accuracy
- Mask loads when K not multiple of BLOCK_SIZE*K
- ``tl.dot()`` uses hardware accelerators (Tensor Cores on NVIDIA)

Step 4: Apply Activation (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

if ACTIVATION == "leaky_relu":
    accumulator = leaky_relu(accumulator)
c = accumulator.to(tl.float16)
::


**Kernel fusion**: Apply activation while data is in registers (fast)!

Step 5: Store Result
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

offs_cm = pid_m * BLOCK_SIZE*M + tl.arange(0, BLOCK_SIZE*M)
offs_cn = pid_n * BLOCK_SIZE*N + tl.arange(0, BLOCK_SIZE*N)
c_ptrs = c_ptr + stride_cm*offs_cm[:, None] + stride_cn*offs_cn[None, :]
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
::


**Masking** ensures we don't write out of bounds.

Tensor Cores

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

Operate on small matrices (e.g., 16x16):
::

D = A x B + C
::

Where A, B, C, D are 16x16 matrices.

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

Arithmetic Intensity
~~~~~~~~~~~~~~~~~~~~

For matmul C = A x B:
- **FLOPs**: 2MNK (each output element: K multiplies + K adds)
- **Memory**: 2(MK + KN + MN) bytes (read A, B, write C)

Arithmetic Intensity = 2MNK / (2(MK + KN + MN))

For square matrices (M=N=K):
::

AI = 2N^3 / (4N^2) = N/2
::


For N=1024: AI = 512 FLOPs/byte

**This is very high!** Matmul is **compute-bound**, not memory-bound.

Roofline Model
~~~~~~~~~~~~~~

Performance is limited by:
::

min(Peak_Compute, Peak_Memory*BW * AI)
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

Software Pipelining
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

num_stages = 3
::


**Idea**: Overlap memory loads with computation

Without pipelining:
::

Load A, B -> Wait -> Compute -> Load A, B -> Wait -> Compute
::


With 3-stage pipelining:
::

Stage 0: Load A_0, B_0
Stage 1: Load A_1, B_1 | Compute with A_0, B_0
Stage 2: Load A_2, B_2 | Compute with A_1, B_1
Stage 3: Load A_3, B_3 | Compute with A_2, B_2
...
::


**Benefit**: Computation and memory loads happen simultaneously!

**Cost**: Need more registers and SRAM to hold multiple stages.

Loop Unrolling
~~~~~~~~~~~~~~

Triton automatically unrolls the K loop when possible:
.. code-block:: python

for k in range(0, tl.cdiv(K, BLOCK_SIZE*K)):
    accumulator = tl.dot(a, b, accumulator)
::


Becomes (if K/BLOCK_K = 4):
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

**Example** (BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, 8 warps):
- Threads per block: 8 * 32 = 256
- Elements per thread: (128*128) / 256 = 64 for accumulator
- Registers needed: ~100-150 per thread

**Limited supply**: 65536 registers per SM on A100

If too many registers -> fewer blocks per SM -> lower occupancy -> lower performance.

**Balance**: Larger blocks = more reuse but more register pressure.

Common Pitfalls

1. Non-Contiguous Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

assert a.is_contiguous(), "Matrix A must be contiguous"
::


Non-contiguous tensors have unexpected strides -> wrong pointer arithmetic -> incorrect results.

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

offs_am = (...) % M  # Modulo for safety
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)  # Mask stores
::


Forgetting these -> out-of-bounds accesses -> crashes or wrong results.

4. Numerical Precision
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

accumulator = tl.zeros((BLOCK_SIZE*M, BLOCK_SIZE*N), dtype=tl.float32)
::


Using FP16 for accumulation -> loss of precision -> degraded accuracy.

**Best practice**: Accumulate in FP32, cast to FP16 for storage.

Benchmarking Tips

1. Warm-up
~~~~~~~~~~
.. code-block:: python

triton.testing.do_bench(fn)  # Automatically does warm-up
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

x_vals=[128 * i for i in range(2, 33)]  # 256, 384, ..., 4096
::


Performance varies with size:
- Small: Memory-bound, lower TFLOPS
- Medium: Transition zone
- Large: Compute-bound, approaching peak

Key Takeaways

1. **Tiling/Blocking is essential**: Reuse data in fast SRAM
2. **Pointer arithmetic**: Understanding strides is crucial for multi-dimensional arrays
3. **L2 cache matters**: Grouped ordering can give 10-20% speedup
4. **Auto-tuning is powerful**: Optimal configs vary with size and hardware
5. **Tensor Cores are game-changers**: 10-100x speedup for matmul
6. **Matmul is compute-bound**: Unlike vector add (memory-bound)
7. **Accumulate in FP32**: Maintain numerical accuracy
8. **Triton simplifies complex optimizations**: Achieves near-cuBLAS performance with readable code

Matrix multiplication is the foundation of deep learning. Mastering these concepts will help you understand and optimize transformers, CNNs, and other neural architectures!
