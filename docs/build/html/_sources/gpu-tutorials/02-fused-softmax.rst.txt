Fused Softmax in Triton

Overview
--------
This tutorial demonstrates **kernel fusion**, a critical GPU optimization technique. By fusing the softmax operation into a single kernel, we can achieve ~4x speedup over naive implementations by reducing memory traffic.

What You'll Learn
- The concept and benefits of **kernel fusion**
- How to perform **reduction operations** on GPU
- Why **bandwidth-bound operations** benefit from fusion
- The difference between **SRAM** and **DRAM**
- **Numerical stability** techniques for softmax

The Problem with Naive Softmax

Standard Softmax Formula
~~~~~~~~~~~~~~~~~~~~~~~~
::

softmax(x_i) = exp(x_i) / SIGMA exp(x_j)
::


For numerical stability, we subtract the max:
::

softmax(x_i) = exp(x_i - max(x)) / SIGMA exp(x_i - max(x))
::


Naive PyTorch Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

x_max = x.max(dim=1)[0]           # Read MN, write M
z = x - x_max[:, None]             # Read MN+M, write MN
numerator = torch.exp(z)           # Read MN, write MN
denominator = numerator.sum(dim=1) # Read MN, write M
ret = numerator / denominator      # Read MN+M, write MN
::


**Total Memory Traffic**:
- **Reads**: 5MN + 2M elements
- **Writes**: 3MN + 2M elements
- **Total**: 8MN + 4M elements

For a matrix of size 4096x1024 (float32):
- Total data movement: ~134 MB
- But the actual result is only: ~16 MB

**Problem**: We're moving 8x more data than necessary!

GPU Memory Hierarchy

DRAM (Global Memory)
~~~~~~~~~~~~~~~~~~~~
- **Size**: 8-80 GB
- **Bandwidth**: 500-2000 GB/s
- **Latency**: 200-400 cycles
- **Location**: Off-chip (separate memory chips)

SRAM (Shared Memory / L1 Cache)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Size**: 48-256 KB per SM
- **Bandwidth**: 10-20 TB/s (10-20x faster!)
- **Latency**: 1-10 cycles
- **Location**: On-chip (on the GPU die)

The Key Insight
~~~~~~~~~~~~~~~
If a row fits in SRAM, we can:
1. Load the row once from DRAM -> SRAM
2. Do all computations in SRAM
3. Write result once back to DRAM

**Memory Traffic**: Read MN, write MN (2MN total) vs. 8MN+4M!

This is what **kernel fusion** achieves.

How the Fused Kernel Works

Block-Level Processing
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

row_idx = tl.program_id(0)
::


Each program processes one or more complete rows:
- **Why rows?** Softmax is row-wise operation (normalize each row independently)
- **One row per program**: Simple, good for small row sizes
- **Multiple rows per program**: Better for tiny rows (reduces launch overhead)

Step 1: Load Row Into SRAM
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

col_offsets = tl.arange(0, BLOCK_SIZE)
input_ptrs = row_start*ptr + col_offsets
mask = col_offsets < n_cols
row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
::


**Important Details**:
- ``BLOCK_SIZE`` must be power of 2 (GPU requirement)
- If ``n_cols < BLOCK_SIZE``, we pad with ``-inf``
- ``-inf`` is safe: ``exp(-inf) = 0``, doesn't affect sum

Step 2: Compute Max (Reduction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

row_minus*max = row - tl.max(row, axis=0)
::


**Reduction Operation**: Combine many values into one
- Triton does this efficiently using **warp shuffles** and **shared memory**
- In CUDA, you'd manually write a tree reduction
- All happens in SRAM (very fast!)

Step 3: Exponentiation
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

numerator = tl.exp(row_minus*max)
::


**GPU Math Functions**:
- Uses hardware-accelerated exp (CUDA ``__expf``)
- Fast but approximate (~1 ULP error)
- Good enough for neural networks!

Step 4: Compute Sum (Another Reduction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

denominator = tl.sum(numerator, axis=0)
::


Again, efficient reduction in SRAM.

Step 5: Normalize and Write Back
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

softmax_output = numerator / denominator
tl.store(output_ptrs, softmax_output, mask=mask)
::


Only one write to DRAM for the entire row!

Power-of-Two Block Sizes

Why Must BLOCK_SIZE Be Power of 2?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU hardware is optimized for power-of-2 sizes:

1. **Memory Alignment**:
   - Memory accesses are most efficient at 128-byte boundaries
   - Powers of 2 naturally align to these boundaries

2. **Warp Operations**:
   - NVIDIA warps are 32 threads
   - Efficient operations require BLOCK_SIZE % 32 == 0
   - Powers of 2 guarantee this

3. **Reduction Trees**:
   - Reduction algorithms are most efficient with power-of-2 sizes
   - Enables balanced tree structure

.. code-block:: python

BLOCK_SIZE = triton.next_power*of_2(n_cols)
::


For example:
- n_cols = 1000 -> BLOCK_SIZE = 1024
- n_cols = 513 -> BLOCK_SIZE = 1024
- n_cols = 512 -> BLOCK_SIZE = 512

Occupancy and Performance Tuning

Number of Warps
~~~~~~~~~~~~~~~
.. code-block:: python

num_warps = 8
::


**Warps** (NVIDIA) or **Wavefronts** (AMD):
- Group of 32 threads (NVIDIA) or 64 threads (AMD)
- Execute instructions in lockstep (SIMD)
- GPU schedules at warp granularity

**Why 8 warps?**
- Total threads = 8 x 32 = 256 threads per block
- Enough to hide memory latency
- Not so many that we run out of registers

Number of Pipeline Stages
~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

num_stages = 4 if SIZE_SMEM > 200000 else 2
::


**Software Pipelining**: Overlap memory loads with computation
- Stage 1: Load block A while computing nothing
- Stage 2: Load block B while computing on block A
- Stage 3: Load block C while computing on block B
- ...

**More stages** = better hiding of memory latency, but requires more SRAM

Computing Occupancy
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
occupancy = min(occupancy, SIZE_SMEM // size_smem)
num_programs = NUM_SM * occupancy
::


**Occupancy**: How many thread blocks can run simultaneously on an SM

**Limiting Factors**:
1. **Registers**: Each thread needs registers; limited supply per SM
2. **Shared Memory**: Each block uses SRAM; limited per SM
3. **Hardware limits**: Max threads/blocks per SM

**Example Calculation** (A100 GPU):
- NUM_SM = 108 (streaming multiprocessors)
- NUM_REGS = 65536 registers per SM
- SIZE_SMEM = 164 KB per SM
- num_warps = 8, WARP_SIZE = 32
- Suppose kernel uses 64 registers/thread

Register occupancy:
::

occupancy = 65536 / (64 * 32 * 8) = 65536 / 16384 ~ 4 blocks per SM
::


Total concurrent blocks:
::

num_programs = 108 * 4 = 432 blocks
::


If you have 4096 rows, many waves needed:
::

waves = ceil(4096 / 432) ~ 10 waves
::


Persistent Kernels

The Pattern
~~~~~~~~~~~
.. code-block:: python

for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
    # process row_idx
::


Instead of launching one kernel per row:
- Launch ``num_programs`` kernels total
- Each kernel processes multiple rows in a loop
- Called **persistent kernels** or **persistent threads**

**Benefits**:
1. Amortize kernel launch overhead
2. Better L2 cache utilization
3. More flexible load balancing

**When to use**:
- Many small tasks (like many small rows)
- When kernel launch overhead is significant

Numerical Stability

Why Subtract Max?
~~~~~~~~~~~~~~~~~
.. code-block:: python

row_minus*max = row - tl.max(row, axis=0)
::


Without this, for large values:
.. code-block:: python

exp(1000) = overflow! (inf in float32)
softmax([1000, 1001, 1002]) -> [nan, nan, nan]
::


With max subtraction:
.. code-block:: python

x = [1000, 1001, 1002]
max_x = 1002
x - max_x = [-2, -1, 0]
exp([-2, -1, 0]) = [0.135, 0.368, 1.0]  # no overflow!
softmax ~ [0.09, 0.24, 0.67]  # correct!
::


**Mathematical correctness**:
::

softmax(x + c) = softmax(x) for any constant c
::


So subtracting max doesn't change the result, just prevents overflow.

Padding with -inf
~~~~~~~~~~~~~~~~~
.. code-block:: python

row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
::


For out-of-bounds elements:
- Setting to -inf means ``exp(-inf) = 0``
- These don't contribute to the sum
- Result is mathematically correct for the actual row length

Memory Bandwidth Analysis

Theoretical Speedup
~~~~~~~~~~~~~~~~~~~
Naive: 8MN + 4M bytes
Fused: 2MN bytes

For large N: speedup ~ (8MN) / (2MN) = **4x**

Actual Performance
~~~~~~~~~~~~~~~~~~
From benchmark results, you might see:
- Naive softmax: ~50 GB/s
- Torch softmax: ~150 GB/s (has some fusion)
- Triton fused: ~200 GB/s

**Why not 4x over naive?**
- Reductions have some overhead
- Computation (exp) takes time too
- Memory bandwidth isn't the only factor

**Why faster than Torch?**
- PyTorch softmax handles general ND tensors
- Triton kernel specialized for row-wise 2D case
- Less overhead, better optimization for this specific case

Key CUDA/Triton Concepts

Reduction Operations
~~~~~~~~~~~~~~~~~~~~
**Challenge**: Combine N values into 1 across many threads

**Triton approach**:
.. code-block:: python

tl.max(row, axis=0)  # Finds max across row
tl.sum(row, axis=0)  # Sums across row
::


**CUDA approach** (what Triton generates):
1. Each thread computes partial result
2. Use warp shuffle instructions for intra-warp reduction
3. Use shared memory for inter-warp reduction
4. Final result in thread 0

Warp Shuffles
~~~~~~~~~~~~~
Hardware instructions to share data between threads in a warp:
- ``**shfl_down*sync()``: Shuffle data from higher-indexed thread
- Much faster than shared memory
- Triton uses these automatically

Block vs Thread
~~~~~~~~~~~~~~~
- **Triton program** = **CUDA thread block**
- Triton abstracts away individual threads
- You think in terms of vectors (BLOCK_SIZE elements)
- Compiler generates thread code automatically

Common Pitfalls

1. Row Too Large for SRAM
~~~~~~~~~~~~~~~~~~~~~~~~~
If ``BLOCK_SIZE * sizeof(float32) > SRAM_SIZE``:
- Kernel won't fit in SRAM
- Will spill to DRAM (slow!)
- Solution: Process row in chunks (not shown in this tutorial)

2. Numerical Precision
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

numerator = tl.exp(row_minus*max)
::

- Using ``float16`` can cause underflow
- Always use ``float32`` for intermediate computations
- Cast to ``float16`` only for final output if needed

3. Masking Errors
~~~~~~~~~~~~~~~~~
.. code-block:: python

mask = col_offsets < n_cols
::

- Forgetting mask -> out-of-bounds access -> crash
- Wrong mask value -> incorrect results
- ``-inf`` is usually the right padding value for softmax

Performance Tips

1. **Ensure rows fit in SRAM**: Check ``BLOCK_SIZE * 4 < SRAM_SIZE``
2. **Use float32 for accumulation**: Better numerical accuracy
3. **Tune num_warps**: Try 4, 8, or 16 depending on row size
4. **Adjust num_stages**: More stages hide latency but need more SRAM
5. **Profile occupancy**: Use ``nsys`` or ``ncu`` to see if you're register/SRAM limited

Comparison to Other Approaches

JIT Fusion (torch.jit.script)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Attempts to fuse operations automatically
- Often doesn't work for complex patterns
- Limited control over optimization

Manual CUDA
~~~~~~~~~~~
- Full control, best possible performance
- Very complex to write and maintain
- Triton achieves 95%+ of CUDA performance with 1/3 the code

CuDNN/CuBLAS
~~~~~~~~~~~~
- Vendor libraries, highly optimized
- Not customizable, no fusion with custom ops
- Triton lets you fuse softmax with other operations

Extensions
----------

Ideas to explore:
1. **Fused operations**: Softmax + scaling, softmax + masking
2. **Backward pass**: Implement grad_softmax
3. **Dimension flexibility**: Handle column-wise softmax
4. **Multi-row processing**: Process multiple rows per program for tiny rows
5. **Flash Attention**: Use this softmax as building block

Key Takeaways

1. **Kernel fusion reduces memory traffic**: Fewer DRAM accesses = faster
2. **SRAM is 10-20x faster than DRAM**: Keep data on-chip when possible
3. **Reductions are fundamental**: Many operations need them (sum, max, etc.)
4. **Power-of-2 sizes**: Required for efficient GPU operations
5. **Numerical stability matters**: Always subtract max in softmax
6. **Triton simplifies GPU programming**: Achieves near-CUDA performance with much simpler code

This pattern of fusion applies to many operations: layer norm, RMSNorm, GELU, and more!
