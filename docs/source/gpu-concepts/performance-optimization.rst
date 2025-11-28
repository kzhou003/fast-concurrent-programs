Performance Optimization Strategies
====================================

A practical guide to making your GPU kernels fast.

The Optimization Process
-------------------------

Step 1: Profile First
~~~~~~~~~~~~~~~~~~~~~

**Never optimize blindly!**

1. Measure baseline performance
2. Identify bottleneck (memory vs compute)
3. Target the bottleneck
4. Measure improvement
5. Repeat

Tools::

    # NVIDIA
    nsys profile --stats=true python script.py
    ncu --set full python script.py

    # AMD
    rocprof --stats python script.py

Step 2: Identify Bottleneck
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory-Bound**: Low arithmetic intensity

* Achieved bandwidth > 60% of peak -> Good!
* Achieved bandwidth < 30% -> Optimization needed
* Focus on: Reducing memory traffic, coalescing

**Compute-Bound**: High arithmetic intensity

* Compute utilization > 80% -> Good!
* Compute utilization < 50% -> Optimization needed
* Focus on: Using Tensor Cores, increasing parallelism

Memory Optimization
-------------------

Strategy 1: Kernel Fusion
~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine multiple operations to reduce memory traffic.

**Before** (3 separate kernels)::

    # Kernel 1: Read X, compute Y = exp(X), write Y
    # Kernel 2: Read Y, compute Z = Y / sum(Y), write Z
    # Kernel 3: Read Z, compute W = Z * scale, write W
    # Total: 3 reads + 3 writes = 6 memory operations

**After** (1 fused kernel)::

    @triton.jit
    def fused_kernel(...):
        x = tl.load(...)      # Read once
        y = tl.exp(x)
        z = y / tl.sum(y)
        w = z * scale
        tl.store(..., w)      # Write once
    # Total: 1 read + 1 write = 2 memory operations
    # 3x reduction in memory traffic!

**Examples in this guide**:

* :doc:`../tutorials/02-fused-softmax`
* :doc:`../tutorials/06-fused-attention`

Strategy 2: Tiling
~~~~~~~~~~~~~~~~~~

Load data into fast SRAM, reuse multiple times.

**Example**: Matrix multiplication::

    # Load A block to SRAM (reused for many B blocks)
    # Load B block to SRAM (reused for many A blocks)
    # Compute in SRAM (fast!)

See :doc:`../tutorials/03-matrix-multiplication` for details.

Strategy 3: Vectorized Loads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load multiple elements per instruction::

    # Bad: Load 1 element at a time (slow)
    for i in range(n):
        x = load(ptr + i)

    # Good: Load BLOCK_SIZE elements at once
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(ptr + offsets)  # Vectorized!

Triton does this automatically for you.

Strategy 4: Memory Coalescing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure adjacent threads access adjacent memory.

**Coalesced** (fast)::

    # Thread i accesses address base + i
    offset = pid * BLOCK_SIZE + thread_id
    data = load(ptr + offset)

**Uncoalesced** (slow)::

    # Random access pattern
    offset = random_indices[thread_id]
    data = load(ptr + offset)

Triton's built-in patterns are usually coalesced.

Compute Optimization
--------------------

Strategy 1: Use Tensor Cores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For matrix operations, Tensor Cores provide 10-100x speedup!

**Automatically used by**::

    accumulator = tl.dot(a, b, accumulator)

**Requirements**:

* FP16, BF16, TF32, FP8, or INT8 inputs
* Block sizes multiple of 16
* Contiguous memory layout

**Performance**:

* Regular cores: ~30 TFLOPS
* With Tensor Cores: ~300 TFLOPS

Strategy 2: Increase Arithmetic Intensity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do more work per byte loaded::

    # Low AI: Load data, do simple operation
    y = x + 1  # AI = 1 FLOP / 8 bytes = 0.125

    # High AI: Load data, do many operations
    y = a * x^3 + b * x^2 + c * x + d
    # AI = 7 FLOPs / 8 bytes = 0.875

Tiling naturally increases AI by reusing data.

Strategy 3: Minimize Thread Divergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use predication**::

    # Bad: Branching
    if x > threshold:
        result = expensive_computation(x)
    else:
        result = 0

    # Good: Predication
    result = tl.where(x > threshold, expensive_computation(x), 0)

**Note**: For expensive computations, short-circuit evaluation helps.

Strategy 4: Optimize Loop Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Unroll loops when possible**::

    # Compiler can unroll for better instruction-level parallelism
    #pragma unroll
    for k in range(0, K, BLOCK_K):
        ...

Triton auto-unrolls when beneficial.

Occupancy Optimization
----------------------

Strategy 1: Reduce Register Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Monitor**::

    ncu --metrics sm__sass_inst_executed_per_thread.avg

**Reduce by**:

* Smaller local arrays
* Recompute instead of store
* Use shared memory for large temporaries

Strategy 2: Tune Shared Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Too much**::

    # Only 2 blocks fit per SM
    @triton.jit
    def kernel(...):
        shared = tl.zeros([1024, 1024])  # Too big!

**Balanced**::

    @triton.jit
    def kernel(...):
        shared = tl.zeros([BLOCK_SIZE, BLOCK_SIZE])
        # BLOCK_SIZE=128 -> 64KB, allows 4 blocks per SM

Strategy 3: Adjust Block Size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Trade-offs**:

* Smaller blocks: Higher occupancy, less data reuse
* Larger blocks: Lower occupancy, more data reuse

**Find sweet spot with auto-tuning!**

Auto-Tuning
-----------

Why Auto-Tune?
~~~~~~~~~~~~~~

Optimal configuration varies with:

* Problem size (M, N, K)
* Hardware (A100 vs H100)
* Data type (FP16 vs FP8)

**Solution**: Try multiple configs, pick the fastest.

Triton Auto-Tuning
~~~~~~~~~~~~~~~~~~

::

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
            # ... more configs
        ],
        key=['M', 'N', 'K'],  # Cache best config for these values
    )
    @triton.jit
    def kernel(...):
        ...

**Process**:

1. First call: Triton benchmarks all configs
2. Caches best config for (M, N, K)
3. Subsequent calls: Uses cached best config

**Config Parameters**:

* ``BLOCK_M, BLOCK_N, BLOCK_K``: Tile sizes
* ``num_stages``: Software pipelining depth (2-5)
* ``num_warps``: Warps per block (2, 4, 8, 16)

Advanced Techniques
-------------------

Warp Specialization
~~~~~~~~~~~~~~~~~~~

Different warps do different tasks::

    # Warp 0-3: Load data from memory
    # Warp 4-7: Compute on data

Benefits:

* Overlap memory and compute
* Better resource utilization

Available on Hopper/Blackwell GPUs.

Persistent Kernels
~~~~~~~~~~~~~~~~~~

Each block processes multiple work items::

    @triton.jit
    def kernel(...):
        for item in range(start, end, stride):
            process(item)

Benefits:

* Amortize launch overhead
* Better L2 cache utilization
* Flexible load balancing

Recomputation
~~~~~~~~~~~~~

Trade compute for memory::

    # Standard: Store intermediate
    intermediate = expensive_compute(x)
    save(intermediate)  # Memory cost
    later_use(intermediate)

    # Optimized: Recompute
    later_use(expensive_compute(x))  # Compute cost, no memory

**When useful**: Memory-bound kernels with spare compute.

**Example**: Flash Attention recomputes attention scores in backward pass.

Common Patterns
---------------

Pattern: Reduction
~~~~~~~~~~~~~~~~~~

Sum/max/min across elements::

    @triton.jit
    def reduce_kernel(...):
        # Load data
        data = tl.load(...)

        # Reduce
        result = tl.sum(data, axis=0)  # Efficient reduction

        # Store
        tl.store(..., result)

**Key**: Triton uses warp shuffles and shared memory efficiently.

Pattern: Element-wise
~~~~~~~~~~~~~~~~~~~~~

Independent operation on each element::

    @triton.jit
    def elementwise_kernel(...):
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        y = f(x)  # Any function
        tl.store(y_ptr + offsets, y)

**Optimization**: Fuse multiple element-wise ops.

Pattern: Matrix Multiply
~~~~~~~~~~~~~~~~~~~~~~~~

Blocked algorithm with tiling::

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)  # Load A block
        b = tl.load(b_ptrs)  # Load B block
        acc += tl.dot(a, b)  # Accumulate

**Key**: Data reuse in SRAM, Tensor Core usage.

Debugging Performance Issues
-----------------------------

Issue: Low Bandwidth
~~~~~~~~~~~~~~~~~~~~

**Symptoms**:

* Achieved bandwidth << peak bandwidth
* Memory-bound operation

**Check**:

1. Coalescing: Use ``ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum``
2. L2 hit rate: Might be hitting cache
3. Occupancy: Too low?

Issue: Low Compute Utilization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**:

* Low % of peak FLOPs
* Compute-bound operation

**Check**:

1. Tensor Core usage: Are they being used?
2. Thread divergence: Branching killing performance?
3. Occupancy: Need more warps?

Issue: Lower Than PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Possible causes**:

1. Not using Tensor Cores (PyTorch does)
2. Sub-optimal config (need auto-tuning)
3. Missing optimizations (fusion, tiling)
4. Data layout issues (non-contiguous)

**Debug**:

* Profile both with ``ncu``
* Compare metrics (bandwidth, compute, occupancy)
* Check if PyTorch uses vendor lib (cuBLAS, cuDNN)

Performance Checklist
---------------------

Before You Optimize
~~~~~~~~~~~~~~~~~~~

[ ] Profile to identify bottleneck
[ ] Measure baseline performance
[ ] Set performance target (realistic!)

Memory Optimizations
~~~~~~~~~~~~~~~~~~~~

[ ] Fuse operations to reduce memory traffic
[ ] Use tiling to maximize SRAM reuse
[ ] Ensure coalesced memory accesses
[ ] Minimize global memory accesses

Compute Optimizations
~~~~~~~~~~~~~~~~~~~~~

[ ] Use Tensor Cores for matmul
[ ] Minimize thread divergence
[ ] Maximize arithmetic intensity
[ ] Use appropriate data types (FP16, BF16)

Occupancy Optimization
~~~~~~~~~~~~~~~~~~~~~~

[ ] Check register usage
[ ] Check shared memory usage
[ ] Tune block size
[ ] Measure actual occupancy

Advanced
~~~~~~~~

[ ] Auto-tune configurations
[ ] Use software pipelining
[ ] Consider persistent kernels
[ ] Profile with vendor tools

Summary
-------

**Optimization hierarchy**:

1. **Algorithm**: Choose efficient algorithm
2. **Memory**: Reduce traffic, maximize reuse
3. **Compute**: Use specialized hardware (Tensor Cores)
4. **Parallelism**: Balance resources for good occupancy
5. **Tuning**: Auto-tune for specific hardware and problem sizes

**Remember**: Profile -> Optimize -> Measure -> Repeat

Next Steps
----------

Apply these concepts in practice:

* :doc:`../tutorials/02-fused-softmax` - Memory optimization through fusion
* :doc:`../tutorials/03-matrix-multiplication` - Compute optimization with Tensor Cores
* :doc:`../tutorials/06-fused-attention` - Advanced optimization combining all techniques
