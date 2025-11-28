GPU Execution Model

Understanding how GPUs schedule and execute work is essential for writing efficient kernels.

Thread Execution

Warps and SIMD Execution
~~~~~~~~~~~~~~~~~~~~~~~~~

GPUs execute threads in groups called **warps** (NVIDIA) or **wavefronts** (AMD):

* **Warp size**: 32 threads (NVIDIA), 64 threads (AMD)
* All threads in a warp execute the **same instruction** simultaneously
* This is **SIMD**: Single Instruction, Multiple Data

Example::

    # All 32 threads in warp execute simultaneously:
    x = tl.load(ptr + offsets)  # 32 loads in parallel
    y = x * 2                    # 32 multiplies in parallel
    tl.store(ptr + offsets, y)   # 32 stores in parallel

Thread Divergence
~~~~~~~~~~~~~~~~~

**Problem**: What if threads in a warp take different paths?

.. code-block:: python

    if condition:
        path_a()  # Some threads take this
    else:
        path_b()  # Other threads take this

**Answer**: Both paths execute! (if any thread needs each path)

* Threads that don't need a path are masked out
* Wastes compute cycles

**Solution**: Use predication instead of branching::

    # Bad (branching)
    if x > 0:
        result = x * 2
    else:
        result = 0

    # Good (predication)
    result = tl.where(x > 0, x * 2, 0)  # No branching!

Occupancy
---------

What is Occupancy?
~~~~~~~~~~~~~~~~~~

**Occupancy** = (Active warps per SM) / (Maximum warps per SM)

* Higher occupancy = more threads = better latency hiding
* But not always! Quality > quantity

Factors Limiting Occupancy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Registers per thread**

   * Each SM has ~65K registers
   * More registers per thread = fewer threads per SM

2. **Shared memory per block**

   * Each SM has 128-256 KB shared memory
   * More shared memory per block = fewer blocks per SM

3. **Thread block size**

   * Maximum 1024 threads per block
   * Too small = poor utilization
   * Too large = resource constraints

4. **Hardware limits**

   * Max 16-32 blocks per SM
   * Max 1024-2048 threads per SM

Example Calculation
~~~~~~~~~~~~~~~~~~~~

For A100 GPU:

* Max 64 warps per SM (2048 threads)
* 65536 registers per SM

If your kernel uses 128 registers per thread::

    Max threads = 65536 / 128 = 512 threads per SM
    Max warps = 512 / 32 = 16 warps per SM
    Occupancy = 16 / 64 = 25%

**Ouch!** Register pressure killed occupancy.

Solutions:

* Reduce local variables
* Use smaller BLOCK_SIZE
* Let compiler optimize

Latency Hiding

Why Occupancy Matters
~~~~~~~~~~~~~~~~~~~~~

Memory access takes ~400 cycles. If only one warp is active::

    Warp 0: Load data ... (wait 400 cycles) ... continue
    -> GPU idle for 400 cycles!

With 16 active warps::

    Warp 0:  Load ... (switch to warp 1)
    Warp 1:  Load ... (switch to warp 2)
    ...
    Warp 15: Load ... (switch to warp 0)
    Warp 0:  Data ready! Continue...
    -> No idle time!

**Latency Hiding**: While one warp waits, others compute.

The Occupancy Sweet Spot
~~~~~~~~~~~~~~~~~~~~~~~~~

**Not always better!**

* 25% occupancy: Often sufficient for memory-bound ops
* 50% occupancy: Good balance
* 100% occupancy: Not always achievable or necessary

For compute-bound operations, higher occupancy helps more.

Kernel Launch Configuration

Grid and Block Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~

In Triton::

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),  # Grid dimension 0
        triton.cdiv(N, meta['BLOCK_SIZE_N']),  # Grid dimension 1
    )
    kernel[grid](...)

Each grid dimension creates more parallel work.

Choosing Block Size
~~~~~~~~~~~~~~~~~~~

**Trade-offs**:

.. list-table::
   :header-rows: 1

   * - Aspect
     - Small Blocks (64-128)
     - Large Blocks (256-1024)
   * - Occupancy
     - Higher (more blocks fit)
     - Lower (fewer blocks fit)
   * - Data reuse
     - Less
     - More
   * - Launch overhead
     - More
     - Less
   * - Flexibility
     - Better load balancing
     - Can waste resources

**Rules of thumb**:

* Multiples of warp size (32 for NVIDIA)
* Powers of 2 usually work well
* 128-256 is often a good starting point
* Use auto-tuning to find optimum

Synchronization

Within a Block
~~~~~~~~~~~~~~

Threads in a block can synchronize::

    # CUDA
    __syncthreads();

    # Triton (automatic at certain operations)
    # Barrier inserted automatically when needed

Use cases:

* All threads load data -> sync -> all threads use data
* Producer-consumer patterns within block

Between Blocks
~~~~~~~~~~~~~~

**Cannot synchronize between blocks directly!**

Blocks must be independent. Why?

* Blocks can execute in any order
* Some blocks might not start until others finish
* Different SMs, no shared synchronization

If you need cross-block coordination:

* Launch multiple kernels
* Use atomic operations (careful!)
* Redesign algorithm

Warp-Level Operations

Warp Shuffles
~~~~~~~~~~~~~

Share data between threads in a warp **without memory**::

    # Each thread has a value
    # Thread 0 wants value from thread 5
    value = warp_shuffle(my_value, source_lane=5)

Extremely fast (1 cycle) vs shared memory (~10 cycles).

Triton uses these automatically in reductions!

Warp-Level Reductions
~~~~~~~~~~~~~~~~~~~~~

::

    # Sum across all threads in warp
    warp_sum = tl.sum(value, axis=0)  # Uses warp shuffles

    # Max across warp
    warp_max = tl.max(value, axis=0)

Much faster than using shared memory for small reductions.

Software Pipelining

Overlap Compute and Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Without pipelining**::

    for i in range(N):
        data = load_from_memory()  # Wait...
        result = compute(data)     # Compute
        store(result)              # Store

    # Timeline: Load | Compute | Store | Load | Compute | Store | ...
    #            ^^^^^^ Idle compute while loading!

**With pipelining** (num_stages=3)::

    # Stage 0: Load data[0]
    # Stage 1: Load data[1], Compute on data[0]
    # Stage 2: Load data[2], Compute on data[1], Store result[0]
    # Stage 3: Load data[3], Compute on data[2], Store result[1]
    # ...

    # Timeline: Multiple operations overlap!
    # Compute runs while memory loads happen

In Triton::

    @triton.jit
    def kernel(..., num_stages=3):  # Enable software pipelining
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)  # Automatically pipelined!
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)

Cost: Need more registers/SRAM to hold multiple stages.

Persistent Kernels

Traditional Approach
~~~~~~~~~~~~~~~~~~~~

Launch one block per task::

    # 1000 tasks -> launch 1000 blocks
    kernel[1000](...)

**Problem**: Launch overhead, poor load balancing for variable-size tasks.

Persistent Approach
~~~~~~~~~~~~~~~~~~~

Launch fewer blocks, each processes multiple tasks::

    @triton.jit
    def persistent_kernel(...):
        for task_id in tl.range(start, end, step):
            # Process task_id
            ...

    # Only 100 blocks, each handles 10 tasks
    kernel[100](...)

**Benefits**:

* Amortize launch overhead
* Better load balancing
* Better L2 cache utilization

See :doc:`../tutorials/02-fused-softmax` for example.

Performance Considerations

Key Factors
~~~~~~~~~~~

1. **Occupancy**: Enough warps to hide latency?
2. **Memory coalescing**: Adjacent threads -> adjacent addresses?
3. **Thread divergence**: Minimize branching
4. **Register pressure**: Don't use too many per thread
5. **Shared memory usage**: Maximize reuse, don't overflow
6. **Instruction mix**: Balance memory and compute

Profiling Tools
~~~~~~~~~~~~~~~

**NVIDIA**::

    # Occupancy
    ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active

    # Memory throughput
    ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed

    # L1/L2 hit rates
    ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.pct_of_peak_sustained_elapsed

**AMD**::

    rocprof --stats python your_script.py

Summary
-------

Key concepts:

* **Warps**: 32 threads execute in lockstep (SIMD)
* **Occupancy**: More active warps = better latency hiding
* **Thread divergence**: Avoid branches, use predication
* **Synchronization**: Within block (yes), between blocks (no)
* **Pipelining**: Overlap memory and compute
* **Persistent kernels**: Amortize overhead, better load balancing

Next: Learn :doc:`performance-optimization` strategies.
