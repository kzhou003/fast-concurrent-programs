GPU Fundamentals

Understanding GPU Architecture

Modern GPUs are massively parallel processors designed to handle thousands of concurrent operations.
Unlike CPUs that optimize for sequential processing with a few powerful cores, GPUs have hundreds
or thousands of smaller cores optimized for parallel workloads.

Key Differences: CPU vs GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CPU Architecture**

* 4-64 powerful cores
* Large caches (MB per core)
* Optimized for latency (single-thread performance)
* Complex control logic and branch prediction
* Best for: Sequential tasks, complex logic

**GPU Architecture**

* 1000s of simple cores
* Small caches (KB per core)
* Optimized for throughput (many threads)
* Simple control logic
* Best for: Parallel tasks, regular computation patterns

GPU Hierarchy

Understanding the GPU hierarchy is essential for writing efficient kernels.

Streaming Multiprocessors (SMs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Modern GPUs have 80-140 SMs (also called Compute Units on AMD)
* Each SM contains:

  * 64-128 CUDA cores (FP32/INT32 units)
  * 32-64 FP64 units
  * 4-8 Tensor Cores (specialized matrix multiply units)
  * 128-256 KB of shared memory (SRAM)
  * 64K 32-bit registers
  * L1 cache

* SMs execute independently and in parallel

Thread Organization
~~~~~~~~~~~~~~~~~~~

**Threads**

* Smallest unit of execution
* Each thread has its own registers
* Threads in a warp execute in lockstep (SIMD)

**Warps (NVIDIA) / Wavefronts (AMD)**

* Group of 32 threads (NVIDIA) or 64 threads (AMD)
* Execute instructions in lockstep
* Hardware scheduling unit
* All threads in warp execute same instruction simultaneously

**Thread Blocks**

* Group of up to 1024 threads
* Threads in a block can:

  * Share data via shared memory
  * Synchronize with barriers
  * Cooperate on a task

* Scheduled to single SM

**Grid**

* Collection of thread blocks
* Defines total parallel work
* Blocks execute independently (can be scheduled to different SMs)

SPMD Programming Model

GPU programming uses the **Single Program, Multiple Data (SPMD)** model:

1. You write code for **one thread/block**
2. GPU launches **thousands of copies** in parallel
3. Each copy processes **different data**

Example Visualization
~~~~~~~~~~~~~~~~~~~~~

For a vector of 1024 elements with block size 256::

    Grid: 4 blocks

In Triton::

    @triton.jit
    def kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # This code runs for EACH block in parallel
        pid = tl.program_id(0)  # Which block am I?
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        # Process my chunk of data...

Compute Capabilities

NVIDIA GPUs are categorized by compute capability (version number):

* **7.x**: Volta (V100) - First generation Tensor Cores
* **8.x**: Ampere (A100, RTX 30 series) - 2nd gen Tensor Cores, BF16 support
* **9.x**: Hopper (H100) - 4th gen Tensor Cores, FP8 support, Tensor Memory Accelerator
* **10.x**: Blackwell (B100) - 5th gen Tensor Cores, enhanced FP8

Higher compute capability = more features and better performance.

Key Concepts Summary

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Concept
     - Description
   * - SM
     - Independent processor with cores, memory, and schedulers
   * - Warp
     - 32 threads executing in lockstep (SIMD)
   * - Thread Block
     - Group of threads that can cooperate (up to 1024 threads)
   * - Grid
     - Complete collection of thread blocks for a kernel
   * - SPMD
     - Single Program, Multiple Data - one code, many instances
   * - Occupancy
     - Ratio of active warps to maximum warps per SM

Next Steps
----------

Now that you understand GPU fundamentals, learn about:

* :doc:`memory-hierarchy` - Understanding fast vs slow memory
* :doc:`execution-model` - How GPUs schedule and execute work
* :doc:`performance-optimization` - Making your kernels fast

Then start with the tutorials:

* :doc:`../tutorials/01-vector-add` - Your first GPU kernel
