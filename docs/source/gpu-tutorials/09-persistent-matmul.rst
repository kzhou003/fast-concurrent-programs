Tutorial 09: Persistent Matmul

Overview
--------

**Persistent kernels** are an advanced GPU programming technique where a fixed number of thread blocks stay resident on the GPU and process multiple tiles of work. This tutorial demonstrates several matmul implementations:

- **Naive matmul** - Standard one-tile-per-CTA approach
- **Persistent matmul** - Fixed CTAs process multiple tiles
- **TMA matmul** - Using Tensor Memory Accelerator (Hopper+)
- **TMA persistent** - Combining both techniques
- **Warp specialization** - Different warps do different work (Blackwell+)

This tutorial also uses **Triton Proton profiler** for detailed performance analysis.

Key Concepts

Persistent Kernel Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Traditional kernel: One CTA per tile
for tile in tiles:
    launch_kernel`1*CTA_per*tile <tile>`_

Persistent kernel: NUM_SMS CTAs process all tiles
launch_kernel`NUM_SMS <all_tiles>`_
for tile in my_assigned*tiles:
    process(tile)
::


**Benefits:**
- Reduced kernel launch overhead
- Better SM utilization
- Amortizes setup costs across multiple tiles
- Enables more sophisticated scheduling

Warp Specialization
~~~~~~~~~~~~~~~~~~~

On Blackwell (compute capability 10.0+), different warps can be assigned different roles:

.. code-block:: python

for ki in tl.range(k_tiles, warp_specialize=True):
    # Hardware scheduler can assign:
    # - Some warps to memory loads (producer)
    # - Other warps to compute (consumer)
    a = a_desc.load([offs_am, offs_k])
    b = b_desc.load([offs_bn, offs_k])
    accumulator = tl.dot(a, b.T, accumulator)
::


**Why it matters:**
- Overlaps memory and compute more effectively
- Improves pipeline utilization
- Hardware-managed producer-consumer pattern

Code Walkthrough

1. Naive Matmul (Baseline)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE*M: tl.constexpr,
                  BLOCK_SIZE*N: tl.constexpr,
                  BLOCK_SIZE*K: tl.constexpr,
                  GROUP_SIZE*M: tl.constexpr):
    pid = tl.program_id(axis=0)

    # Swizzling for better L2 cache hit rate
    num_pid*m = tl.cdiv(M, BLOCK_SIZE*M)
    num_pid*n = tl.cdiv(N, BLOCK_SIZE*N)
    num_pid*in_group = GROUP_SIZE*M * num_pid*n
    group_id = pid // num_pid*in_group
    first_pid*m = group_id * GROUP_SIZE*M
    group_size*m = min(num_pid*m - first_pid*m, GROUP_SIZE*M)
    pid_m = first_pid*m + (pid % group_size*m)
    pid_n = (pid % num_pid*in_group) // group_size*m

    # Standard blocked matmul
    accumulator = tl.zeros((BLOCK_SIZE*M, BLOCK_SIZE*N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE*K)):
        a = tl.load(a_ptrs, mask=...)
        b = tl.load(b_ptrs, mask=...)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE*K * stride_ak
        b_ptrs += BLOCK_SIZE*K * stride_bk

    c = accumulator.to(tl.float16)
    tl.store(c_ptrs, c, mask=c_mask)
::


**Grid launch:**
.. code-block:: python

grid = lambda META: (
    triton.cdiv(M, META["BLOCK_SIZE*M"]) *
    triton.cdiv(N, META["BLOCK_SIZE*N"]),
)
::

- One CTA per output tile
- Simple but has kernel launch overhead

2. Persistent Matmul
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.jit
def matmul_kernel*persistent(a_ptr, b_ptr, c_ptr, ...,
                             NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_pid*m = tl.cdiv(M, BLOCK_SIZE*M)
    num_pid*n = tl.cdiv(N, BLOCK_SIZE*N)
    num_tiles = num_pid*m * num_pid*n

    # Loop over tiles assigned to this CTA
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        # Compute which tile this is
        pid_m, pid_n = *compute_pid(tile_id, ...)

        # Load and compute for this tile
        accumulator = tl.zeros((BLOCK_SIZE*M, BLOCK_SIZE*N), dtype=tl.float32)
        for ki in range(k_tiles):
            a = tl.load(...)
            b = tl.load(...)
            accumulator = tl.dot(a, b, accumulator)

        # Store result
        c = accumulator.to(output_dtype)
        tl.store(c_ptrs, c, mask=c_mask)
::


**Grid launch:**
.. code-block:: python

NUM_SMS = torch.cuda.get_device*properties("cuda").multi_processor*count
grid = lambda META: (
    min(NUM_SMS, num_tiles),  # Don't launch more CTAs than tiles
)
::


**Key differences:**
- Fixed number of CTAs (NUM_SMS)
- Each CTA processes ``ceil(num_tiles / NUM_SMS)`` tiles
- Single kernel launch for entire matmul

3. TMA Matmul
~~~~~~~~~~~~~

For Hopper (compute capability 9.0+):

.. code-block:: python

@triton.jit
def matmul_kernel*tma(a_desc, b_desc, c_desc,
                      M, N, K,
                      BLOCK_SIZE*M: tl.constexpr,
                      BLOCK_SIZE*N: tl.constexpr,
                      BLOCK_SIZE*K: tl.constexpr,
                      FP8_OUTPUT: tl.constexpr,
                      WARP_SPECIALIZE: tl.constexpr):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    pid = tl.program_id(axis=0)
    # ... compute pid_m, pid_n ...

    k_tiles = tl.cdiv(K, BLOCK_SIZE*K)
    accumulator = tl.zeros((BLOCK_SIZE*M, BLOCK_SIZE*N), dtype=tl.float32)

    # Warp specialization for better overlap
    for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * BLOCK_SIZE*K
        # TMA loads
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(dtype)
    c_desc.store([offs_cm, offs_cn], c)
::


**TMA Descriptor creation:**
.. code-block:: python

from triton.tools.tensor_descriptor import TensorDescriptor

a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K])
c_desc = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])
::


4. TMA Persistent with Epilogue Subtiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced optimization for memory-bound kernels:

.. code-block:: python

@triton.jit
def matmul_kernel*tma_persistent(...,
                                 EPILOGUE_SUBTILE: tl.constexpr):
    # ... main computation ...

    if EPILOGUE_SUBTILE:
        # Split BLOCK_M x BLOCK_N into 2 BLOCK_M x (BLOCK_N//2) chunks
        acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        # Store in two parts
        c0 = acc0.to(dtype)
        c_desc.store([offs_cm, offs_cn], c0)

        c1 = acc1.to(dtype)
        c_desc.store([offs_cm, offs_cn + BLOCK_N // 2], c1)
    else:
        accumulator = accumulator.to(dtype)
        c_desc.store([offs_cm, offs_cn], accumulator)
::


**Why epilogue subtiling?**
- Reduces shared memory usage in epilogue
- Frees SRAM for more pipeline stages
- Improves register utilization
- Can increase overall throughput

Proton Profiler Integration

This tutorial demonstrates using Triton's built-in profiler:

.. code-block:: python

import triton.profiler as proton

Start profiling
proton.start("matmul", hook="triton")
proton.deactivate()  # Don't profile initialization

Run benchmarks
for K in range(K_min, K_max, K_step):
    proton.activate(0)
    for * in range(reps):
        matmul(a, b)
    proton.deactivate(0)

Finalize and show results
proton.finalize()
show_profile("fp16", "matmul")
::


**Metrics collected:**
- Time per kernel (ms)
- TFLOPS (teraflops per second)
- Memory bandwidth utilization
- Kernel launch counts

Viewing Profile Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

def show_profile(precision, profile_name):
    import triton.profiler.viewer as proton_viewer

    metric_names = ["time/ms"]
    if precision == 'fp8':
        metric_names = ["tflop8/s"] + metric_names
    elif precision == 'fp16':
        metric_names = ["tflop16/s"] + metric_names

    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)
::


Output shows hierarchical breakdown:
::

matmul
::


Device-side vs Host-side Descriptors

Host-side (TensorDescriptor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from triton.tools.tensor_descriptor import TensorDescriptor

Created on CPU, passed to GPU
a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
matmul_kernel*tma`grid <a_desc, b_desc, c_desc, ...>`_
::


**Pros:** Simpler, works on Hopper
**Cons:** Descriptor creation overhead, limited flexibility

Device-side (tl.make_tensor*descriptor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.jit
def kernel(a_ptr, ...):
    # Created inside kernel
    a_desc = tl.make_tensor*descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    a = a_desc.load([offs_am, offs_k])
::


**Pros:** No host overhead, required for Blackwell warp spec
**Cons:** Only available on newer GPUs

Warp Specialization Details

How It Works
~~~~~~~~~~~~

.. code-block:: python

for ki in tl.range(k_tiles, warp_specialize=True):
    # Hardware may assign:
    # Warps 0-3: Producer (memory loads)
    # Warps 4-7: Consumer (compute)
    a = a_desc.load([offs_am, offs_k])  # Producer
    b = b_desc.load([offs_bn, offs_k])  # Producer
    acc = tl.dot(a, b.T, acc)          # Consumer
::


**Without warp specialization:**
- All warps do: load A -> load B -> compute -> repeat
- Memory and compute are serialized

**With warp specialization:**
- Producer warps continuously load data
- Consumer warps continuously compute
- Better overlap, higher throughput

Requirements
~~~~~~~~~~~~

.. code-block:: python

HAS_WARP*SPECIALIZE = supports_ws() and HAS_TENSOR*DESC

def supports_ws():
    return is_cuda() and torch.cuda.get_device*capability()[0] >= 9

On Hopper: Software pipelining
On Blackwell: Hardware warp specialization
::


Flattening in Persistent Loops

.. code-block:: python

for tile_id in tl.range(start_pid, num_tiles, NUM_SMS,
                       flatten=True,
                       warp_specialize=WARP_SPECIALIZE):
::


**``flatten=True``:**
- Removes loop-carried dependencies
- Allows better scheduling
- Required for software pipelining on Hopper

**``flatten=False``:**
- Keeps loop structure
- Required for Blackwell hardware warp specialization
- Better for async warp scheduling

.. code-block:: python

Choose based on GPU generation
flatten = False if (warp_specialize and is_hopper()) else True
::


Performance Comparison

Expected speedups (relative to naive):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Variant | FP16 Speedup | FP8 Speedup | Notes |
| Persistent | 1.05-1.15x | 1.05-1.1x | Saves launch overhead |
| TMA | 1.15-1.25x | 1.2-1.3x | Better memory access |
| TMA Persistent | 1.2-1.35x | 1.25-1.4x | Combined benefits |
| + Warp Spec (Blackwell) | 1.3-1.5x | 1.4-1.6x | Hardware overlap |

When Each Variant Wins
~~~~~~~~~~~~~~~~~~~~~~

**Naive:**
- Small matrices (< 1024x1024)
- Single matmul
- Minimal kernel launch overhead needed

**Persistent:**
- Medium to large matrices
- Amortizes setup cost
- Many tiles to process

**TMA:**
- Memory-bound workloads
- Hopper+ GPUs
- Complex memory access patterns

**TMA Persistent + Warp Spec:**
- Large matrices (>= 4096x4096)
- Blackwell GPUs
- Maximum performance needed

Precision Support

FP16 (Float16)
~~~~~~~~~~~~~~
.. code-block:: python

a = torch.randn((M, K), device='cuda', dtype=torch.float16)
b = torch.randn((K, N), device='cuda', dtype=torch.float16)
::

- Widely supported
- Good balance of range and precision
- Standard for training

FP8 (Float8)
~~~~~~~~~~~~
.. code-block:: python

dtype = torch.float8_e4m3fn  # E4M3 format
a = torch.randn((M, K), dtype=torch.float16).to(dtype)
b = torch.randn((K, N), dtype=torch.float16).to(dtype)
::

- Requires compute capability >= 9.0
- 2x speedup potential
- Lower precision, faster compute
- Ideal for inference

Auto-tuning Configuration

.. code-block:: python

def matmul_get*configs():
    return [
        triton.Config(
            {'BLOCK_SIZE*M': BM, 'BLOCK_SIZE*N': BN,
             'BLOCK_SIZE*K': BK, 'GROUP_SIZE*M': 8},
            num_stages=s, num_warps=w
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]
::


**Parameters:**
- ``BLOCK_SIZE*M/N/K``: Tile dimensions
- ``num_stages``: Pipeline depth (2-4 typical)
- ``num_warps``: Threads per CTA (4 or 8)
- ``GROUP_SIZE*M``: Swizzling factor for L2 locality

Command-line Usage

.. code-block:: bash

FP16 matmul
python 09-persistent-matmul.py --prec fp16 --K_range 128 1024 --K_step 128

FP8 matmul (requires Hopper+)
python 09-persistent-matmul.py --prec fp8 --K 512

Profile specific K dimension
python 09-persistent-matmul.py --prec fp16 -K 2048
::


**Note:** May fail on GPUs with small shared memory (e.g., RTX 4090). Reduce ``num_stages`` if needed.

Common Pitfalls

1. Wrong B Matrix Layout for TMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

TMA expects B to be transposed
b = b.T.contiguous()  # Make sure it's contiguous!

matmul_tma(a, b, warp_specialize=False)
::


2. Mixing Host and Device Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Bad: Using both in same kernel
a_desc = TensorDescriptor.from_tensor(a, ...)  # Host-side
tl.make_tensor*descriptor(...)  # Device-side

Good: Pick one approach
if HAS_HOST*TENSOR_DESC:
    use_host*descriptors()
else:
    use_device*descriptors()
::


3. Forgetting FP8 Support Check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

if args.prec == 'fp8':
    if not hasattr(torch, 'float8_e4m3fn') or not is_cuda():
        print("This example requires CUDA with fp8 support.")
        exit(1)
::


Summary
-------

**Persistent matmul** demonstrates advanced GPU programming techniques:

- **Persistent kernels** reduce launch overhead and improve SM utilization
- **TMA (Tensor Memory Accelerator)** simplifies memory access on Hopper+
- **Warp specialization** overlaps memory and compute on Blackwell
- **Epilogue subtiling** reduces shared memory pressure
- **Proton profiler** provides detailed performance insights

**Kernel variants:**
1. **Naive**: Baseline, one CTA per tile
2. **Persistent**: Fixed CTAs, multiple tiles each
3. **TMA**: Hardware-accelerated loads (Hopper+)
4. **TMA Persistent**: Combining persistence and TMA
5. **Warp Spec**: Producer-consumer pattern (Blackwell+)

**Performance tips:**
- Use TMA on Hopper and newer GPUs
- Enable warp specialization on Blackwell
- Profile with Proton to identify bottlenecks
- Consider epilogue subtiling for memory-bound kernels
- Auto-tune for your specific hardware and problem sizes

This tutorial shows the evolution of matmul optimizations across GPU generations, from simple tiling to sophisticated hardware-software co-design!
