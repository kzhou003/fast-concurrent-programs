Tutorial 09: Persistent Matmul
==============================

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
------------

Persistent Kernel Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Traditional kernel: One CTA per tile
====================================
for tile in tiles:
    launch*kernel`1*CTA*per*tile <tile>`_

Persistent kernel: NUM*SMS CTAs process all tiles
=================================================
launch*kernel`NUM*SMS <all*tiles>`_
for tile in my*assigned*tiles:
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

for ki in tl.range(k*tiles, warp*specialize=True):
    # Hardware scheduler can assign:
    # - Some warps to memory loads (producer)
    # - Other warps to compute (consumer)
    a = a*desc.load([offs*am, offs*k])
    b = b*desc.load([offs*bn, offs*k])
    accumulator = tl.dot(a, b.T, accumulator)
::


**Why it matters:**
- Overlaps memory and compute more effectively
- Improves pipeline utilization
- Hardware-managed producer-consumer pattern

Code Walkthrough
----------------

1. Naive Matmul (Baseline)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.jit
def matmul*kernel(a*ptr, b*ptr, c*ptr,
                  M, N, K,
                  stride*am, stride*ak,
                  stride*bk, stride*bn,
                  stride*cm, stride*cn,
                  BLOCK*SIZE*M: tl.constexpr,
                  BLOCK*SIZE*N: tl.constexpr,
                  BLOCK*SIZE*K: tl.constexpr,
                  GROUP*SIZE*M: tl.constexpr):
    pid = tl.program*id(axis=0)

    # Swizzling for better L2 cache hit rate
    num*pid*m = tl.cdiv(M, BLOCK*SIZE*M)
    num*pid*n = tl.cdiv(N, BLOCK*SIZE*N)
    num*pid*in*group = GROUP*SIZE*M * num*pid*n
    group*id = pid // num*pid*in*group
    first*pid*m = group*id * GROUP*SIZE*M
    group*size*m = min(num*pid*m - first*pid*m, GROUP*SIZE*M)
    pid*m = first*pid*m + (pid % group*size*m)
    pid*n = (pid % num*pid*in*group) // group*size*m

    # Standard blocked matmul
    accumulator = tl.zeros((BLOCK*SIZE*M, BLOCK*SIZE*N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK*SIZE*K)):
        a = tl.load(a*ptrs, mask=...)
        b = tl.load(b*ptrs, mask=...)
        accumulator = tl.dot(a, b, accumulator)
        a*ptrs += BLOCK*SIZE*K * stride*ak
        b*ptrs += BLOCK*SIZE*K * stride*bk

    c = accumulator.to(tl.float16)
    tl.store(c*ptrs, c, mask=c*mask)
::


**Grid launch:**
.. code-block:: python

grid = lambda META: (
    triton.cdiv(M, META["BLOCK*SIZE*M"]) *
    triton.cdiv(N, META["BLOCK*SIZE*N"]),
)
::

- One CTA per output tile
- Simple but has kernel launch overhead

2. Persistent Matmul
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.jit
def matmul*kernel*persistent(a*ptr, b*ptr, c*ptr, ...,
                             NUM*SMS: tl.constexpr):
    start*pid = tl.program*id(axis=0)
    num*pid*m = tl.cdiv(M, BLOCK*SIZE*M)
    num*pid*n = tl.cdiv(N, BLOCK*SIZE*N)
    num*tiles = num*pid*m * num*pid*n

    # Loop over tiles assigned to this CTA
    for tile*id in tl.range(start*pid, num*tiles, NUM*SMS, flatten=True):
        # Compute which tile this is
        pid*m, pid*n = *compute*pid(tile*id, ...)

        # Load and compute for this tile
        accumulator = tl.zeros((BLOCK*SIZE*M, BLOCK*SIZE*N), dtype=tl.float32)
        for ki in range(k*tiles):
            a = tl.load(...)
            b = tl.load(...)
            accumulator = tl.dot(a, b, accumulator)

        # Store result
        c = accumulator.to(output*dtype)
        tl.store(c*ptrs, c, mask=c*mask)
::


**Grid launch:**
.. code-block:: python

NUM*SMS = torch.cuda.get*device*properties("cuda").multi*processor*count
grid = lambda META: (
    min(NUM*SMS, num*tiles),  # Don't launch more CTAs than tiles
)
::


**Key differences:**
- Fixed number of CTAs (NUM*SMS)
- Each CTA processes ``ceil(num*tiles / NUM*SMS)`` tiles
- Single kernel launch for entire matmul

3. TMA Matmul
~~~~~~~~~~~~~

For Hopper (compute capability 9.0+):

.. code-block:: python

@triton.jit
def matmul*kernel*tma(a*desc, b*desc, c*desc,
                      M, N, K,
                      BLOCK*SIZE*M: tl.constexpr,
                      BLOCK*SIZE*N: tl.constexpr,
                      BLOCK*SIZE*K: tl.constexpr,
                      FP8*OUTPUT: tl.constexpr,
                      WARP*SPECIALIZE: tl.constexpr):
    dtype = tl.float8e4nv if FP8*OUTPUT else tl.float16
    pid = tl.program*id(axis=0)
    # ... compute pid*m, pid*n ...

    k*tiles = tl.cdiv(K, BLOCK*SIZE*K)
    accumulator = tl.zeros((BLOCK*SIZE*M, BLOCK*SIZE*N), dtype=tl.float32)

    # Warp specialization for better overlap
    for k in tl.range(k*tiles, warp*specialize=WARP*SPECIALIZE):
        offs*k = k * BLOCK*SIZE*K
        # TMA loads
        a = a*desc.load([offs*am, offs*k])
        b = b*desc.load([offs*bn, offs*k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(dtype)
    c*desc.store([offs*cm, offs*cn], c)
::


**TMA Descriptor creation:**
.. code-block:: python

from triton.tools.tensor*descriptor import TensorDescriptor

a*desc = TensorDescriptor.from*tensor(a, [BLOCK*M, BLOCK*K])
b*desc = TensorDescriptor.from*tensor(b, [BLOCK*N, BLOCK*K])
c*desc = TensorDescriptor.from*tensor(c, [BLOCK*M, BLOCK*N])
::


4. TMA Persistent with Epilogue Subtiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced optimization for memory-bound kernels:

.. code-block:: python

@triton.jit
def matmul*kernel*tma*persistent(...,
                                 EPILOGUE*SUBTILE: tl.constexpr):
    # ... main computation ...

    if EPILOGUE*SUBTILE:
        # Split BLOCK*M x BLOCK*N into 2 BLOCK*M x (BLOCK*N//2) chunks
        acc = tl.reshape(accumulator, (BLOCK*M, 2, BLOCK*N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        # Store in two parts
        c0 = acc0.to(dtype)
        c*desc.store([offs*cm, offs*cn], c0)

        c1 = acc1.to(dtype)
        c*desc.store([offs*cm, offs*cn + BLOCK*N // 2], c1)
    else:
        accumulator = accumulator.to(dtype)
        c*desc.store([offs*cm, offs*cn], accumulator)
::


**Why epilogue subtiling?**
- Reduces shared memory usage in epilogue
- Frees SRAM for more pipeline stages
- Improves register utilization
- Can increase overall throughput

Proton Profiler Integration
---------------------------

This tutorial demonstrates using Triton's built-in profiler:

.. code-block:: python

import triton.profiler as proton

Start profiling
===============
proton.start("matmul", hook="triton")
proton.deactivate()  # Don't profile initialization

Run benchmarks
==============
for K in range(K*min, K*max, K*step):
    proton.activate(0)
    for * in range(reps):
        matmul(a, b)
    proton.deactivate(0)

Finalize and show results
=========================
proton.finalize()
show*profile("fp16", "matmul")
::


**Metrics collected:**
- Time per kernel (ms)
- TFLOPS (teraflops per second)
- Memory bandwidth utilization
- Kernel launch counts

Viewing Profile Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

def show*profile(precision, profile*name):
    import triton.profiler.viewer as proton*viewer

    metric*names = ["time/ms"]
    if precision == 'fp8':
        metric*names = ["tflop8/s"] + metric*names
    elif precision == 'fp16':
        metric*names = ["tflop16/s"] + metric*names

    file*name = f"{profile*name}.hatchet"
    tree, metrics = proton*viewer.parse(metric*names, file*name)
    proton*viewer.print*tree(tree, metrics)
::


Output shows hierarchical breakdown:
::

matmul
├─ naive [M=8192, N=8192, K=512]      1.23 ms, 220 tflop16/s
├─ persistent [M=8192, N=8192, K=512] 1.15 ms, 235 tflop16/s
├─ tma [M=8192, N=8192, K=512]        1.05 ms, 257 tflop16/s
└─ tma*persistent [...]               0.98 ms, 276 tflop16/s
::


Device-side vs Host-side Descriptors
------------------------------------

Host-side (TensorDescriptor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from triton.tools.tensor*descriptor import TensorDescriptor

Created on CPU, passed to GPU
=============================
a*desc = TensorDescriptor.from*tensor(a, [BLOCK*M, BLOCK*K])
matmul*kernel*tma`grid <a*desc, b*desc, c*desc, ...>`_
::


**Pros:** Simpler, works on Hopper
**Cons:** Descriptor creation overhead, limited flexibility

Device-side (tl.make*tensor*descriptor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.jit
def kernel(a*ptr, ...):
    # Created inside kernel
    a*desc = tl.make*tensor*descriptor(
        a*ptr,
        shape=[M, K],
        strides=[K, 1],
        block*shape=[BLOCK*M, BLOCK*K],
    )
    a = a*desc.load([offs*am, offs*k])
::


**Pros:** No host overhead, required for Blackwell warp spec
**Cons:** Only available on newer GPUs

Warp Specialization Details
---------------------------

How It Works
~~~~~~~~~~~~

.. code-block:: python

for ki in tl.range(k*tiles, warp*specialize=True):
    # Hardware may assign:
    # Warps 0-3: Producer (memory loads)
    # Warps 4-7: Consumer (compute)
    a = a*desc.load([offs*am, offs*k])  # Producer
    b = b*desc.load([offs*bn, offs*k])  # Producer
    acc = tl.dot(a, b.T, acc)          # Consumer
::


**Without warp specialization:**
- All warps do: load A → load B → compute → repeat
- Memory and compute are serialized

**With warp specialization:**
- Producer warps continuously load data
- Consumer warps continuously compute
- Better overlap, higher throughput

Requirements
~~~~~~~~~~~~

.. code-block:: python

HAS*WARP*SPECIALIZE = supports*ws() and HAS*TENSOR*DESC

def supports*ws():
    return is*cuda() and torch.cuda.get*device*capability()[0] >= 9

On Hopper: Software pipelining
==============================
On Blackwell: Hardware warp specialization
==========================================
::


Flattening in Persistent Loops
------------------------------

.. code-block:: python

for tile*id in tl.range(start*pid, num*tiles, NUM*SMS,
                       flatten=True,
                       warp*specialize=WARP*SPECIALIZE):
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
==============================
flatten = False if (warp*specialize and is*hopper()) else True
::


Performance Comparison
----------------------

Expected speedups (relative to naive):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Variant | FP16 Speedup | FP8 Speedup | Notes |
|---------|--------------|-------------|-------|
| Persistent | 1.05-1.15× | 1.05-1.1× | Saves launch overhead |
| TMA | 1.15-1.25× | 1.2-1.3× | Better memory access |
| TMA Persistent | 1.2-1.35× | 1.25-1.4× | Combined benefits |
| + Warp Spec (Blackwell) | 1.3-1.5× | 1.4-1.6× | Hardware overlap |

When Each Variant Wins
~~~~~~~~~~~~~~~~~~~~~~

**Naive:**
- Small matrices (< 1024×1024)
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
- Large matrices (≥ 4096×4096)
- Blackwell GPUs
- Maximum performance needed

Precision Support
-----------------

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

dtype = torch.float8*e4m3fn  # E4M3 format
a = torch.randn((M, K), dtype=torch.float16).to(dtype)
b = torch.randn((K, N), dtype=torch.float16).to(dtype)
::

- Requires compute capability ≥ 9.0
- 2× speedup potential
- Lower precision, faster compute
- Ideal for inference

Auto-tuning Configuration
-------------------------

.. code-block:: python

def matmul*get*configs():
    return [
        triton.Config(
            {'BLOCK*SIZE*M': BM, 'BLOCK*SIZE*N': BN,
             'BLOCK*SIZE*K': BK, 'GROUP*SIZE*M': 8},
            num*stages=s, num*warps=w
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]
::


**Parameters:**
- ``BLOCK*SIZE*M/N/K``: Tile dimensions
- ``num*stages``: Pipeline depth (2-4 typical)
- ``num*warps``: Threads per CTA (4 or 8)
- ``GROUP*SIZE*M``: Swizzling factor for L2 locality

Command-line Usage
------------------

.. code-block:: bash

FP16 matmul
===========
python 09-persistent-matmul.py --prec fp16 --K*range 128 1024 --K*step 128

FP8 matmul (requires Hopper+)
=============================
python 09-persistent-matmul.py --prec fp8 --K 512

Profile specific K dimension
============================
python 09-persistent-matmul.py --prec fp16 -K 2048
::


**Note:** May fail on GPUs with small shared memory (e.g., RTX 4090). Reduce ``num*stages`` if needed.

Common Pitfalls
---------------

1. Wrong B Matrix Layout for TMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

TMA expects B to be transposed
==============================
b = b.T.contiguous()  # Make sure it's contiguous!

matmul*tma(a, b, warp*specialize=False)
::


2. Mixing Host and Device Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Bad: Using both in same kernel
==============================
a*desc = TensorDescriptor.from*tensor(a, ...)  # Host-side
tl.make*tensor*descriptor(...)  # Device-side

Good: Pick one approach
=======================
if HAS*HOST*TENSOR*DESC:
    use*host*descriptors()
else:
    use*device*descriptors()
::


3. Forgetting FP8 Support Check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

if args.prec == 'fp8':
    if not hasattr(torch, 'float8*e4m3fn') or not is_cuda():
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
