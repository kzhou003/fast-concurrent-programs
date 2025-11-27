Tutorial 10: Block Scaled Matrix Multiplication
===============================================

Overview
--------

**Block scaled matmul** enables low-precision matrix multiplication (FP4/FP8) with per-block scaling factors. This technique is crucial for:

- **Quantized inference** - Running large language models with reduced memory
- **Mixed precision training** - Different precision for different operations
- **Memory bandwidth optimization** - Fewer bits transferred = faster compute
- **Hardware acceleration** - Specialized tensor cores for low-precision ops

This tutorial supports **four quantization formats**:
1. **nvfp4** - NVIDIA's FP4 format (16 elements per scale, NVIDIA-only)
2. **mxfp4** - Microscaling FP4 (32 elements per scale, OCP standard)
3. **mxfp8** - Microscaling FP8 (32 elements per scale)
4. **mixed** - FP8 × FP4 mixed precision

**Hardware requirements:**
- **NVIDIA**: Blackwell (compute capability 10.0+) with 5th-gen Tensor Cores
- **AMD**: CDNA4 architecture (MI300 series) with scaled MFMA instructions

Key Concepts
------------

Block Scaling Fundamentals
~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of storing full-precision values, we store:
::

Low-precision value + Scale factor
::


**Standard matmul:**
::

C = A @ B
::


**Block scaled matmul:**
::

C = (A * scale*a) @ (B * scale*b)

where:
  A, B are low-precision (fp4/fp8)
  scale*a, scale*b are per-block scale factors
  C is full precision (fp16/fp32)
::


Quantization Formats
~~~~~~~~~~~~~~~~~~~~

| Format | Bits/elem | Vec Size | Hardware | Notes |
|--------|-----------|----------|----------|-------|
| nvfp4 | 4 | 16 | NVIDIA Blackwell | Proprietary, optimized for NVIDIA |
| mxfp4 | 4 | 32 | NVIDIA/AMD | OCP standard, better portability |
| mxfp8 | 8 | 32 | NVIDIA/AMD | Higher precision, still efficient |
| mixed | 4×8 | varies | NVIDIA Blackwell | A in fp8, B in fp4 |

**Vec Size** = number of elements sharing one scale factor

Memory Savings
~~~~~~~~~~~~~~

::

FP16: 2 bytes per element
FP8:  1 byte per element  → 2× memory reduction
FP4:  0.5 bytes per element → 4× memory reduction

Plus scale factors:
  - 1 scale per 16-32 elements
  - Scales typically stored as fp8 or e8m0 (exponent only)
  - Overhead: ~3-6% of original size
::


**Example for 8192×8192 matrix:**
- FP16: 128 MB
- FP8 + scales: 64 MB + 2 MB = 66 MB (48% reduction)
- FP4 + scales: 32 MB + 2 MB = 34 MB (73% reduction)

Scale Preshuffling (NVIDIA)
---------------------------

Why Preshuffling?
~~~~~~~~~~~~~~~~~

Tensor cores load scales in specific patterns. To avoid non-contiguous access, scales must be reorganized:

.. code-block:: python

Original linear layout: [M, K // VEC*SIZE]
==========================================
Each row has K // VEC*SIZE scales
=================================

Preshuffled layout: [M // 128, K // VEC*SIZE // 4, 32, 4, 4]
============================================================
Organized for 128-element blocks along M
========================================
::


5D Preshuffled Layout
~~~~~~~~~~~~~~~~~~~~~

::

Dimension breakdown:
  [M // 128]         - Number of 128-row blocks
  [K // VEC*SIZE // 4] - Number of K scale blocks
  [32]               - 32 rows per sub-block
  [4]                - 4 scale groups
  [4]                - 4 scales per group
::


**Memory access pattern:**
.. code-block:: python

for each BLOCK*M x BLOCK*K tile:
    Load 128 rows × (BLOCK*K // VEC*SIZE) scales contiguously
    No strided access → better memory bandwidth
::


Reshaping and Transposing
~~~~~~~~~~~~~~~~~~~~~~~~~

Inside the kernel:
.. code-block:: python

Load in 5D preshuffled format
=============================
scale*a = a*scale*desc.load([0, offs*scale*m, offs*scale*k, 0, 0])

Reshape to 5D
=============
scale*a = scale*a.reshape(rep*m, rep*k, 32, 4, 4)

Transpose to logical 2D layout
==============================
scale*a = scale*a.trans(0, 3, 2, 1, 4).reshape(BLOCK*M, BLOCK*K // VEC*SIZE)

Now ready for tl.dot*scaled
===========================
::


Scale Preshuffling (AMD CDNA4)
------------------------------

MFMA Scale Organization
~~~~~~~~~~~~~~~~~~~~~~~

AMD's MFMA (Matrix Fused Multiply-Add) instructions require different shuffling:

.. code-block:: python

def shuffle*scales*cdna4(scales, mfma*nonkdim):
    """
    Reorganize scales for MFMA instructions.

    mfma*nonkdim: 16 or 32
      - 16: mfma*scaled*16x16x128
      - 32: mfma*scaled*32x32x64
    """
    sm, sn = scales.shape

    if mfma*nonkdim == 32:
        # For 32x32 MFMA: pack 4 ops in order 0,1,2,3
        scales*shuffled = scales.view(sm // 32, 32, sn // 8, 4, 2, 1)
        scales*shuffled = scales*shuffled.permute(0, 2, 4, 1, 3, 5).contiguous()

    elif mfma*nonkdim == 16:
        # For 16x16 MFMA: pack 4 ops in order 0,2,1,3
        scales*shuffled = scales.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        scales*shuffled = scales*shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()

    return scales*shuffled.view(sm // 32, sn * 32)
::


**Key insight:** Each thread needs 4 scale values for 4 MFMA operations, packed contiguously.

Thread-level Access Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Without shuffling:
  Thread 0 needs scales at: [0, 32, 64, 96] - strided access

With shuffling:
  Thread 0 needs scales at: [0, 1, 2, 3]    - contiguous access
  Thread 1 needs scales at: [4, 5, 6, 7]
  ...
::


**Benefits:**
- Vectorized memory loads (4 bytes at once)
- Better global memory coalescing
- Lower LDS (shared memory) pressure

Code Walkthrough
----------------

1. NVIDIA Kernel (TMA-based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.jit
def block*scaled*matmul*kernel(
    a*desc, b*desc, c*desc,
    a*scale*desc, b*scale*desc,
    M, N, K,
    output*type: tl.constexpr,
    ELEM*PER*BYTE*A: tl.constexpr,  # 1 for fp8, 2 for fp4
    ELEM*PER*BYTE*B: tl.constexpr,
    VEC*SIZE: tl.constexpr,          # 16 for nvfp4, 32 for mxfp4/mxfp8
    BLOCK*M: tl.constexpr,
    BLOCK*N: tl.constexpr,
    BLOCK*K: tl.constexpr,
    rep*m: tl.constexpr,             # BLOCK*M // 128
    rep*n: tl.constexpr,             # BLOCK*N // 128
    rep*k: tl.constexpr,             # BLOCK*K // VEC*SIZE // 4
    NUM*STAGES: tl.constexpr,
):
    pid = tl.program*id(axis=0)
    num*pid*m = tl.cdiv(M, BLOCK*M)
    pid*m = pid % num*pid*m
    pid*n = pid // num*pid*m

    offs*am = pid*m * BLOCK*M
    offs*bn = pid*n * BLOCK*N
    offs*k*a = 0
    offs*k*b = 0
    offs*scale*m = pid*m * rep*m
    offs*scale*n = pid*n * rep*n
    offs*scale*k = 0

    accumulator = tl.zeros((BLOCK*M, BLOCK*N), dtype=tl.float32)

    # Pipelined loop over K
    for k in tl.range(0, tl.cdiv(K, BLOCK*K), num*stages=NUM*STAGES):
        # Load data blocks
        a = a*desc.load([offs*am, offs*k*a])
        b = b*desc.load([offs*bn, offs*k*b])

        # Load and reshape scales
        scale*a = a*scale*desc.load([0, offs*scale*m, offs*scale*k, 0, 0])
        scale*b = b*scale*desc.load([0, offs*scale*n, offs*scale*k, 0, 0])

        # Reshape from 5D to 2D
        scale*a = scale*a.reshape(rep*m, rep*k, 32, 4, 4) \
                         .trans(0, 3, 2, 1, 4) \
                         .reshape(BLOCK*M, BLOCK*K // VEC*SIZE)
        scale*b = scale*b.reshape(rep*n, rep*k, 32, 4, 4) \
                         .trans(0, 3, 2, 1, 4) \
                         .reshape(BLOCK*N, BLOCK*K // VEC*SIZE)

        # Perform scaled dot product
        if ELEM*PER*BYTE*A == 1 and ELEM*PER*BYTE*B == 2:
            # Mixed precision: A is fp8, B is fp4
            accumulator = tl.dot*scaled(
                a, scale*a, "e4m3",
                b.T, scale*b, "e2m1",
                accumulator
            )
        elif ELEM*PER*BYTE*A == 2 and ELEM*PER*BYTE*B == 2:
            # Both fp4
            accumulator = tl.dot*scaled(
                a, scale*a, "e2m1",
                b.T, scale*b, "e2m1",
                accumulator
            )
        else:
            # Both fp8
            accumulator = tl.dot*scaled(
                a, scale*a, "e4m3",
                b.T, scale*b, "e4m3",
                accumulator
            )

        # Advance pointers
        offs*k*a += BLOCK*K // ELEM*PER*BYTE*A
        offs*k*b += BLOCK*K // ELEM*PER*BYTE*B
        offs*scale*k += rep*k

    # Store result
    c*desc.store([offs*am, offs*bn], accumulator.to(output*dtype))
::


**Key operations:**
- ``tl.dot*scaled()`` - Triton's scaled matmul intrinsic
- Format strings: ``"e4m3"`` (fp8), ``"e2m1"`` (fp4)
- Automatic broadcast of scales across VEC*SIZE elements

2. AMD CDNA4 Kernel
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.jit
def block*scaled*matmul*kernel*cdna4(
    a*ptr, b*ptr, c*ptr,
    a*scales*ptr, b*scales*ptr,
    M, N, K,
    stride*am, stride*ak, stride*bk, stride*bn,
    stride*cm, stride*cn,
    stride*asm, stride*ask, stride*bsn, stride*bsk,
    BLOCK*M: tl.constexpr,
    BLOCK*N: tl.constexpr,
    BLOCK*K: tl.constexpr,
    mfma*nonkdim: tl.constexpr,  # 16 or 32
):
    pid = tl.program*id(axis=0)
    num*pid*n = tl.cdiv(N, BLOCK*N)
    pid*m = pid // num*pid*n
    pid*n = pid % num*pid*n

    # Packed data: 2 fp4 elements per byte
    offs*k = tl.arange(0, BLOCK*K // 2)
    offs*am = (pid*m * BLOCK*M + tl.arange(0, BLOCK*M)) % M
    offs*bn = (pid*n * BLOCK*N + tl.arange(0, BLOCK*N)) % N

    a*ptrs = a*ptr + offs*am[:, None] * stride*am + offs*k[None, :] * stride*ak
    b*ptrs = b*ptr + offs*k[:, None] * stride*bk + offs*bn[None, :] * stride*bn

    # Scale pointers (32 elements per scale)
    SCALE*GROUP*SIZE = 32
    offs*ks = tl.arange(0, BLOCK*K // SCALE*GROUP*SIZE * 32)

    offs*asm = (pid*m * (BLOCK*M // 32) + tl.arange(0, BLOCK*M // 32)) % M
    a*scale*ptrs = a*scales*ptr + offs*asm[:, None] * stride*asm + \
                   offs*ks[None, :] * stride*ask

    offs*asn = (pid*n * (BLOCK*N // 32) + tl.arange(0, BLOCK*N // 32)) % N
    b*scale*ptrs = b*scales*ptr + offs*asn[:, None] * stride*bsn + \
                   offs*ks[None, :] * stride*bsk

    accumulator = tl.zeros((BLOCK*M, BLOCK*N), dtype=tl.float32)

    num*k*iter = tl.cdiv(K, BLOCK*K // 2)
    for k in range(num*k*iter):
        # Load and unshuffle scales
        if mfma*nonkdim == 32:
            a*scales = tl.load(a*scale*ptrs) \
                .reshape(BLOCK*M // 32, BLOCK*K // 32 // 8, 2, 32, 4, 1) \
                .permute(0, 3, 1, 4, 2, 5) \
                .reshape(BLOCK*M, BLOCK*K // 32)
            b*scales = tl.load(b*scale*ptrs) \
                .reshape(BLOCK*N // 32, BLOCK*K // 32 // 8, 2, 32, 4, 1) \
                .permute(0, 3, 1, 4, 2, 5) \
                .reshape(BLOCK*N, BLOCK*K // 32)
        elif mfma*nonkdim == 16:
            a*scales = tl.load(a*scale*ptrs) \
                .reshape(BLOCK*M // 32, BLOCK*K // 32 // 8, 4, 16, 2, 2, 1) \
                .permute(0, 5, 3, 1, 4, 2, 6) \
                .reshape(BLOCK*M, BLOCK*K // 32)
            # Similar for b*scales

        # Load packed data
        a = tl.load(a*ptrs)
        b = tl.load(b*ptrs)

        # Scaled matmul
        accumulator += tl.dot*scaled(a, a*scales, "e2m1",
                                     b, b*scales, "e2m1")

        # Advance pointers
        a*ptrs += (BLOCK*K // 2) * stride*ak
        b*ptrs += (BLOCK*K // 2) * stride*bk
        a*scale*ptrs += BLOCK*K * stride*ask
        b*scale*ptrs += BLOCK*K * stride*bsk

    c = accumulator.to(c*ptr.type.element*ty)
    # Store with write-through cache hint
    tl.store(c*ptrs, c, mask=c*mask, cache*modifier=".wt")
::


**AMD-specific features:**
- Explicit unshuffling of scales in kernel
- Support for two MFMA shapes (16x16, 32x32)
- Write-through cache modifier for better performance
- E8M0 scale format (exponent-only, 8 bits)

Initialization and Setup
------------------------

NVIDIA Version
~~~~~~~~~~~~~~

.. code-block:: python

def initialize*block*scaled(M, N, K, block*scale*type="nvfp4"):
    # Configuration based on format
    BLOCK*M = 128
    BLOCK*N = 256
    BLOCK*K = 256 if "fp4" in block*scale*type else 128
    VEC*SIZE = 16 if block*scale*type == "nvfp4" else 32
    ELEM*PER*BYTE*A = 2 if "fp4" in block*scale*type else 1
    ELEM*PER*BYTE*B = 1 if block*scale*type == "mxfp8" else 2

    # Generate random data using mxfp helper
    from triton.tools.mxfp import MXFP4Tensor

    a*ref = MXFP4Tensor(size=(M, K), device="cuda").random()
    b*ref = MXFP4Tensor(size=(N, K), device="cuda").random()  # Transposed

    # Pack for fp4 (2 elements per byte)
    if "fp4" in block*scale*type and block*scale*type != "mxfp8":
        a = a*ref.to*packed*tensor(dim=1)
    else:
        a = a*ref.to(torch.float8*e4m3fn)

    # Create TMA descriptors
    a*desc = TensorDescriptor.from*tensor(a, [BLOCK*M, BLOCK*K // ELEM*PER*BYTE*A])
    b*desc = TensorDescriptor.from*tensor(b, [BLOCK*N, BLOCK*K // ELEM*PER*BYTE*B])

    # Generate scales in 5D preshuffled format
    a*scale*shape = [M // 128, K // VEC*SIZE // 4, 32, 16]
    b*scale*shape = [N // 128, K // VEC*SIZE // 4, 32, 16]

    epsilon = 1e-8
    a*scale = torch.rand(a*scale*shape, device="cuda") + epsilon
    b*scale = torch.rand(b*scale*shape, device="cuda") + epsilon

    # Reshape to 5D TMA format
    a*scale = a*scale.reshape(1, a*scale*shape[0], a*scale.shape[1], 2, 256)
    b*scale = b*scale.reshape(1, b*scale*shape[0], b*scale.shape[1], 2, 256)

    a*scale*desc = TensorDescriptor.from*tensor(a*scale, block*shape=[1, rep*m, rep*k, 2, 256])
    b*scale*desc = TensorDescriptor.from*tensor(b*scale, block*shape=[1, rep*n, rep*k, 2, 256])

    return a*desc, a*scale*desc, b*desc, b*scale*desc, ...
::


AMD Version
~~~~~~~~~~~

.. code-block:: python

def initialize*block*scaled*amd(M, N, K, mfma*nonkdim):
    BLOCK*M = 128
    BLOCK*N = 128
    BLOCK*K = 256

    x = MXFP4Tensor(size=(M, K), device="cuda").random()
    w = MXFP4Tensor(size=(N, K), device="cuda").random()

    # E8M0 scales (exponent only, 8 bits)
    x*scales = torch.randint(124, 128, (K // 32, M), dtype=torch.uint8, device="cuda")
    w*scales = torch.randint(124, 128, (K // 32, N), dtype=torch.uint8, device="cuda")

    x*scales = x*scales.T
    w*scales = w*scales.T

    # Preshuffle for MFMA access pattern
    x*scales*shuffled = shuffle*scales*cdna4(x*scales, mfma*nonkdim)
    w*scales*shuffled = shuffle*scales*cdna4(w*scales, mfma*nonkdim)

    # Pack 2 fp4 elements per byte
    x*packed = x.to*packed*tensor(dim=1)
    w*packed = w.to*packed*tensor(dim=1)

    return x*packed, w*packed, x*scales*shuffled, w*scales*shuffled, ...
::


Performance Characteristics
---------------------------

Theoretical Speedup
~~~~~~~~~~~~~~~~~~~

Compute-bound workload:
::

FP16 throughput: 312 TFLOPS (H100)
FP8 throughput:  989 TFLOPS (H100) → 3.17× faster
FP4 throughput:  1978 TFLOPS (theoretical) → 6.3× faster
::


Memory bandwidth reduction:
::

FP16: 2 bytes/elem
FP8:  1 byte/elem + scales → ~45% savings
FP4:  0.5 bytes/elem + scales → ~70% savings
::


Real-world Performance
~~~~~~~~~~~~~~~~~~~~~~

From benchmarking on H100:
- **mxfp8**: 1.8-2.2× speedup over FP16
- **mxfp4**: 2.5-3.5× speedup over FP16
- **nvfp4**: 3.0-4.0× speedup over FP16 (NVIDIA-specific optimizations)
- **mixed (fp8×fp4)**: 2.2-3.0× speedup

**Factors affecting performance:**
- Matrix size (larger = better amortization)
- Scale overhead (smaller VEC*SIZE = more overhead)
- Memory vs compute bound (FP4 helps more when memory-bound)

Numerical Considerations
------------------------

Quantization Error
~~~~~~~~~~~~~~~~~~

.. code-block:: python

FP16 range: ±65504, ~3 decimal digits
=====================================
FP8 (E4M3) range: ±448, ~2 decimal digits
=========================================
FP4 (E2M1) range: ±6, ~1 decimal digit
======================================
::


**Per-block scaling helps:**
.. code-block:: python

Without scaling:
================
fp4*val = quantize*fp16*to*fp4(1234.5)  # Overflow!

With scaling:
=============
scale = 1234.5 / 6.0  # ~205
fp4*val = quantize*fp16*to*fp4(1234.5 / scale)  # ≈ 6
reconstructed = fp4*val * scale  # ≈ 1234.5
::


E8M0 Scale Format (AMD)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

def e8m0*to*f32(x):
    """Convert 8-bit exponent-only to float32."""
    # No mantissa, only exponent
    # Value = 2^(exponent - 127)
    return 2 ** ((x - 127).to(torch.float32))

Example:
========
x = 135 → 2^(135-127) = 2^8 = 256
=================================
x = 127 → 2^0 = 1
=================
x = 119 → 2^(-8) = 0.00390625
=============================
::


**Why exponent-only?**
- Scales are typically powers of 2
- 8 bits gives wide dynamic range
- Simpler hardware implementation
- Exact representation for power-of-2 scales

Usage Examples
--------------

Command-line Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

NVIDIA FP4
==========
python 10-block-scaled-matmul.py --format nvfp4 --K*range 512 8192 --bench

NVIDIA FP8
==========
python 10-block-scaled-matmul.py --format mxfp8 --K 4096 --bench

AMD MXFP4 (automatic detection)
===============================
python 10-block-scaled-matmul.py --format mxfp4 --bench

Mixed precision
===============
python 10-block-scaled-matmul.py --format mixed --K*range 2048 8192 --K*step 2048
::


Validation
~~~~~~~~~~

.. code-block:: python

NVIDIA
======
validate*block*scaled(8192, 8192, 8192, block*scale*type="nvfp4")
✅ (pass nvfp4)
==============

AMD with both MFMA shapes
=========================
validate*block*scaled*amd(8192, 8192, 8192, block*scale*type="mxfp4", mfma*nonkdim=16)
✅ (pass mxfp4, mfma*nonk*dim 16)
================================

validate*block*scaled*amd(8192, 8192, 8192, block*scale*type="mxfp4", mfma*nonkdim=32)
✅ (pass mxfp4, mfma*nonk*dim 32)
================================
::


Common Pitfalls
---------------

1. Unsupported Hardware
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

if not supports*block*scaling():
    print("⛔ This example requires GPU support for block scaled matmul")
    exit(1)

def supports*block*scaling():
    return (is*cuda() and torch.cuda.get*device*capability()[0] == 10) or \
           is*hip*cdna4()
::


2. Format Mismatch
~~~~~~~~~~~~~~~~~~

.. code-block:: python

Bad: Using AMD kernel with NVIDIA formats
=========================================
if is*hip*cdna4():
    assert args.format == "mxfp4", "AMD only supports mxfp4"
::


3. Wrong Scale Shape
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Scales must match data dimensions
=================================
assert a*scale.shape == [M // 128, K // VEC*SIZE // 4, 32, 16]

After packing to 5D
===================
assert a*scale.shape == [1, M // 128, K // VEC*SIZE // 4, 2, 256]
::


4. Missing TMA Allocator
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from typing import Optional

def alloc*fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set*allocator(alloc*fn)
::


Summary
-------

**Block scaled matmul** enables efficient low-precision matrix multiplication:

- **4-8× memory reduction** compared to FP16
- **2-4× compute speedup** on specialized hardware
- **Per-block scaling** maintains numerical accuracy
- **Hardware acceleration** via 5th-gen Tensor Cores (NVIDIA) and CDNA4 (AMD)

**Supported formats:**
- ``nvfp4``: NVIDIA-optimized FP4 (16 elem/scale)
- ``mxfp4``: OCP standard FP4 (32 elem/scale)
- ``mxfp8``: OCP standard FP8 (32 elem/scale)
- ``mixed``: FP8×FP4 mixed precision

**Key techniques:**
- **Scale preshuffling** for contiguous memory access
- **TMA descriptors** for hardware-accelerated loads
- **tl.dot*scaled** intrinsic for scaled operations
- **5D tensor layouts** optimized for tensor cores

**When to use:**
- Large language model inference
- Memory-constrained workloads
- Inference on edge devices
- Training with mixed precision

**Requirements:**
- NVIDIA Blackwell (CC 10.0+) or AMD CDNA4
- Triton with mxfp support
- Careful attention to scale layout and format

This is the cutting edge of GPU matrix multiplication, enabling the next generation of efficient AI!
