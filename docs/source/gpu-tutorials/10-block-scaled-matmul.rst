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
4. **mixed** - FP8 x FP4 mixed precision

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

C = (A * scale_a) @ (B * scale_b)

where:
  A, B are low-precision (fp4/fp8)
  scale_a, scale_b are per-block scale factors
  C is full precision (fp16/fp32)
::


Quantization Formats
~~~~~~~~~~~~~~~~~~~~

| Format | Bits/elem | Vec Size | Hardware | Notes |
|--------|-----------|----------|----------|-------|
| nvfp4 | 4 | 16 | NVIDIA Blackwell | Proprietary, optimized for NVIDIA |
| mxfp4 | 4 | 32 | NVIDIA/AMD | OCP standard, better portability |
| mxfp8 | 8 | 32 | NVIDIA/AMD | Higher precision, still efficient |
| mixed | 4x8 | varies | NVIDIA Blackwell | A in fp8, B in fp4 |

**Vec Size** = number of elements sharing one scale factor

Memory Savings
~~~~~~~~~~~~~~

::

FP16: 2 bytes per element
FP8:  1 byte per element  -> 2x memory reduction
FP4:  0.5 bytes per element -> 4x memory reduction

Plus scale factors:
  - 1 scale per 16-32 elements
  - Scales typically stored as fp8 or e8m0 (exponent only)
  - Overhead: ~3-6% of original size
::


**Example for 8192x8192 matrix:**
- FP16: 128 MB
- FP8 + scales: 64 MB + 2 MB = 66 MB (48% reduction)
- FP4 + scales: 32 MB + 2 MB = 34 MB (73% reduction)

Scale Preshuffling (NVIDIA)
---------------------------

Why Preshuffling?
~~~~~~~~~~~~~~~~~

Tensor cores load scales in specific patterns. To avoid non-contiguous access, scales must be reorganized:

.. code-block:: python

Original linear layout: [M, K // VEC_SIZE]
==========================================
Each row has K // VEC_SIZE scales
=================================

Preshuffled layout: [M // 128, K // VEC_SIZE // 4, 32, 4, 4]
============================================================
Organized for 128-element blocks along M
========================================
::


5D Preshuffled Layout
~~~~~~~~~~~~~~~~~~~~~

::

Dimension breakdown:
  [M // 128]         - Number of 128-row blocks
  [K // VEC_SIZE // 4] - Number of K scale blocks
  [32]               - 32 rows per sub-block
  [4]                - 4 scale groups
  [4]                - 4 scales per group
::


**Memory access pattern:**
.. code-block:: python

for each BLOCK_M x BLOCK_K tile:
    Load 128 rows x (BLOCK_K // VEC_SIZE) scales contiguously
    No strided access -> better memory bandwidth
::


Reshaping and Transposing
~~~~~~~~~~~~~~~~~~~~~~~~~

Inside the kernel:
.. code-block:: python

Load in 5D preshuffled format
=============================
scale_a = a_scale*desc.load([0, offs_scale*m, offs_scale*k, 0, 0])

Reshape to 5D
=============
scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4)

Transpose to logical 2D layout
==============================
scale_a = scale_a.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)

Now ready for tl.dot_scaled
===========================
::


Scale Preshuffling (AMD CDNA4)
------------------------------

MFMA Scale Organization
~~~~~~~~~~~~~~~~~~~~~~~

AMD's MFMA (Matrix Fused Multiply-Add) instructions require different shuffling:

.. code-block:: python

def shuffle_scales*cdna4(scales, mfma_nonkdim):
    """
    Reorganize scales for MFMA instructions.

    mfma_nonkdim: 16 or 32
      - 16: mfma_scaled*16x16x128
      - 32: mfma_scaled*32x32x64
    """
    sm, sn = scales.shape

    if mfma_nonkdim == 32:
        # For 32x32 MFMA: pack 4 ops in order 0,1,2,3
        scales_shuffled = scales.view(sm // 32, 32, sn // 8, 4, 2, 1)
        scales_shuffled = scales_shuffled.permute(0, 2, 4, 1, 3, 5).contiguous()

    elif mfma_nonkdim == 16:
        # For 16x16 MFMA: pack 4 ops in order 0,2,1,3
        scales_shuffled = scales.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()

    return scales_shuffled.view(sm // 32, sn * 32)
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
def block_scaled*matmul_kernel(
    a_desc, b_desc, c_desc,
    a_scale*desc, b_scale*desc,
    M, N, K,
    output_type: tl.constexpr,
    ELEM_PER*BYTE_A: tl.constexpr,  # 1 for fp8, 2 for fp4
    ELEM_PER*BYTE_B: tl.constexpr,
    VEC_SIZE: tl.constexpr,          # 16 for nvfp4, 32 for mxfp4/mxfp8
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    rep_m: tl.constexpr,             # BLOCK_M // 128
    rep_n: tl.constexpr,             # BLOCK_N // 128
    rep_k: tl.constexpr,             # BLOCK_K // VEC_SIZE // 4
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid*m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid*m
    pid_n = pid // num_pid*m

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k*a = 0
    offs_k*b = 0
    offs_scale*m = pid_m * rep_m
    offs_scale*n = pid_n * rep_n
    offs_scale*k = 0

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pipelined loop over K
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        # Load data blocks
        a = a_desc.load([offs_am, offs_k*a])
        b = b_desc.load([offs_bn, offs_k*b])

        # Load and reshape scales
        scale_a = a_scale*desc.load([0, offs_scale*m, offs_scale*k, 0, 0])
        scale_b = b_scale*desc.load([0, offs_scale*n, offs_scale*k, 0, 0])

        # Reshape from 5D to 2D
        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4) \
                         .trans(0, 3, 2, 1, 4) \
                         .reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4) \
                         .trans(0, 3, 2, 1, 4) \
                         .reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        # Perform scaled dot product
        if ELEM_PER*BYTE_A == 1 and ELEM_PER*BYTE_B == 2:
            # Mixed precision: A is fp8, B is fp4
            accumulator = tl.dot_scaled(
                a, scale_a, "e4m3",
                b.T, scale_b, "e2m1",
                accumulator
            )
        elif ELEM_PER*BYTE_A == 2 and ELEM_PER*BYTE_B == 2:
            # Both fp4
            accumulator = tl.dot_scaled(
                a, scale_a, "e2m1",
                b.T, scale_b, "e2m1",
                accumulator
            )
        else:
            # Both fp8
            accumulator = tl.dot_scaled(
                a, scale_a, "e4m3",
                b.T, scale_b, "e4m3",
                accumulator
            )

        # Advance pointers
        offs_k*a += BLOCK_K // ELEM_PER*BYTE_A
        offs_k*b += BLOCK_K // ELEM_PER*BYTE_B
        offs_scale*k += rep_k

    # Store result
    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))
::


**Key operations:**
- ``tl.dot_scaled()`` - Triton's scaled matmul intrinsic
- Format strings: ``"e4m3"`` (fp8), ``"e2m1"`` (fp4)
- Automatic broadcast of scales across VEC_SIZE elements

2. AMD CDNA4 Kernel
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

@triton.jit
def block_scaled*matmul_kernel*cdna4(
    a_ptr, b_ptr, c_ptr,
    a_scales*ptr, b_scales*ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask, stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    mfma_nonkdim: tl.constexpr,  # 16 or 32
):
    pid = tl.program_id(axis=0)
    num_pid*n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid*n
    pid_n = pid % num_pid*n

    # Packed data: 2 fp4 elements per byte
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Scale pointers (32 elements per scale)
    SCALE_GROUP*SIZE = 32
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP*SIZE * 32)

    offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, BLOCK_M // 32)) % M
    a_scale*ptrs = a_scales*ptr + offs_asm[:, None] * stride_asm + \
                   offs_ks[None, :] * stride_ask

    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % N
    b_scale*ptrs = b_scales*ptr + offs_asn[:, None] * stride_bsn + \
                   offs_ks[None, :] * stride_bsk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k*iter = tl.cdiv(K, BLOCK_K // 2)
    for k in range(num_k*iter):
        # Load and unshuffle scales
        if mfma_nonkdim == 32:
            a_scales = tl.load(a_scale*ptrs) \
                .reshape(BLOCK_M // 32, BLOCK_K // 32 // 8, 2, 32, 4, 1) \
                .permute(0, 3, 1, 4, 2, 5) \
                .reshape(BLOCK_M, BLOCK_K // 32)
            b_scales = tl.load(b_scale*ptrs) \
                .reshape(BLOCK_N // 32, BLOCK_K // 32 // 8, 2, 32, 4, 1) \
                .permute(0, 3, 1, 4, 2, 5) \
                .reshape(BLOCK_N, BLOCK_K // 32)
        elif mfma_nonkdim == 16:
            a_scales = tl.load(a_scale*ptrs) \
                .reshape(BLOCK_M // 32, BLOCK_K // 32 // 8, 4, 16, 2, 2, 1) \
                .permute(0, 5, 3, 1, 4, 2, 6) \
                .reshape(BLOCK_M, BLOCK_K // 32)
            # Similar for b_scales

        # Load packed data
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # Scaled matmul
        accumulator += tl.dot_scaled(a, a_scales, "e2m1",
                                     b, b_scales, "e2m1")

        # Advance pointers
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        a_scale*ptrs += BLOCK_K * stride_ask
        b_scale*ptrs += BLOCK_K * stride_bsk

    c = accumulator.to(c_ptr.type.element_ty)
    # Store with write-through cache hint
    tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")
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

def initialize_block*scaled(M, N, K, block_scale*type="nvfp4"):
    # Configuration based on format
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale*type else 128
    VEC_SIZE = 16 if block_scale*type == "nvfp4" else 32
    ELEM_PER*BYTE_A = 2 if "fp4" in block_scale*type else 1
    ELEM_PER*BYTE_B = 1 if block_scale*type == "mxfp8" else 2

    # Generate random data using mxfp helper
    from triton.tools.mxfp import MXFP4Tensor

    a_ref = MXFP4Tensor(size=(M, K), device="cuda").random()
    b_ref = MXFP4Tensor(size=(N, K), device="cuda").random()  # Transposed

    # Pack for fp4 (2 elements per byte)
    if "fp4" in block_scale*type and block_scale*type != "mxfp8":
        a = a_ref.to_packed*tensor(dim=1)
    else:
        a = a_ref.to(torch.float8_e4m3fn)

    # Create TMA descriptors
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER*BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER*BYTE_B])

    # Generate scales in 5D preshuffled format
    a_scale*shape = [M // 128, K // VEC_SIZE // 4, 32, 16]
    b_scale*shape = [N // 128, K // VEC_SIZE // 4, 32, 16]

    epsilon = 1e-8
    a_scale = torch.rand(a_scale*shape, device="cuda") + epsilon
    b_scale = torch.rand(b_scale*shape, device="cuda") + epsilon

    # Reshape to 5D TMA format
    a_scale = a_scale.reshape(1, a_scale*shape[0], a_scale.shape[1], 2, 256)
    b_scale = b_scale.reshape(1, b_scale*shape[0], b_scale.shape[1], 2, 256)

    a_scale*desc = TensorDescriptor.from_tensor(a_scale, block_shape=[1, rep_m, rep_k, 2, 256])
    b_scale*desc = TensorDescriptor.from_tensor(b_scale, block_shape=[1, rep_n, rep_k, 2, 256])

    return a_desc, a_scale*desc, b_desc, b_scale*desc, ...
::


AMD Version
~~~~~~~~~~~

.. code-block:: python

def initialize_block*scaled_amd(M, N, K, mfma_nonkdim):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256

    x = MXFP4Tensor(size=(M, K), device="cuda").random()
    w = MXFP4Tensor(size=(N, K), device="cuda").random()

    # E8M0 scales (exponent only, 8 bits)
    x_scales = torch.randint(124, 128, (K // 32, M), dtype=torch.uint8, device="cuda")
    w_scales = torch.randint(124, 128, (K // 32, N), dtype=torch.uint8, device="cuda")

    x_scales = x_scales.T
    w_scales = w_scales.T

    # Preshuffle for MFMA access pattern
    x_scales*shuffled = shuffle_scales*cdna4(x_scales, mfma_nonkdim)
    w_scales*shuffled = shuffle_scales*cdna4(w_scales, mfma_nonkdim)

    # Pack 2 fp4 elements per byte
    x_packed = x.to_packed*tensor(dim=1)
    w_packed = w.to_packed*tensor(dim=1)

    return x_packed, w_packed, x_scales*shuffled, w_scales*shuffled, ...
::


Performance Characteristics
---------------------------

Theoretical Speedup
~~~~~~~~~~~~~~~~~~~

Compute-bound workload:
::

FP16 throughput: 312 TFLOPS (H100)
FP8 throughput:  989 TFLOPS (H100) -> 3.17x faster
FP4 throughput:  1978 TFLOPS (theoretical) -> 6.3x faster
::


Memory bandwidth reduction:
::

FP16: 2 bytes/elem
FP8:  1 byte/elem + scales -> ~45% savings
FP4:  0.5 bytes/elem + scales -> ~70% savings
::


Real-world Performance
~~~~~~~~~~~~~~~~~~~~~~

From benchmarking on H100:
- **mxfp8**: 1.8-2.2x speedup over FP16
- **mxfp4**: 2.5-3.5x speedup over FP16
- **nvfp4**: 3.0-4.0x speedup over FP16 (NVIDIA-specific optimizations)
- **mixed (fp8xfp4)**: 2.2-3.0x speedup

**Factors affecting performance:**
- Matrix size (larger = better amortization)
- Scale overhead (smaller VEC_SIZE = more overhead)
- Memory vs compute bound (FP4 helps more when memory-bound)

Numerical Considerations
------------------------

Quantization Error
~~~~~~~~~~~~~~~~~~

.. code-block:: python

FP16 range: +/-65504, ~3 decimal digits
=====================================
FP8 (E4M3) range: +/-448, ~2 decimal digits
=========================================
FP4 (E2M1) range: +/-6, ~1 decimal digit
======================================
::


**Per-block scaling helps:**
.. code-block:: python

Without scaling:
================
fp4_val = quantize_fp16*to_fp4(1234.5)  # Overflow!

With scaling:
=============
scale = 1234.5 / 6.0  # ~205
fp4_val = quantize_fp16*to_fp4(1234.5 / scale)  # ~ 6
reconstructed = fp4_val * scale  # ~ 1234.5
::


E8M0 Scale Format (AMD)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

def e8m0_to*f32(x):
    """Convert 8-bit exponent-only to float32."""
    # No mantissa, only exponent
    # Value = 2^(exponent - 127)
    return 2 ** ((x - 127).to(torch.float32))

Example:
========
x = 135 -> 2^(135-127) = 2^8 = 256
=================================
x = 127 -> 2^0 = 1
=================
x = 119 -> 2^(-8) = 0.00390625
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
python 10-block-scaled-matmul.py --format nvfp4 --K_range 512 8192 --bench

NVIDIA FP8
==========
python 10-block-scaled-matmul.py --format mxfp8 --K 4096 --bench

AMD MXFP4 (automatic detection)
===============================
python 10-block-scaled-matmul.py --format mxfp4 --bench

Mixed precision
===============
python 10-block-scaled-matmul.py --format mixed --K_range 2048 8192 --K_step 2048
::


Validation
~~~~~~~~~~

.. code-block:: python

NVIDIA
======
validate_block*scaled(8192, 8192, 8192, block_scale*type="nvfp4")
[[OK]] (pass nvfp4)
==============

AMD with both MFMA shapes
=========================
validate_block*scaled_amd(8192, 8192, 8192, block_scale*type="mxfp4", mfma_nonkdim=16)
[[OK]] (pass mxfp4, mfma_nonk*dim 16)
================================

validate_block*scaled_amd(8192, 8192, 8192, block_scale*type="mxfp4", mfma_nonkdim=32)
[[OK]] (pass mxfp4, mfma_nonk*dim 32)
================================
::


Common Pitfalls
---------------

1. Unsupported Hardware
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

if not supports_block*scaling():
    print("[blocked] This example requires GPU support for block scaled matmul")
    exit(1)

def supports_block*scaling():
    return (is_cuda() and torch.cuda.get_device*capability()[0] == 10) or \
           is_hip*cdna4()
::


2. Format Mismatch
~~~~~~~~~~~~~~~~~~

.. code-block:: python

Bad: Using AMD kernel with NVIDIA formats
=========================================
if is_hip*cdna4():
    assert args.format == "mxfp4", "AMD only supports mxfp4"
::


3. Wrong Scale Shape
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Scales must match data dimensions
=================================
assert a_scale.shape == [M // 128, K // VEC_SIZE // 4, 32, 16]

After packing to 5D
===================
assert a_scale.shape == [1, M // 128, K // VEC_SIZE // 4, 2, 256]
::


4. Missing TMA Allocator
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from typing import Optional

def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)
::


Summary
-------

**Block scaled matmul** enables efficient low-precision matrix multiplication:

- **4-8x memory reduction** compared to FP16
- **2-4x compute speedup** on specialized hardware
- **Per-block scaling** maintains numerical accuracy
- **Hardware acceleration** via 5th-gen Tensor Cores (NVIDIA) and CDNA4 (AMD)

**Supported formats:**
- ``nvfp4``: NVIDIA-optimized FP4 (16 elem/scale)
- ``mxfp4``: OCP standard FP4 (32 elem/scale)
- ``mxfp8``: OCP standard FP8 (32 elem/scale)
- ``mixed``: FP8xFP4 mixed precision

**Key techniques:**
- **Scale preshuffling** for contiguous memory access
- **TMA descriptors** for hardware-accelerated loads
- **tl.dot_scaled** intrinsic for scaled operations
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
