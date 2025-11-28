==================================
# Triton GPU Programming Concepts


This document explains the core concepts of Triton GPU programming, including kernel launching, grid specifications, program indexing, and how parallel execution works on GPUs.

.. contents:: Table of Contents
   :local:
   :depth: 3

# Introduction to Triton


Triton is a Python-based GPU programming language that allows you to write high-performance GPU kernels without needing to learn CUDA or other low-level languages. It provides a SPMD (Single Program, Multiple Data) execution model where the same kernel code runs on multiple GPU cores in parallel.

Key advantages:
- Write GPU code in Python
- Automatic performance optimization
- No manual memory management
- Abstract away hardware-specific details

## Vector Addition Example


Throughout this document, we'll use a simple vector addition kernel as an example:
``python
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

=============================

When you launch a Triton kernel, you use this syntax:
``python
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)


1. **Grid specification**: Defines how many parallel programs to launch
2. **Indexing syntax** `[grid]`: Uses Python's `__getitem__` magic method
3. **Kernel arguments**: Data passed to each program
4. **Synchronization**: Optional explicit synchronization with `torch.cuda.synchronize()`

# The `[grid]` Syntax Sugar: Python's `__getitem__`


## How It Works


The `add_kernel[grid]` syntax is **syntactic sugar** for launching a kernel with a specific grid configuration. Under the hood, it uses Python's `__getitem__` magic method.

### Implementation Details


When you write:
``python
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

``python
    add_kernel.__getitem__(grid)(x, y, output, n_elements, BLOCK_SIZE=1024)

```python
    class KernelInterface(Generic[T]):
        def __getitem__(self, grid) -> T:
            """
            A JIT function is launched with: fn[grid](*args, **kwargs).
            Hence JITFunction.__getitem__ returns a callable proxy that
            memorizes the grid.
            """
            return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)

- `__getitem__` receives the `grid` parameter
- Returns a **lambda function** that captures the grid
- The lambda calls `self.run()` with the stored grid when invoked
- `self.run()` handles actual kernel compilation and GPU launch

### The Complete Flow

```text
    1. Create kernel instance
       add_kernel = JITFunction(add_kernel_fn)

    2. Index with grid (calls __getitem__)
       callable_with_grid = add_kernel[grid]
       # Returns: lambda *args, **kwargs: add_kernel.run(grid=grid, warmup=False, *args, **kwargs)

    3. Call with arguments (invokes the lambda)
       add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
       # Calls: add_kernel.run(grid=(97,), warmup=False, x, y, output, n_elements, BLOCK_SIZE=1024)

    4. Inside run()
       - Specializes kernel based on argument types
       - Compiles if not cached
       - Extracts grid dimensions
       - Launches GPU kernel

=================================

## Purpose


`triton.cdiv(a, b)` computes the **ceiling division** of `a` by `b`:
``text
    triton.cdiv(a, b) = ceil(a / b) = (a + b - 1) // b


### Example with Vector Addition


For 98,432 elements and BLOCK_SIZE=1024:
``python
    grid = lambda meta: (triton.cdiv(98432, 1024), )
    # Evaluates to: (97,)

- **97 programs** will be launched
- Each program processes **1,024 elements**
- Last program (pid=96) processes remaining 432 elements
- Total coverage: 97 * 1024 = 99,328 > 98,432 [OK]

### Why Ceiling Division?


If you used regular division:
``python
    grid_wrong = 98432 // 1024  # = 96 programs
    # This would miss the last 432 elements!

``python
    grid_correct = triton.cdiv(98432, 1024)  # = 97 programs
    # All elements are covered

```python
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)  # Masked load prevents out-of-bounds access

=======================================

## The SPMD Model


Triton uses SPMD (Single Program, Multiple Data):
- **One program definition**: The same kernel code
- **Multiple data partitions**: Each program processes different data
- **Parallel execution**: All programs run simultaneously on different GPU cores

## Program ID Assignment


When the GPU launches a kernel with grid size `(97,)`, the GPU runtime automatically assigns each program a unique **program ID**:
``python
    # Inside the kernel
    pid = tl.program_id(axis=0)

    # pid values across all programs:
    # Program 0: pid = 0
    # Program 1: pid = 1
    # Program 2: pid = 2
    # ...
    # Program 96: pid = 96

-----------------

Each program calculates which data elements it should process:
``python
    # Line 44-45 in vector-add.py
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

~~~~~~~~~~~~~~~~~~~

For BLOCK_SIZE=1024:
``text
    Program | pid | block_start | offsets
    ======= === ========== ========================================
    0       0   0          [0, 1, 2, ..., 1023]
    1       1   1024       [1024, 1025, 1026, ..., 2047]
    2       2   2048       [2048, 2049, 2050, ..., 3071]
    96      96  98304      [98304, 98305, ..., 99327]

~~~~~~~~~~~~~~~~~~~~~

Each program loads and processes different tensor slices:
``python
    # Line 50-54
    x = tl.load(x_ptr + offsets, mask=mask)  # Load program's slice
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y                            # Element-wise addition
    tl.store(output_ptr + offsets, output, mask=mask)  # Store results

```text
    Program 0:  x[0:1024] + y[0:1024]     -> output[0:1024]
    Program 1:  x[1024:2048] + y[1024:2048] -> output[1024:2048]
    Program 2:  x[2048:3072] + y[2048:3072] -> output[2048:3072]
    ...
    Program 96: x[98304:99328] + y[98304:99328] -> output[98304:99328]

===========================

## Implementation Stack


`tl.program_id()` is implemented across multiple layers:

**1. User Code (core.py:1605)**
``python
    def program_id(axis, _semantic=None):
        return _semantic.program_id(axis)

``python
    def program_id(self, axis: int) -> TensorTy:
        if axis not in (0, 1, 2):
            raise ValueError(f"program_id axis must be 0, 1, or 2 but got {axis}")
        return self.tensor(
            self.builder.create_get_program_id(axis),  # Generate LLVM IR
            tl.int32
        )


The `create_get_program_id(axis)` method generates LLVM Intermediate Representation code that retrieves the program ID from the GPU.

**4. Hardware-Specific Code Generation**

LLVM compiles to GPU assembly:

**NVIDIA PTX (CUDA):**
``ptx
    mov.u32 %r0, %ctaid.x  // Move block ID to register

``amdgpu
    s_get_workgroup_id_x s0  // Get block ID in SGPR


At runtime, the GPU provides the program ID via special registers:
- NVIDIA: `blockIdx.x`, `blockIdx.y`, `blockIdx.z`
- AMD: Workgroup ID registers

The GPU's thread scheduler automatically assigns each block a unique ID when launching the kernel.

### Flow Diagram

``text
    Host (Python):
        grid = (97,)
        kernel[grid](...)
        |
        v
    Triton Compiler:
        tl.program_id(axis=0)
        |
        v
    LLVM IR Generation:
        builder.create_get_program_id(0)
        |
        v
    GPU Code Generation:
        PTX: mov.u32 %r0, %ctaid.x
        AMDGPU: s_get_workgroup_id_x
        |
        v
    GPU Runtime (CUDA/HIP Driver):
        Launch 97 blocks with blockIdx = 0..96
        |
        v
    GPU Hardware:
        Each block reads its blockIdx from register
        Program 0: blockIdx.x = 0
        Program 1: blockIdx.x = 1
        ...
        Program 96: blockIdx.x = 96

===========================================

## Is Kernel Launch Synchronous or Asynchronous?


**The kernel launch is ASYNCHRONOUS** by default. When you call:
``python
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)


### Manual Synchronization


You only need `torch.cuda.synchronize()` if you need to measure execution time:
``python
    # Without synchronization (asynchronous)
    start = time.time()
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    elapsed = time.time() - start  # Measures launch overhead only, not execution

    # With synchronization (measures actual execution)
    start = time.time()
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    torch.cuda.synchronize()  # Wait for GPU to finish
    elapsed = time.time() - start  # Measures actual kernel execution

~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch automatically synchronizes when you use the tensor:
``python
    output_triton = add(x, y)  # Kernel launches asynchronously
    print(output_triton)       # Synchronizes before printing

    # Or when moving to CPU
    output_cpu = output_triton.cpu()  # Synchronizes before copying

``python
    output_torch = x + y                      # CPU-side, returns immediately
    output_triton = add(x, y)                 # GPU kernel, launches asynchronously
    print(output_torch)                       # Synchronizes implicitly
    print(output_triton)                      # Synchronizes implicitly
    print(f'Max difference: {torch.max(torch.abs(output_torch - output_triton))}')
    # By here, both are ready and comparison works

~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This design allows:
1. **Overlapping computation**: CPU can queue multiple kernels while GPU executes
2. **Better resource utilization**: No blocking waits
3. **Higher throughput**: Multiple kernels can run concurrently on different GPU streams

# Advanced Topics


## Multi-Dimensional Grids


For 2D operations, you can use multi-dimensional grids:
``python
    @triton.jit
    def matrix_op_kernel(ptr, M: tl.constexpr, N: tl.constexpr):
        pid_m = tl.program_id(axis=0)  # Row block ID
        pid_n = tl.program_id(axis=1)  # Column block ID
        # ... process matrix block (pid_m, pid_n)

    # Launch 2D grid
    grid = lambda meta: (triton.cdiv(M, 32), triton.cdiv(N, 32))

--------------------

The `tl.constexpr` annotation marks parameters as compile-time constants:
``python
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # BLOCK_SIZE is a compile-time constant
        # Triton specializes the kernel for each unique BLOCK_SIZE value
        offsets = tl.arange(0, BLOCK_SIZE)  # Unrolled at compile time

- Unroll loops
- Optimize memory access patterns
- Generate specialized code per BLOCK_SIZE

## Kernel Caching


Triton caches compiled kernels:
``python
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)  # Compiles
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)  # Cached!

- Kernel signature
- Argument specialization
- Compilation options

# Performance Considerations


## Grid Size Selection


Choose BLOCK_SIZE based on GPU architecture:
``python
    # Good choices
    BLOCK_SIZE = 256   # Common for NVIDIA
    BLOCK_SIZE = 512
    BLOCK_SIZE = 1024

    # Compute grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)

----------------

Always mask out-of-bounds accesses:
``python
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

- Segmentation faults
- Invalid memory access
- Undefined behavior

## Memory Coalescing


Triton optimizes memory access patterns automatically, but contiguous access is still important:
``python
    # Good: contiguous memory access
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=mask)

    # Bad: strided memory access
    offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * stride
    x = tl.load(x_ptr + offsets, mask=mask)

===================================================

When optimizing kernels for specific GPU architectures, you often need to:
1. Compile the kernel ahead of time
2. Extract resource usage information
3. Calculate optimal grid size based on GPU properties
4. Load the binary on the GPU

This section explains the functions that enable this workflow.

## is_hip() - Detect AMD GPU Backend


**Purpose**: Check if the kernel is running on an AMD GPU (HIP backend) vs NVIDIA (CUDA).

**Definition**:
``python
    def is_hip():
        return triton.runtime.driver.active.get_current_target().backend == "hip"

- Returns `True` if backend is HIP (AMD's Heterogeneous-Interface for Portability)
- Returns `False` if backend is CUDA (NVIDIA)

**Why it matters:**

AMD and NVIDIA GPUs have fundamentally different architectures:
``text
    NVIDIA Architecture          AMD RDNA/CDNA Architecture
    ├── CUDA cores              ├── Stream Processors (SPs)
    ├── Warps (32 threads)      ├── Waves (64 threads)
    ├── Registers: ~256 per warp ├── VGPRs: 256 per wave
    └── Occupancy formula       └── Different occupancy formula
       based on register usage      (includes register pools)

``python
    if is_hip():
        # AMD-specific calculation
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        # NVIDIA-specific calculation
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

-----------------------------------------

**Purpose**: Check if the AMD GPU is CDNA architecture (data center) vs RDNA (gaming).

**Definition**:
``python
    def is_cdna():
        return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
            'gfx940', 'gfx941', 'gfx942',  # CDNA 3 (MI300)
            'gfx90a', 'gfx908'              # CDNA 1-2 (MI200/MI100)
        )

``text
    AMD GPUs
    ├── RDNA (Gaming/Consumer)
    │   ├── RX 6800, 6900, 7000 series
    │   └── Single register pool (256 VGPRs per wave)
    │
    └── CDNA (Data Center)
        ├── MI100 (CDNA 1)
        │   └── arch: gfx908
        ├── MI200 (CDNA 2)
        │   └── arch: gfx90a
        └── MI300 (CDNA 3)
            ├── arch: gfx940, gfx941, gfx942
            └── Dual register pools: 512 total VGPRs


CDNA architecture has a unique register organization for matrix operations:
``text
    CDNA Register Layout
    ├── Regular VGPRs: 256 registers per wave
    ├── Accumulation VGPRs: 256 registers per wave
    └── Total: 512 registers available per wave


**Example usage from fused softmax:**
``python
    if is_hip():
        NUM_GPRS = NUM_REGS  # Start with regular registers
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2  # CDNA can use 2x registers
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        # NVIDIA doesn't have dual pools
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

------------------------------------------------

**Purpose**: Compile the kernel and extract resource metadata (registers, shared memory) without actually running it on the GPU.

**Signature**:
``python
    def warmup(self, *args, grid, **kwargs):
        return self.run(grid=grid, warmup=True, *map(MockTensor.wrap_dtype, args), **kwargs)


1. **Compiles** the kernel using the Triton compiler pipeline
   - Python source code → LLVM IR → GPU assembly (PTX/AMDGPU)
2. **Analyzes** the compiled binary to extract resource usage
3. **Returns** a kernel object with metadata properties
4. **Does NOT execute** on the GPU (warmup=True flag)

**Resource information extracted:**
``python
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0),
                                   n_rows, n_cols,
                                   BLOCK_SIZE=256,
                                   num_stages=4,
                                   num_warps=8,
                                   grid=(1,))

    # After warmup, these properties are available:
    n_regs = kernel.n_regs                    # Registers per thread
    size_smem = kernel.metadata.shared        # Shared memory in bytes
    n_spills = kernel.n_spills                # Spilled registers to memory
    n_max_threads = kernel.n_max_threads      # Max concurrent threads


Occupancy calculations require **actual compiled kernel properties**, not just source code:
``text
    Without warmup:
        Source code → ?? registers needed ??
        Can't calculate occupancy accurately

    With warmup:
        Source code → Compiler → Analyze binary
        |
        ├── n_regs = 64 (per thread)
        ├── metadata.shared = 2048 bytes
        └── Can now calculate:
            occupancy = NUM_REGS / (64 * WARP_SIZE * num_warps)
            grid_size = NUM_SM * occupancy


The warmup function respects constexpr specialization:
``python
    # Each unique constexpr combination requires separate warmup
    kernel_256 = softmax_kernel.warmup(..., BLOCK_SIZE=256, num_stages=4, ...)
    # kernel_256.n_regs = 32 (optimized for 256 elements)

    kernel_512 = softmax_kernel.warmup(..., BLOCK_SIZE=512, num_stages=4, ...)
    # kernel_512.n_regs = 64 (needs more registers for 512 elements)

    # Different specializations have different resource usage!

-----------------------------------------------

**Purpose**: Load the compiled kernel binary on the GPU and initialize runtime handles.

**Signature**:
``python
    def _init_handles(self):
        if self.module is not None:
            return  # Already initialized

        device = driver.active.get_current_device()
        self._run = driver.active.launcher_cls(self.src, self.metadata)

        # Validate shared memory
        max_shared = max_shared_mem(device)
        if self.metadata.shared > max_shared:
            raise OutOfResources(...)

        # Load binary and extract register info from the loaded module
        self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = \
            driver.active.utils.load_binary(self.name, self.kernel,
                                            self.metadata.shared, device)

        # Validate thread resources
        if self.metadata.num_warps * warp_size > self.n_max_threads:
            raise OutOfResources(...)


1. **Loads** the binary on the current GPU device
2. **Validates** that kernel resources fit within GPU limits
3. **Extracts** register and thread information from the loaded module
4. **Initializes** GPU-specific launcher and function pointers
5. **Raises errors** if resources exceed GPU capabilities

**Resource validation:**
``python
    # Check 1: Shared memory
    if kernel.metadata.shared > max_shared_mem:
        raise OutOfResources("shared memory", actual, limit)

    # Check 2: Tensor memory (Blackwell)
    if kernel.metadata.tmem_size > 512:
        raise OutOfResources("tensor memory", actual, limit)

    # Check 3: Thread count
    max_threads = num_warps * warp_size
    if max_threads > max_threads_per_sm:
        raise OutOfResources("threads", actual, limit)


`_init_handles()` is called lazily (on demand) for several reasons:
``python
    # Lazy initialization pattern:
    @property
    def run(self):
        if self._run is None:
            self._init_handles()  # Only when first accessed
        return self._run

    # Benefits:
    # 1. Speed: Don't load binaries until actually needed
    # 2. Device switching: Can switch GPU devices before first run
    # 3. Memory efficiency: Defer loading expensive binaries


You explicitly call `_init_handles()` when you need resource information **before** launching:
``python
    # Warmup: compile and get basic metadata
    kernel = softmax_kernel.warmup(..., BLOCK_SIZE=256, num_stages=4, ...)

    # Initialize GPU handles to confirm resource usage
    kernel._init_handles()

    # Now safely access n_regs (confirmed from loaded binary)
    n_regs = kernel.n_regs

    # Calculate occupancy with verified register count
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

    # Calculate grid size
    num_programs = NUM_SM * occupancy

    # Launch with optimized grid
    kernel[(num_programs, 1, 1)](y, x, ...)

-----------------------------------------

Here's how all these functions work together in the fused softmax kernel:
``python
    def softmax(x):
        n_rows, n_cols = x.shape

        # 1. Calculate basic parameters
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 8
        num_stages = 4

        # 2. WARMUP: Compile kernel and get resource usage
        kernel = softmax_kernel.warmup(
            y, x, x.stride(0), y.stride(0), n_rows, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,)  # Dummy grid for warmup
        )

        # 3. INIT_HANDLES: Load binary on GPU and validate
        kernel._init_handles()

        # 4. EXTRACT METADATA
        n_regs = kernel.n_regs          # Now confirmed from loaded binary
        size_smem = kernel.metadata.shared

        # 5. DETECT GPU ARCHITECTURE
        if is_hip():
            NUM_GPRS = NUM_REGS
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2  # CDNA has dual register pools

            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs,
                           max_num_waves) // num_warps
        else:
            # NVIDIA GPU
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

        # 6. APPLY CONSTRAINTS
        occupancy = min(occupancy, SIZE_SMEM // size_smem)

        # 7. CALCULATE GRID
        num_programs = NUM_SM * occupancy
        num_programs = min(num_programs, n_rows)

        # 8. LAUNCH with optimized grid
        kernel[(num_programs, 1, 1)](
            y, x, x.stride(0), y.stride(0), n_rows, n_cols,
            BLOCK_SIZE, num_stages
        )
        return y

~~~~~~~~~~~~~~~~~
``text
    Source Code (02-fused-softmax.py)
    └─ softmax_kernel (JIT function)
       │
       └─ warmup()  [Step 2]
          ├─ Compile: Python → LLVM → GPU Assembly
          ├─ Extract: n_regs = 64, shared = 2048
          └─ Return: kernel object

          kernel._init_handles()  [Step 3]
          ├─ Load binary on GPU
          ├─ Validate resources
          ├─ Confirm: n_regs = 64 ✓

          is_hip()  [Step 5a]
          └─ Check: backend == "hip" → True

          is_cdna()  [Step 5b]
          └─ Check: arch in CDNA list → False

          Occupancy Calculation  [Step 5-7]
          ├─ occupancy = NUM_REGS // (64 * WARP_SIZE * 8)
          ├─ occupancy = 65536 // (64 * 32 * 8) = 4
          └─ num_programs = 160 SM * 4 = 640

          Kernel Launch  [Step 8]
          └─ kernel[(min(640, 1823), 1, 1)](...)
             └─ Launch 1823 blocks (one per row)

~~~~~~~~~~~~~
```text
    Function        | Type    | Purpose                        | Returns
    =============== ========= ================================ ==========================
    is_hip()        | Check   | Detect AMD GPU backend         | True/False
    is_cdna()       | Check   | Detect AMD CDNA arch           | True/False
    warmup()        | Compile | Pre-compile + extract metadata | Kernel object with n_regs, shared
    _init_handles() | Init    | Load binary on GPU + validate  | (Initializes internal state)

===========================================

When writing high-performance GPU kernels, you often know certain conditions will always be true at runtime. Rather than letting the compiler generate defensive code to handle all cases, you can use `tl.assume()` to tell the compiler these guarantees, enabling aggressive optimizations.

## What is tl.assume()?


**Purpose**: Provide compile-time hints to the Triton compiler that a condition is guaranteed to be true.

**Definition**:
```python
    def assume(cond, _semantic=None):
        '''
        Allow compiler to assume the :code:`cond` is True.
        '''
        return _semantic.assume(_semantic.to_tensor(cond))


- Tells the compiler: "This condition will always be true at runtime"
- Allows the compiler to:
  1. Eliminate redundant bounds checks
  2. Simplify pointer arithmetic
  3. Remove impossible code branches
  4. Generate smaller, faster code
  5. Enable more aggressive optimizations

**Important caveat**: If your assumption is false, the compiler generates incorrect code and you get undefined behavior!

## Why Assumptions Help Optimization


Consider address calculation in matrix multiplication:
```python
    # Without assumption:
    a_ptr = base_ptr + (offset * stride)
    # Compiler must check:
    # - Is stride positive or negative?
    # - Could offset * stride overflow?
    # - Is pointer within valid bounds?
    # Generated code: ~5-7 instructions

    # With assumption:
    tl.assume(stride > 0)
    a_ptr = base_ptr + (offset * stride)
    # Compiler knows stride is positive:
    # - offset * stride always non-negative
    # - No overflow for positive values
    # - Can simplify calculations
    # Generated code: ~2-3 instructions


## Matrix Multiplication Example


In `03-matrix-multiplication.py` (lines 267-277), the kernel makes the following assumptions:
``python
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)


**Program IDs are non-negative:**
``python
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)


**Strides are positive:**
``python
    tl.assume(stride_am > 0)  # stride from tensor.stride(0)
    tl.assume(stride_ak > 0)  # stride from tensor.stride(1)
    # ... and so on

- Skip overflow checks on stride multiplication
- Avoid handling backward (negative stride) cases
- Simplify bounds validation
- Generate optimal addressing code

**Real-world impact:**

After these assumptions, the kernel calculates:
``python
    # Pointer arithmetic for matrix blocks
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # With positive stride assumptions, these simplify significantly:
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)


## When to Use `tl.assume()`


Use `tl.assume()` when you know a condition is **guaranteed** because:

1. **GPU runtime guarantees it**
``python
    pid = tl.program_id(axis=0)
    tl.assume(pid >= 0)  # Always true from GPU runtime

``python
    stride = tensor.stride(0)
    tl.assume(stride > 0)  # True for contiguous tensors

``python
    num_blocks = tl.cdiv(n, block_size)
    tl.assume(num_blocks >= 1)  # cdiv always returns >= 1

``python
    # In Python wrapper:
    assert stride > 0, "Tensor must be contiguous"

    # In kernel:
    tl.assume(stride > 0)  # Safe because Python layer checked it

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```python
    # WRONG - Makes false assumption
    stride = tensor.stride(0)  # Could be negative!
    tl.assume(stride > 0)  # DANGER: False assumption!

    # Compiler generates code assuming stride > 0
    # But if stride is actually negative, UNDEFINED BEHAVIOR!
    # Results in memory corruption, crashes, wrong answers

---------------------------------------

`tl.assume()` is similar to `tl.static_assert()` but with important differences:
``text
    Feature              | tl.assume()              | tl.static_assert()
    ==================== ========================= ======================
    When evaluated       | Compile time (hint)      | Compile time (check)
    If condition false   | Undefined behavior       | Compilation error
    Purpose              | Optimization hint        | Safety validation
    Use case             | Known truths             | Verify preconditions
    Performance impact   | Enables optimizations    | None (fails to compile)

``python
    # Use static_assert to verify kernel preconditions
    @triton.jit
    def kernel(x_ptr, stride, BLOCK_SIZE: tl.constexpr):
        # Check: Is BLOCK_SIZE valid at compile time?
        tl.static_assert(BLOCK_SIZE >= 1)  # Error if BLOCK_SIZE < 1

        # Assume: stride will always be positive at runtime
        tl.assume(stride > 0)  # Hint for optimization

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `tl.assume()` calls in matrix multiplication are critical for performance:

Without assumptions:
    - Compiler generates defensive code
    - Includes bounds checks and overflow validation
    - Handles negative strides and offsets
    - ~50% more instructions in address calculation loop
    - Lower throughput: ~200 TFLOPS

With assumptions:
    - Compiler generates optimal code
    - Eliminates impossible cases
    - Simplifies stride multiplication
    - ~33% fewer instructions
    - Higher throughput: ~240-260 TFLOPS

**This 20-30% performance difference comes purely from compiler optimization enabled by assumptions!**

## How Compiler Uses Assumptions


The Triton compiler's integer analysis backend uses `tl.assume()` to:

1. **Range analysis**: Determine possible value ranges
``python
    tl.assume(pid_m >= 0)
    # Compiler knows: pid_m ∈ [0, ∞)
    # Can eliminate negative checks

``python
    tl.assume(stride > 0)
    tl.assume(offset < MAX_INT)
    # Compiler knows: stride * offset won't overflow
    # Can skip overflow checks

``python
    tl.assume(ptr >= base_ptr)
    tl.assume(ptr < end_ptr)
    # Compiler knows: ptr is always in bounds
    # Can eliminate bounds checking

``python
    tl.assume(condition)
    if condition:
        # Path taken
    else:
        # Dead code - compiler eliminates it
        pass

---------------------------------

**Key takeaways:**

- **Purpose**: Optimization hint telling compiler about guaranteed conditions
- **Use when**: You know a condition is absolutely true
- **Benefits**: 2-3x fewer instructions for pointer arithmetic
- **Risk**: False assumptions cause undefined behavior
- **Best practice**: Only assume things guaranteed by runtime/library/preconditions
- **Performance impact**: 10-30% improvement in pointer-heavy kernels

### Comparison with Other Optimization Techniques

```text
    Technique           | Purpose                | When to Use
    =================== ======================== =========================
    tl.assume()         | Compiler optimization  | Known guaranteed conditions
    tl.static_assert()  | Validation/safety      | Verify kernel preconditions
    tl.static_print()   | Debugging              | Print compile-time values
    Constexpr params    | Specialization         | Compile-time constants
    Unrolling hints     | Loop optimization      | Control loop unrolling

=======

Key Concepts Recap:

1. **Grid Specification**: `triton.cdiv()` calculates how many programs to launch
2. **Syntax Sugar**: `kernel[grid]()` uses Python's `__getitem__` to create a callable proxy
3. **Program Identification**: `tl.program_id()` retrieves the program's unique ID from GPU hardware
4. **Data Partitioning**: Each program uses `pid * BLOCK_SIZE` to partition data
5. **SPMD Model**: All programs run the same code on different data concurrently
6. **Asynchronous Execution**: Kernels launch asynchronously; use `torch.cuda.synchronize()` or PyTorch operations to wait
7. **GPU Hardware**: `program_id()` maps to hardware block indices (blockIdx in CUDA)

The beauty of Triton is that it handles all the low-level details (LLVM IR generation, memory optimization, GPU-specific code generation) automatically, letting you write high-performance GPU code in Python.

# References


- Triton Documentation: https://triton-lang.org/
- NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Vector Addition Example: `/triton_cuda/triton_practice/01-vector-add.py`
- Triton Runtime JIT: `/triton/python/triton/runtime/jit.py`
- Triton Language Semantic: `/triton/python/triton/language/semantic.py`
