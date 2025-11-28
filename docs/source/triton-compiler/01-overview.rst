Triton Compiler Architecture Overview
======================================

This document provides a comprehensive overview of how the Triton compiler works, from Python decorators to GPU binary code.

What is Triton?
---------------

**Triton** is a language and compiler for writing highly efficient GPU kernels using Python-like syntax. Unlike CUDA, which requires explicit memory management and complex threading, Triton provides a high-level block-based programming model that automatically handles many low-level details.

Key Features
~~~~~~~~~~~~

- **Python-based DSL** - Write GPU kernels using Python syntax
- **Automatic Memory Management** - No need to manually manage shared memory
- **Block-level Programming** - Work with blocks of data instead of individual threads
- **JIT Compilation** - Kernels are compiled at runtime for flexibility
- **MLIR-based** - Leverages modern compiler infrastructure
- **Multi-backend** - Supports NVIDIA (CUDA), AMD (ROCm), and potentially others

Compilation Pipeline Overview
------------------------------

The Triton compiler transforms Python code through several stages:

.. code-block:: text

    Python Function (@triton.jit)
           down
    Python AST (Abstract Syntax Tree)
           down
    Triton IR (TTIR) - High-level intermediate representation
           down
    Triton GPU IR (TTGIR) - GPU-specific intermediate representation
           down
    LLVM IR (LLIR) - Low-level intermediate representation
           down
    PTX / AMDGCN - GPU assembly
           down
    CUBIN / HSACO - GPU binary

Each stage performs specific transformations and optimizations.

Architecture Components
------------------------

The Triton compiler consists of several key components:

1. **Frontend (Python)**

   - ``@triton.jit`` decorator
   - AST parsing and analysis
   - Type inference
   - Dependency tracking

   *Location:* ``python/triton/runtime/jit.py``

2. **Code Generator**

   - Converts Python AST to Triton IR (TTIR)
   - Handles control flow
   - Manages value types

   *Location:* ``python/triton/compiler/code_generator.py``

3. **MLIR Pipeline**

   - TTIR -> TTGIR lowering
   - GPU-specific optimizations
   - Memory coalescing
   - Shared memory allocation

   *Location:* ``lib/Dialect/`` (C++)

4. **Backend**

   - LLVM IR generation
   - Target-specific code generation
   - Binary assembly (PTX, AMDGCN)

   *Location:* ``third_party/nvidia/backend/`` or ``third_party/amd/backend/``

5. **Runtime**

   - Kernel caching
   - Auto-tuning
   - Grid computation
   - Kernel launching

   *Location:* ``python/triton/runtime/``

Compilation Stages in Detail
-----------------------------

Stage 1: Python AST Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@triton.jit`` decorator captures the Python function and parses its source code into an Abstract Syntax Tree (AST).

.. code-block:: python

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

The AST representation captures:

- Function signature and parameters
- Control flow structures (if/for/while)
- Function calls and operations
- Constant expressions (``tl.constexpr``)

Stage 2: Code Generation (TTIR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code generator walks the Python AST and generates Triton IR (TTIR), a high-level representation that is backend-independent.

**TTIR Example:**

.. code-block:: mlir

    module {
      tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                  %arg2: !tt.ptr<f32>, %arg3: i32) {
        %c0_i32 = arith.constant 0 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %BLOCK_SIZE : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
        %3 = tt.splat %1 : i32 -> tensor<128xi32>
        %4 = arith.addi %3, %2 : tensor<128xi32>
        %5 = tt.splat %arg3 : i32 -> tensor<128xi32>
        %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
        %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %9 = tt.load %8, %6 : tensor<128x!tt.ptr<f32>>
        // ... more operations ...
        tt.return
      }
    }

Key features of TTIR:

- Uses MLIR (Multi-Level Intermediate Representation) infrastructure
- Supports tensor types (``tensor<128xf32>``)
- Pointer arithmetic (``tt.addptr``)
- Block operations (``tt.load``, ``tt.store``)

Stage 3: Triton GPU IR (TTGIR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TTIR is lowered to TTGIR, which adds GPU-specific information:

- **Thread layout** - How threads are organized within blocks
- **Data layout** - How data is distributed across threads
- **Memory hierarchy** - Shared memory allocation
- **Synchronization** - Barrier placement

**Transformations:**

- Memory coalescing optimization
- Shared memory insertion
- Warp-level operations
- Tensor core utilization (for matmul)

Stage 4: LLVM IR
~~~~~~~~~~~~~~~~

TTGIR is lowered to LLVM IR, which is closer to assembly:

.. code-block:: llvm

    define void @add_kernel(float* %arg0, float* %arg1, float* %arg2, i32 %arg3) {
    entry:
      %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
      %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
      %block_size = mul i32 %bid, 128
      %offset = add i32 %block_size, %tid
      %in_bounds = icmp slt i32 %offset, %arg3
      br i1 %in_bounds, label %load, label %exit
    load:
      %x_ptr = getelementptr float, float* %arg0, i32 %offset
      %y_ptr = getelementptr float, float* %arg1, i32 %offset
      %x_val = load float, float* %x_ptr
      %y_val = load float, float* %y_ptr
      %result = fadd float %x_val, %y_val
      %out_ptr = getelementptr float, float* %arg2, i32 %offset
      store float %result, float* %out_ptr
      br label %exit
    exit:
      ret void
    }

Stage 5: PTX / AMDGCN
~~~~~~~~~~~~~~~~~~~~~

LLVM IR is compiled to GPU assembly:

**NVIDIA PTX:**

.. code-block:: ptx

    .version 8.0
    .target sm_80
    .address_size 64

    .visible .entry add_kernel(
        .param .u64 add_kernel_param_0,
        .param .u64 add_kernel_param_1,
        .param .u64 add_kernel_param_2,
        .param .u32 add_kernel_param_3
    )
    {
        .reg .pred %p<2>;
        .reg .f32 %f<3>;
        .reg .b32 %r<5>;
        .reg .b64 %rd<7>;

        ld.param.u64 %rd1, [add_kernel_param_0];
        ld.param.u64 %rd2, [add_kernel_param_1];
        ld.param.u64 %rd3, [add_kernel_param_2];
        ld.param.u32 %r1, [add_kernel_param_3];

        mov.u32 %r2, %tid.x;
        mov.u32 %r3, %ctaid.x;
        mul.lo.s32 %r4, %r3, 128;
        add.s32 %r2, %r2, %r4;

        setp.ge.s32 %p1, %r2, %r1;
        @%p1 bra EXIT;

        cvt.s64.s32 %rd4, %r2;
        shl.b64 %rd5, %rd4, 2;
        add.s64 %rd6, %rd1, %rd5;
        ld.global.f32 %f1, [%rd6];

        // ... more assembly ...

    EXIT:
        ret;
    }

Stage 6: Binary (CUBIN / HSACO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PTX is assembled into CUBIN (CUDA binary) using ``ptxas``, or AMDGCN is assembled into HSACO (HSA Code Object) using AMD's assembler.

This binary can be executed on the GPU.

Key Design Decisions
--------------------

Block-based Programming Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Triton uses **SPMD (Single Program, Multiple Data)** with blocks:

.. code-block:: python

    # Instead of thinking about individual threads:
    # thread_id = get_thread_id()
    # output[thread_id] = input[thread_id] * 2

    # Triton thinks in blocks:
    pid = tl.program_id(0)  # Which block am I?
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Block of indices
    data = tl.load(input_ptr + offsets)  # Load entire block
    result = data * 2
    tl.store(output_ptr + offsets, result)  # Store entire block

**Benefits:**

- Easier to write and understand
- Compiler handles thread-level details
- Automatic memory coalescing
- Better optimization opportunities

JIT Compilation
~~~~~~~~~~~~~~~

Triton compiles kernels **at runtime** when first called:

.. code-block:: python

    @triton.jit
    def kernel(...):
        pass

    # First call: compilation happens here
    kernel[grid](args)  # ~100ms-1s for first compile

    # Second call: cached, very fast
    kernel[grid](args)  # ~microseconds

**Benefits:**

- Can specialize on runtime values (``constexpr``)
- No separate compilation step
- Flexible auto-tuning

**Caching:**

Compiled kernels are cached by:

- Source code hash
- Compile-time constants
- Compiler options
- Environment variables

MLIR Infrastructure
~~~~~~~~~~~~~~~~~~~

Triton uses **MLIR (Multi-Level Intermediate Representation)** instead of custom IR:

**Advantages:**

- Reusable passes and transformations
- Interoperability with other MLIR-based compilers
- Well-tested infrastructure
- Growing ecosystem

**Dialects used:**

- ``tt`` - Triton dialect (TTIR)
- ``ttg`` - Triton GPU dialect (TTGIR)
- ``arith`` - Arithmetic operations
- ``scf`` - Structured control flow
- ``llvm`` - LLVM dialect
- ``nvgpu`` - NVIDIA GPU specific
- ``rocdl`` - AMD ROCm specific

Source Code Organization
-------------------------

Python Components
~~~~~~~~~~~~~~~~~

``python/triton/``

- ``runtime/jit.py`` - ``@triton.jit`` decorator, JITFunction class
- ``compiler/compiler.py`` - Main compilation orchestration
- ``compiler/code_generator.py`` - AST -> TTIR conversion
- ``backends/compiler.py`` - Backend abstraction
- ``language/`` - Triton language primitives (tl.load, tl.store, etc.)

C++ Components
~~~~~~~~~~~~~~

``lib/Dialect/``

- ``Triton/`` - TTIR dialect definition
- ``TritonGPU/`` - TTGIR dialect and passes
- ``TritonNvidiaGPU/`` - NVIDIA-specific passes

``include/triton/``

- MLIR operation definitions (``.td`` files)
- Pass headers

Backend Components
~~~~~~~~~~~~~~~~~~

``third_party/nvidia/backend/``

- ``compiler.py`` - NVIDIA backend implementation
- ``driver.py`` - CUDA driver interface

``third_party/amd/backend/``

- ``compiler.py`` - AMD backend implementation
- ``driver.py`` - ROCm driver interface

Links to Source Code
---------------------

All links reference Triton v3.5.1:

- `JIT Decorator <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/jit.py>`_
- `Compiler <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/compiler/compiler.py>`_
- `Code Generator <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/compiler/code_generator.py>`_
- `Backend Base <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/backends/compiler.py>`_
- `NVIDIA Backend <https://github.com/triton-lang/triton/tree/v3.5.1/third_party/nvidia/backend/compiler.py>`_
- `Triton Dialect <https://github.com/triton-lang/triton/tree/v3.5.1/lib/Dialect/Triton>`_
- `TritonGPU Dialect <https://github.com/triton-lang/triton/tree/v3.5.1/lib/Dialect/TritonGPU>`_

Next Steps
----------

Continue to:

- :doc:`02-jit-decorator` - Deep dive into ``@triton.jit``
- :doc:`03-code-generation` - Python AST to TTIR conversion
- :doc:`04-mlir-lowering` - MLIR passes and optimizations
- :doc:`05-backend-compilation` - Backend-specific compilation
- :doc:`06-kernel-launch` - Runtime and kernel execution
