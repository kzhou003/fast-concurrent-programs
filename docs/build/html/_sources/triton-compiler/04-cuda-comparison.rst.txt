Triton vs CUDA C++: Compilation Comparison

This document explains how Triton's compilation pipeline relates to traditional CUDA C++ compilation, showing where they differ and where they converge.

Overview: Two Paths, Same Destination

Both Triton and CUDA C++ ultimately produce the same binary artifacts that run on NVIDIA GPUs:

- **PTX (Parallel Thread Execution)** - GPU assembly language
- **CUBIN (CUDA Binary)** - GPU machine code

However, they take **completely different paths** to get there:

.. code-block:: text

    CUDA C++ Path:
    .cu file (C++ with CUDA extensions)
           down
    [nvcc frontend] - Parse C++ and CUDA syntax
           down
    CUDA C++ IR (NVIDIA proprietary)
           down
    [cicc compiler] - NVIDIA's internal compiler
           down
    PTX assembly
           down
    [ptxas assembler]
           down
    CUBIN binary

    Triton Path:
    Python function (@triton.jit)
           down
    Python AST (standard Python parser)
           down
    Triton IR (TTIR) - MLIR-based
           down
    Triton GPU IR (TTGIR) - MLIR-based
           down
    LLVM IR - Standard LLVM
           down
    [LLVM NVPTX backend]
           down
    PTX assembly
           down
    [ptxas assembler] <- Same tool as CUDA!
           down
    CUBIN binary

**Key insight:** Triton and CUDA C++ converge at PTX. From PTX onward, they use the **same NVIDIA tools** (ptxas).

Traditional CUDA C++ Compilation

The NVCC Compiler
~~~~~~~~~~~~~~~~~

``nvcc`` is NVIDIA's proprietary compiler for CUDA C++.

**Example CUDA C++ kernel:**

.. code-block:: cuda

    // add.cu
    __global__ void add_kernel(float* x, float* y, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = x[idx] + y[idx];
        }
    }

**Compilation command:**

.. code-block:: bash

    nvcc -arch=sm_80 add.cu -o add.o

NVCC Compilation Stages
~~~~~~~~~~~~~~~~~~~~~~~~

``nvcc`` is actually a **driver** that orchestrates multiple tools:

1. **Preprocessing** - Handle ``#include``, ``#define``, etc.

   .. code-block:: bash

       # Expand macros and includes
       nvcc -E add.cu > add.i

2. **Frontend (cudafe++)** - Parse CUDA C++ syntax

   - Separates device code (``__global__``, ``__device__``) from host code
   - Generates CUDA C++ IR (proprietary format)

3. **Device compiler (cicc)** - NVIDIA's internal compiler

   - Optimizes device code
   - Generates PTX assembly

4. **PTX assembler (ptxas)** - Assembles PTX to CUBIN

   .. code-block:: bash

       ptxas -arch=sm_80 kernel.ptx -o kernel.cubin

5. **Host compiler (g++/cl)** - Compiles CPU code

   - Links with CUDA runtime library (``cudart``)

**Full pipeline visualization:**

.. code-block:: text

    add.cu
      down
    [nvcc driver]
            down
          host.o
            down
    [linker] Combine host + device
            down
        add.exe/add.out

PTX Assembly Output
~~~~~~~~~~~~~~~~~~~

For our CUDA C++ kernel, ``nvcc`` generates PTX like this:

.. code-block:: ptx

    .version 8.0
    .target sm_80
    .address_size 64

    .visible .entry add_kernel(
        .param .u64 add_kernel_param_0,  // float* x
        .param .u64 add_kernel_param_1,  // float* y
        .param .u64 add_kernel_param_2,  // float* out
        .param .u32 add_kernel_param_3   // int n
    )
    {
        .reg .pred %p<2>;
        .reg .f32 %f<3>;
        .reg .b32 %r<5>;
        .reg .b64 %rd<7>;

        // Load parameters
        ld.param.u64 %rd1, [add_kernel_param_0];
        ld.param.u64 %rd2, [add_kernel_param_1];
        ld.param.u64 %rd3, [add_kernel_param_2];
        ld.param.u32 %r1, [add_kernel_param_3];

        // Calculate thread index
        mov.u32 %r2, %tid.x;
        mov.u32 %r3, %ctaid.x;
        mov.u32 %r4, %ntid.x;
        mad.lo.s32 %r2, %r3, %r4, %r2;  // idx = blockIdx.x * blockDim.x + threadIdx.x

        // Bounds check
        setp.ge.s32 %p1, %r2, %r1;
        @%p1 bra EXIT;

        // Pointer arithmetic and load
        mul.wide.s32 %rd4, %r2, 4;
        add.s64 %rd5, %rd1, %rd4;
        ld.global.f32 %f1, [%rd5];  // x[idx]

        add.s64 %rd6, %rd2, %rd4;
        ld.global.f32 %f2, [%rd6];  // y[idx]

        // Addition
        add.f32 %f3, %f1, %f2;

        // Store result
        add.s64 %rd7, %rd3, %rd4;
        st.global.f32 [%rd7], %f3;  // out[idx] = x[idx] + y[idx]

    EXIT:
        ret;
    }

Triton Compilation Revisited

The LLVM Path
~~~~~~~~~~~~~

Triton uses **open-source LLVM** instead of NVIDIA's proprietary compiler.

**Equivalent Triton kernel:**

.. code-block:: python

    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

**Compilation stages:**

1. **Python AST parsing** - Standard Python ``ast.parse()``
2. **Code generation** - Python AST -> Triton IR (MLIR)
3. **GPU lowering** - TTIR -> TTGIR (add layouts, shared memory)
4. **LLVM lowering** - TTGIR -> LLVM IR
5. **LLVM backend** - LLVM IR -> PTX (using NVPTX backend)
6. **PTX assembler** - PTX -> CUBIN (using **same ptxas** as CUDA!)

LLVM IR Stage
~~~~~~~~~~~~~

Before generating PTX, Triton produces LLVM IR:

.. code-block:: llvm

    ; LLVM IR for Triton add_kernel (simplified)
    define void @add_kernel(float* %x_ptr, float* %y_ptr, float* %out_ptr, i32 %n) {
    entry:
      ; Get thread/block IDs using NVVM intrinsics
      %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
      %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

      ; Calculate offset
      %block_start = mul i32 %bid, 128
      %offset = add i32 %block_start, %tid

      ; Bounds check
      %in_bounds = icmp slt i32 %offset, %n
      br i1 %in_bounds, label %load, label %exit

    load:
      ; Pointer arithmetic
      %x_gep = getelementptr float, float* %x_ptr, i32 %offset
      %y_gep = getelementptr float, float* %y_ptr, i32 %offset

      ; Load values
      %x_val = load float, float* %x_gep, align 4
      %y_val = load float, float* %y_gep, align 4

      ; Compute
      %sum = fadd float %x_val, %y_val

      ; Store
      %out_gep = getelementptr float, float* %out_ptr, i32 %offset
      store float %sum, float* %out_gep, align 4
      br label %exit

    exit:
      ret void
    }

    ; NVVM intrinsics for GPU built-ins
    declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

**LLVM NVPTX Backend** - Converts LLVM IR to PTX

The NVPTX backend is part of open-source LLVM. It knows how to:

- Translate LLVM instructions to PTX instructions
- Map NVVM intrinsics (``llvm.nvvm.*``) to PTX special registers
- Generate PTX directives (``.version``, ``.target``, etc.)

PTX Generated by Triton
~~~~~~~~~~~~~~~~~~~~~~~~

Triton's LLVM backend generates PTX that looks **very similar** to CUDA C++:

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

        // Nearly identical to CUDA C++ PTX!
        ld.param.u64 %rd1, [add_kernel_param_0];
        ld.param.u64 %rd2, [add_kernel_param_1];
        ld.param.u64 %rd3, [add_kernel_param_2];
        ld.param.u32 %r1, [add_kernel_param_3];

        mov.u32 %r2, %tid.x;
        mov.u32 %r3, %ctaid.x;
        // ... rest is similar ...
    }

Convergence Point: PTX and CUBIN

Same Tools, Same Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**PTX Assembler (ptxas)**

Both CUDA C++ and Triton use **the same tool**:

.. code-block:: bash

    # CUDA C++ (called by nvcc)
    ptxas -arch=sm_80 cuda_kernel.ptx -o cuda_kernel.cubin

    # Triton (called by Triton compiler)
    ptxas -arch=sm_80 triton_kernel.ptx -o triton_kernel.cubin

``ptxas`` is NVIDIA's proprietary assembler, distributed with CUDA Toolkit.

**CUBIN Binary Format**

The output ``.cubin`` files have the **identical format** regardless of source:

- ELF binary format
- GPU machine code for specific architecture (e.g., sm_80)
- Metadata (register usage, shared memory, etc.)
- Relocatable or executable sections

**Binary Equivalence:**

.. code-block:: bash

    # Inspect CUBIN from CUDA C++
    cuobjdump -sass cuda_kernel.cubin

    # Inspect CUBIN from Triton
    cuobjdump -sass triton_kernel.cubin

Both show the same SASS (low-level GPU assembly).

Key Differences

Source Language
~~~~~~~~~~~~~~~

Aspect              CUDA C++                  Triton
Language            C++ with extensions       Python DSL
Syntax              ``__global__``, etc.      ``@triton.jit``
Type System         C++ static types          Python + inference
Memory Mgmt         Manual (``__shared__``)   Automatic
Threading Model     SIMT (per-thread)         Block-level SPMD

Compiler Stack
~~~~~~~~~~~~~~

Stage               CUDA C++                  Triton
Frontend            cudafe++ (proprietary)    Python AST (open)
Mid-level IR        CUDA IR (proprietary)     MLIR (open)
Low-level IR        NVIDIA internal           LLVM IR (open)
Backend             cicc (proprietary)        LLVM NVPTX (open)
Assembler           ptxas (NVIDIA)            ptxas (NVIDIA)

**Open vs Proprietary:**

- CUDA C++: Most of the stack is **closed-source**
- Triton: Everything up to PTX is **open-source** (except ptxas)

Compilation Time
~~~~~~~~~~~~~~~~

.. code-block:: text

    CUDA C++ (nvcc):
    - First compile: 1-5 seconds (C++ parsing overhead)
    - Incremental: Fast with proper build system
    - JIT (NVRTC): 100-500ms runtime compilation

    Triton:
    -------
    - First compile: 250-1000ms (MLIR + LLVM overhead)
    - Cached: < 1ms (hash-based cache)
    - Always JIT: No separate compilation step

Optimization Levels
~~~~~~~~~~~~~~~~~~~

**CUDA C++ (nvcc):**

.. code-block:: bash

    nvcc -O0  # No optimization
    nvcc -O1  # Basic optimization
    nvcc -O2  # Default
    nvcc -O3  # Aggressive (default for device code)

**Triton:**

Triton doesn't expose optimization levels. Instead:

- MLIR passes always run (coalescing, layout optimization)
- LLVM optimization level is fixed (usually -O3)
- User controls performance through kernel design (block size, etc.)

Interoperability

Can Triton and CUDA C++ Work Together?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Yes!** Because they produce the same artifacts (CUBIN), you can:

1. **Mix kernels in the same application**

   .. code-block:: python

       # PyTorch example
       import torch
       from torch.utils.cpp_extension import load

       # Load CUDA C++ kernel
       cuda_kernel = load(name="cuda_add", sources=["add.cu"])

       # Use Triton kernel
       import triton

       @triton.jit
       def triton_add(...):
           pass

       # Use both in the same program!
       cuda_kernel.add(x, y, out)
       triton_add[grid](x, y, out)

2. **Link compiled objects**

   Triton kernels compile to ``.cubin`` files that can be loaded by CUDA runtime:

   .. code-block:: cpp

       // C++ code loading Triton-compiled kernel
       CUmodule module;
       cuModuleLoad(&module, "triton_kernel.cubin");

       CUfunction kernel;
       cuModuleGetFunction(&kernel, module, "add_kernel");

       // Launch Triton kernel from C++!
       cuLaunchKernel(kernel, grid_x, grid_y, grid_z, ...);

3. **Share memory between kernels**

   Both use CUDA unified memory or device pointers:

   .. code-block:: python

       # Allocate with PyTorch
       x = torch.randn(1024, device='cuda')

       # Use with CUDA C++ kernel
       cuda_kernel.process(x)

       # Use with Triton kernel
       triton_kernel[grid](x, ...)

Why Triton Chose This Path

Advantages of LLVM Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Open Source** - Entire stack (except ptxas) is open and modifiable
2. **Portable** - LLVM supports AMD (AMDGCN), Intel, ARM GPUs
3. **Modern** - MLIR provides better optimization infrastructure
4. **Ecosystem** - Reuse LLVM tools (opt, llc, llvm-dis)
5. **Research-Friendly** - Easy to experiment with new passes

Why Not Use NVCC?
~~~~~~~~~~~~~~~~~

- NVCC is **closed-source** - Can't modify internals
- NVCC is **NVIDIA-only** - Can't target AMD, Intel
- NVCC requires **C++ parsing** - Complex language frontend
- NVCC is **hard to extend** - Adding new IR passes is difficult

Trade-offs
~~~~~~~~~~

**Advantages of Triton's approach:**

[[OK]] Full control over compilation pipeline
[[OK]] Easy to add new optimizations (MLIR passes)
[[OK]] Multi-vendor GPU support (NVIDIA, AMD)
[[OK]] Python-friendly (no C++ build complexity)
[[OK]] Reproducible builds (open toolchain)

**Disadvantages:**

[[FAIL]] Dependency on LLVM version
[[FAIL]] Can't use some NVIDIA-specific optimizations (in cicc)
[[FAIL]] Still depends on proprietary ptxas
[[FAIL]] Compilation overhead from LLVM

Compilation Artifacts Comparison

File Types
~~~~~~~~~~

Artifact            CUDA C++             Triton                 Shared?
Source              ``.cu``              ``.py``                No
IR (high-level)     CUDA IR              TTIR/TTGIR (MLIR)      No
IR (low-level)      NVIDIA internal      LLVM IR                No
Assembly            ``.ptx``             ``.ptx``               **Yes**
Binary              ``.cubin``           ``.cubin``             **Yes**
Fat binary          ``.fatbin``          N/A                    No
Object              ``.o``               N/A                    No
Executable          ``.exe/.out``        N/A (Python runtime)   No

Example Directory Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CUDA C++ project:**

.. code-block:: text

    cuda_project/

**Triton project:**

.. code-block:: text

    triton_project/

**Notice:** Triton caches **all intermediate representations**, while CUDA C++ only keeps final artifacts by default.

Inspecting Compilation Artifacts

PTX Inspection
~~~~~~~~~~~~~~

**From CUDA C++:**

.. code-block:: bash

    # Generate PTX
    nvcc -ptx -arch=sm_80 add.cu -o add.ptx

    # View PTX
    cat add.ptx

**From Triton:**

.. code-block:: bash

    # Find cached PTX
    find ~/.triton/cache -name "*.ptx" | head -1

    # View PTX
    cat ~/.triton/cache/7a3f2e1b.../add_kernel.ptx

CUBIN Inspection
~~~~~~~~~~~~~~~~

**Disassemble CUBIN to SASS:**

.. code-block:: bash

    # Works for both CUDA and Triton!
    cuobjdump -sass kernel.cubin

**View metadata:**

.. code-block:: bash

    cuobjdump -elf kernel.cubin

**Compare binaries:**

.. code-block:: bash

    # Generate from both sources
    nvcc -cubin add.cu -o cuda.cubin
    # (Triton generates triton.cubin)

    # Compare SASS
    diff <(cuobjdump -sass cuda.cubin) <(cuobjdump -sass triton.cubin)

Often, the SASS is **nearly identical** for simple kernels!

Summary
-------

Compilation Paths Compared
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    CUDA C++:  .cu -> [nvcc] -> CUDA IR -> [cicc] -> PTX -> [ptxas] -> CUBIN
                                up                         up           up
                           Proprietary              Proprietary   Shared

    Triton:    .py -> [AST] -> TTIR -> TTGIR -> LLVM IR -> [NVPTX] -> PTX -> [ptxas] -> CUBIN
                      up       up       up        up         up        up       up        up
                    Open    Open    Open     Open      Open    Shared  Propr.   Shared

**Convergence:** Both paths produce **identical PTX and CUBIN formats**.

Key Takeaways
~~~~~~~~~~~~~

1. **Same destination, different routes**

   - CUDA C++: Proprietary NVIDIA toolchain
   - Triton: Open-source LLVM toolchain

2. **Binary compatibility**

   - Both produce standard PTX and CUBIN
   - Can mix kernels from both sources
   - Use same CUDA runtime APIs

3. **Trade-offs**

   - CUDA C++: Mature, heavily optimized, NVIDIA-only
   - Triton: Flexible, portable, easier to extend

4. **Shared infrastructure**

   - Both use ``ptxas`` for final assembly
   - Both run on same CUDA driver
   - Both produce GPU binaries with identical format

5. **Inspection and debugging**

   - Same tools work for both (``cuobjdump``, ``nvprof``, ``nsight``)
   - PTX is human-readable assembly
   - CUBIN contains actual machine code

Further Reading

- `PTX ISA Reference <https://docs.nvidia.com/cuda/parallel-thread-execution/>`_ - Official PTX documentation
- `NVCC Documentation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/>`_ - NVIDIA's compiler guide
- `LLVM NVPTX Backend <https://llvm.org/docs/NVPTXUsage.html>`_ - LLVM's PTX generator
- `CUDA Binary Utilities <https://docs.nvidia.com/cuda/cuda-binary-utilities/>`_ - cuobjdump and friends
