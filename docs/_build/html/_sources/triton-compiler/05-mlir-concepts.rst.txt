MLIR: Core Concepts and Triton Usage
=====================================

This document explains MLIR (Multi-Level Intermediate Representation), the compiler infrastructure that powers Triton's compilation pipeline.

What is MLIR?
-------------

**MLIR (Multi-Level Intermediate Representation)** is a compiler infrastructure project that provides a flexible framework for building optimizing compilers.

**Created by:** Google (now part of LLVM project)

**Purpose:** Enable building domain-specific compilers with reusable components

**Key idea:** Instead of one "universal" IR, support multiple IRs (dialects) that can coexist and transform between each other.

The Problem MLIR Solves
------------------------

Traditional Compiler Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**LLVM IR** (the traditional choice) has limitations:

.. code-block:: text

    High-level code (Python, TensorFlow, etc.)
            ↓
    [MASSIVE LOWERING GAP] ← Information loss!
            ↓
    LLVM IR (very low-level)
            ↓
    Machine code

**Problems:**

1. **Information loss** - High-level semantics disappear immediately
2. **Optimization difficulty** - Hard to optimize after lowering
3. **Single abstraction level** - Can't represent domain-specific concepts
4. **Vendor lock-in** - Different vendors create incompatible IRs

Example: Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**High-level code:**

.. code-block:: python

    C = matmul(A, B)  # Semantic: "matrix multiplication"

**Direct lowering to LLVM IR loses meaning:**

.. code-block:: llvm

    ; LLVM IR - just loops and arithmetic!
    ; Lost information: "this is a matrix multiply"
    for i in rows(A):
        for j in cols(B):
            for k in cols(A):
                C[i][j] += A[i][k] * B[k][j]  # Generic loops

**With MLIR, preserve semantics longer:**

.. code-block:: mlir

    // High-level: Linalg dialect preserves "matrix multiply" operation
    linalg.matmul ins(%A, %B) outs(%C)

    // Mid-level: Affine dialect with loop structure
    affine.for %i in [0, M):
        affine.for %j in [0, N):
            affine.for %k in [0, K):
                // ...

    // Low-level: LLVM dialect
    llvm.mul %a, %b

This **gradual lowering** preserves optimization opportunities at each level.

MLIR Philosophy
~~~~~~~~~~~~~~~

.. code-block:: text

    Traditional Compiler:
    ─────────────────────
    Source → [BIG STEP] → LLVM IR → Binary
                    ↑
            Loss of information!

    MLIR Compiler:
    ──────────────
    Source → Dialect1 → Dialect2 → ... → DialectN → Binary
             ↓         ↓         ↓         ↓
          Small, incremental lowering steps
          (Preserve information as long as possible)

**Key benefits:**

- ✅ Gradual lowering preserves semantics
- ✅ Multiple abstraction levels coexist
- ✅ Reusable optimization passes
- ✅ Domain-specific dialects

Core MLIR Concepts
------------------

1. Dialects
~~~~~~~~~~~

A **dialect** is a namespace for operations, types, and attributes.

**Think of dialects as "sublanguages" within MLIR.**

Common Dialects
^^^^^^^^^^^^^^^

==================  ==============================================  ====================
Dialect             Purpose                                         Abstraction Level
==================  ==============================================  ====================
``linalg``          Linear algebra operations                       High-level
``tensor``          Tensor operations                               High-level
``scf``             Structured control flow (for, while, if)        Mid-level
``affine``          Polyhedral loop optimizations                   Mid-level
``arith``           Arithmetic operations (add, mul, etc.)          Low-level
``llvm``            LLVM IR operations                              Very low-level
``gpu``             Generic GPU operations                          GPU-specific
``nvgpu``           NVIDIA GPU-specific operations                  NVIDIA-specific
==================  ==============================================  ====================

Triton's Custom Dialects
^^^^^^^^^^^^^^^^^^^^^^^^^

Triton defines its own dialects:

==================  ==============================================  ====================
Dialect             Purpose                                         File Location
==================  ==============================================  ====================
``tt``              Triton dialect (TTIR)                           ``lib/Dialect/Triton/``
``ttg``             Triton GPU dialect (TTGIR)                      ``lib/Dialect/TritonGPU/``
``ttng``            Triton NVIDIA GPU dialect                       ``lib/Dialect/TritonNvidiaGPU/``
==================  ==============================================  ====================

**Example:** Mixing dialects in one function

.. code-block:: mlir

    func.func @example(%arg0: tensor<128xf32>) -> tensor<128xf32> {
        %c0 = arith.constant 0 : index        // arith dialect
        %c128 = arith.constant 128 : index

        %0 = scf.for %i = %c0 to %c128        // scf dialect
            iter_args(%arg = %arg0) -> tensor<128xf32> {
            %1 = linalg.generic { ... }       // linalg dialect
            scf.yield %1
        }

        return %0 : tensor<128xf32>
    }

2. Operations
~~~~~~~~~~~~~

**Operations** (ops) are the fundamental unit of computation in MLIR.

Operation Anatomy
^^^^^^^^^^^^^^^^^

.. code-block:: mlir

    %result = dialect.operation(%operand1, %operand2) {attribute = value} : type

    // Example:
    %sum = arith.addi %a, %b : i32
    //  ↑       ↑      ↑   ↑    ↑
    // result  op   operands  type

**Components:**

- **Opcode:** ``dialect.operation`` (e.g., ``arith.addi``)
- **Operands:** Input values (SSA form)
- **Results:** Output values (SSA form)
- **Attributes:** Compile-time constants (metadata)
- **Types:** Data types of operands and results
- **Regions:** Nested code blocks (optional)

SSA Form (Static Single Assignment)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every value is defined **exactly once**:

.. code-block:: mlir

    // Valid SSA:
    %0 = arith.constant 5 : i32
    %1 = arith.constant 10 : i32
    %2 = arith.addi %0, %1 : i32  // %2 defined once

    // Invalid (not SSA):
    %x = arith.constant 5 : i32
    %x = arith.addi %x, %x : i32  // ERROR: %x redefined!

**Benefits:**

- Simplifies optimization (no aliasing confusion)
- Easier data flow analysis
- Natural for functional-style transformations

Operation Examples
^^^^^^^^^^^^^^^^^^

**Arithmetic:**

.. code-block:: mlir

    %sum = arith.addi %a, %b : i32           // Integer addition
    %prod = arith.mulf %x, %y : f32          // Float multiplication
    %cmp = arith.cmpi slt, %a, %b : i32      // Compare: a < b

**Triton-specific:**

.. code-block:: mlir

    %pid = tt.get_program_id x : i32         // Get block ID
    %range = tt.make_range {start=0, end=128} : tensor<128xi32>
    %ptr = tt.addptr %base, %offset : !tt.ptr<f32>
    %data = tt.load %ptr, %mask : tensor<128xf32>

**Control flow:**

.. code-block:: mlir

    scf.if %condition {
        // true branch
    } else {
        // false branch
    }

    scf.for %i = %lb to %ub step %step {
        // loop body
    }

3. Regions and Blocks
~~~~~~~~~~~~~~~~~~~~~

Regions
^^^^^^^

A **region** is a container for code, similar to a scope.

.. code-block:: mlir

    scf.if %cond {
        // ← This is a region
        %x = arith.constant 1 : i32
    }

**Properties:**

- Isolated scope (can have local SSA values)
- Can contain multiple basic blocks
- Used for control flow (if, for, while, functions)

Blocks
^^^^^^

A **block** is a sequence of operations with a single entry point.

.. code-block:: mlir

    ^bb0(%arg0: i32, %arg1: i32):  // Block with arguments
        %sum = arith.addi %arg0, %arg1 : i32
        cf.br ^bb1(%sum : i32)      // Branch to next block

    ^bb1(%result: i32):
        return %result : i32

**Key properties:**

- Basic block (no control flow within the block)
- Can have block arguments (like function parameters)
- Ends with a terminator (return, branch, etc.)

Example with Regions and Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: mlir

    func.func @example(%n: i32) -> i32 {
        // Function body is a region with one block

        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32

        %result = scf.for %i = %c0 to %n step %c1
            iter_args(%acc = %c0) -> i32 {
            // ← For loop body is a region

            %new_acc = arith.addi %acc, %i : i32
            scf.yield %new_acc : i32  // Yield to next iteration
        }

        return %result : i32
    }

4. Types
~~~~~~~~

MLIR has a **flexible type system** that dialects can extend.

Built-in Types
^^^^^^^^^^^^^^

.. code-block:: mlir

    // Integers
    i1                      // 1-bit (boolean)
    i8, i16, i32, i64       // Signed integers
    ui8, ui16, ui32         // Unsigned integers

    // Floats
    f16, f32, f64           // IEEE floats
    bf16                    // bfloat16

    // Vectors and tensors
    vector<4xf32>           // 4-element float vector
    tensor<128x128xf32>     // 2D tensor
    tensor<?x?xf32>         // Dynamic-shape tensor

    // Pointers
    !llvm.ptr<f32>          // LLVM pointer to float

    // Functions
    (i32, i32) -> i32       // Function type

Triton Custom Types
^^^^^^^^^^^^^^^^^^^

.. code-block:: mlir

    !tt.ptr<f32>                    // Triton pointer
    !tt.ptr<tensor<128xf32>>        // Pointer to tensor
    tensor<128x!tt.ptr<f32>>        // Tensor of pointers

**Why custom types?**

- Triton pointers have different semantics than LLVM pointers
- Support block/tensor operations naturally
- Enable Triton-specific optimizations

Type Conversion
^^^^^^^^^^^^^^^

Types change as you lower between dialects:

.. code-block:: mlir

    // High-level (Triton)
    %data : tensor<128xf32>

    // Mid-level (TritonGPU) - add layout
    %data : tensor<128xf32, #ttg.blocked<{threads=128}>>

    // Low-level (LLVM) - flatten to individual values
    %data : !llvm.array<128 x f32>

5. Attributes
~~~~~~~~~~~~~

**Attributes** are compile-time constants attached to operations.

.. code-block:: mlir

    // Integer attribute
    %c = arith.constant 42 : i32
    //                   ↑↑
    //              attribute value

    // Dictionary attribute
    tt.make_range {start = 0 : i32, end = 128 : i32}
    //            ↑                                  ↑
    //            dictionary with start/end fields

    // Array attribute
    llvm.call @func(%arg) {fastmathFlags = #llvm.fastmath<fast>}

Common Attribute Types
^^^^^^^^^^^^^^^^^^^^^^

- **IntegerAttr:** ``42 : i32``
- **FloatAttr:** ``3.14 : f32``
- **StringAttr:** ``"hello"``
- **ArrayAttr:** ``[1, 2, 3]``
- **DictionaryAttr:** ``{key1 = val1, key2 = val2}``
- **TypeAttr:** Type as a value
- **UnitAttr:** Presence/absence flag

Triton Layout Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: mlir

    // Blocked layout: how data is distributed across threads
    #ttg.blocked<{
        sizePerThread = [1, 4],
        threadsPerWarp = [2, 16],
        warpsPerCTA = [4, 1],
        order = [1, 0]
    }>

    // Shared memory layout
    #ttg.shared<{
        vec = 8,
        perPhase = 2,
        maxPhase = 4
    }>

6. Passes
~~~~~~~~~

**Passes** are transformations that modify MLIR code.

Pass Types
^^^^^^^^^^

1. **Analysis passes** - Gather information (don't modify IR)
2. **Transformation passes** - Modify IR
3. **Canonicalization** - Simplify IR to canonical form
4. **Lowering passes** - Convert between dialects

Example Pass Pipeline
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Triton's pass pipeline (simplified)
    pm = PassManager()

    # High-level optimizations (TTIR)
    pm.add_pass("triton-combine")              # Combine operations
    pm.add_pass("canonicalize")                # Simplify

    # Lower to GPU IR (TTIR → TTGIR)
    pm.add_pass("triton-gpu-coalesce")         # Coalesce memory accesses
    pm.add_pass("triton-gpu-pipeline")         # Software pipelining
    pm.add_pass("triton-gpu-prefetch")         # Insert prefetch
    pm.add_pass("triton-gpu-optimize-dot")     # Optimize matmul

    # Lower to LLVM (TTGIR → LLVM)
    pm.add_pass("convert-triton-gpu-to-llvm")
    pm.add_pass("convert-scf-to-cf")           # Structured → unstructured control flow
    pm.add_pass("convert-arith-to-llvm")

    pm.run(module)

Triton-Specific Passes
^^^^^^^^^^^^^^^^^^^^^^

*Location:* `lib/Dialect/TritonGPU/Transforms/ <https://github.com/triton-lang/triton/tree/v3.5.1/lib/Dialect/TritonGPU/Transforms>`_

**Key passes:**

- ``TritonGPUCoalesce`` - Optimize memory access patterns
- ``TritonGPUPipeline`` - Overlap computation and memory transfers
- ``TritonGPUPrefetch`` - Insert prefetch instructions
- ``TritonGPUAccelerateMatmul`` - Use tensor cores for matmul
- ``TritonGPURemoveLayoutConversions`` - Eliminate redundant layout changes

MLIR in Triton
--------------

Why Triton Uses MLIR
~~~~~~~~~~~~~~~~~~~~

1. **Multi-level representation**

   - TTIR: High-level block operations
   - TTGIR: GPU-specific with layouts
   - LLVM IR: Low-level machine operations

2. **Reusable infrastructure**

   - Don't reinvent parsing, printing, pass management
   - Use existing dialects (arith, scf, llvm)
   - Benefit from MLIR community improvements

3. **Extensibility**

   - Easy to add new operations (TableGen)
   - Define custom types and attributes
   - Write dialect-specific passes

4. **Multi-backend support**

   - Same TTIR can target NVIDIA (NVPTX) or AMD (AMDGCN)
   - Backend-specific dialects (ttng for NVIDIA, ttag for AMD)

Triton's MLIR Dialects
~~~~~~~~~~~~~~~~~~~~~~~

Triton Dialect (tt)
^^^^^^^^^^^^^^^^^^^

*Location:* `lib/Dialect/Triton/ <https://github.com/triton-lang/triton/tree/v3.5.1/lib/Dialect/Triton>`_

**High-level operations, backend-agnostic.**

.. code-block:: mlir

    // Get block ID
    %pid = tt.get_program_id x : i32

    // Create range
    %range = tt.make_range {start = 0 : i32, end = 128 : i32}
        : tensor<128xi32>

    // Broadcast scalar to tensor
    %splat = tt.splat %value : i32 -> tensor<128xi32>

    // Pointer arithmetic
    %ptrs = tt.addptr %base_ptr, %offsets
        : tensor<128x!tt.ptr<f32>>, tensor<128xi32>

    // Load from memory
    %data = tt.load %ptrs, %mask, %other
        : tensor<128x!tt.ptr<f32>>

    // Store to memory
    tt.store %ptrs, %data, %mask : tensor<128x!tt.ptr<f32>>

    // Matrix multiplication (conceptual)
    %c = tt.dot %a, %b, %acc
        : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>

TritonGPU Dialect (ttg)
^^^^^^^^^^^^^^^^^^^^^^^

*Location:* `lib/Dialect/TritonGPU/ <https://github.com/triton-lang/triton/tree/v3.5.1/lib/Dialect/TritonGPU>`_

**GPU-specific operations with data layout information.**

.. code-block:: mlir

    // Define layout encoding
    #blocked = #ttg.blocked<{
        sizePerThread = [1, 4],
        threadsPerWarp = [2, 16],
        warpsPerCTA = [4, 1]
    }>

    // Tensor with layout
    %data : tensor<128x128xf32, #blocked>

    // Convert between layouts
    %new_data = ttg.convert_layout %data
        : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked2>

    // Allocate shared memory
    %smem = ttg.alloc_tensor : tensor<128x128xf32, #shared>

    // Insert barrier
    ttg.barrier

    // Async operations (for pipelining)
    %token = ttg.async_commit_group
    ttg.async_wait {num = 0 : i32}

TritonNvidiaGPU Dialect (ttng)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Location:* `lib/Dialect/TritonNvidiaGPU/ <https://github.com/triton-lang/triton/tree/v3.5.1/lib/Dialect/TritonNvidiaGPU>`_

**NVIDIA-specific operations (Hopper+, tensor cores, TMA).**

.. code-block:: mlir

    // Warp group dot (Hopper tensor cores)
    %c = ttng.warp_group_dot %a, %b, %acc
        : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>

    // Tensor Memory Accelerator (TMA) load
    %data = ttng.tma_load %desc, %coords
        : !ttng.tma_descriptor -> tensor<128x128xf16>

    // Distributed shared memory (Hopper)
    %smem = ttng.alloc_dsmem : tensor<128x128xf32, #ttng.dsmem>

Example: Lowering Through Dialects
-----------------------------------

Let's trace a simple operation through all dialects.

Python Source
~~~~~~~~~~~~~

.. code-block:: python

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
        offs = tl.arange(0, 128)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        out = x + y
        tl.store(out_ptr + offs, out)

Stage 1: Triton IR (TTIR)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Backend-agnostic, high-level block operations.**

.. code-block:: mlir

    module {
      tt.func @add_kernel(%x_ptr: !tt.ptr<f32>,
                          %y_ptr: !tt.ptr<f32>,
                          %out_ptr: !tt.ptr<f32>) {
        // Create range [0, 128)
        %range = tt.make_range {start = 0, end = 128} : tensor<128xi32>

        // Broadcast base pointers to tensors
        %x_ptr_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %y_ptr_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %out_ptr_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>

        // Compute pointer offsets
        %x_ptrs = tt.addptr %x_ptr_splat, %range : tensor<128x!tt.ptr<f32>>
        %y_ptrs = tt.addptr %y_ptr_splat, %range : tensor<128x!tt.ptr<f32>>
        %out_ptrs = tt.addptr %out_ptr_splat, %range : tensor<128x!tt.ptr<f32>>

        // Load data
        %x = tt.load %x_ptrs : tensor<128xf32>
        %y = tt.load %y_ptrs : tensor<128xf32>

        // Compute
        %out = arith.addf %x, %y : tensor<128xf32>

        // Store result
        tt.store %out_ptrs, %out : tensor<128xf32>

        tt.return
      }
    }

**Notice:**

- No layout information yet
- No GPU-specific operations
- Pure block-level semantics

Stage 2: TritonGPU IR (TTGIR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Add GPU layout information.**

.. code-block:: mlir

    #blocked = #ttg.blocked<{
        sizePerThread = [4],
        threadsPerWarp = [32],
        warpsPerCTA = [1]
    }>

    module {
      tt.func @add_kernel(%x_ptr: !tt.ptr<f32>,
                          %y_ptr: !tt.ptr<f32>,
                          %out_ptr: !tt.ptr<f32>) {
        // Range with layout
        %range = tt.make_range {start = 0, end = 128}
            : tensor<128xi32, #blocked>

        // Pointers with layout
        %x_ptrs = tt.addptr %x_ptr_splat, %range
            : tensor<128x!tt.ptr<f32>, #blocked>

        // Load with layout (coalesced access)
        %x = tt.load %x_ptrs
            : tensor<128xf32, #blocked>
        %y = tt.load %y_ptrs
            : tensor<128xf32, #blocked>

        // Compute with layout
        %out = arith.addf %x, %y
            : tensor<128xf32, #blocked>

        // Store with layout
        tt.store %out_ptrs, %out
            : tensor<128xf32, #blocked>

        tt.return
      }
    }

**Notice:**

- Layout attributes added (``#blocked``)
- Specifies data distribution across threads
- Enables memory coalescing optimizations

Stage 3: LLVM Dialect
~~~~~~~~~~~~~~~~~~~~~~

**Lowered to LLVM IR (within MLIR).**

.. code-block:: mlir

    module {
      llvm.func @add_kernel(%x_ptr: !llvm.ptr<f32>,
                            %y_ptr: !llvm.ptr<f32>,
                            %out_ptr: !llvm.ptr<f32>) {
        // Get thread ID
        %tid = llvm.call @llvm.nvvm.read.ptx.sreg.tid.x()
            : () -> i32

        // Each thread handles 4 elements (sizePerThread = 4)
        %c0 = llvm.mlir.constant(0 : i32) : i32
        %c4 = llvm.mlir.constant(4 : i32) : i32

        // Loop over thread's elements
        llvm.br ^loop(%c0 : i32)

      ^loop(%i: i32):
        %cond = llvm.icmp "slt" %i, %c4 : i32
        llvm.cond_br %cond, ^body, ^exit

      ^body:
        // Calculate global offset
        %offset = llvm.add %tid, %i : i32

        // Load x[offset]
        %x_gep = llvm.getelementptr %x_ptr[%offset]
            : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
        %x_val = llvm.load %x_gep : !llvm.ptr<f32>

        // Load y[offset]
        %y_gep = llvm.getelementptr %y_ptr[%offset]
            : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
        %y_val = llvm.load %y_gep : !llvm.ptr<f32>

        // Compute
        %sum = llvm.fadd %x_val, %y_val : f32

        // Store out[offset]
        %out_gep = llvm.getelementptr %out_ptr[%offset]
            : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
        llvm.store %sum, %out_gep : !llvm.ptr<f32>

        %i_next = llvm.add %i, %c1 : i32
        llvm.br ^loop(%i_next : i32)

      ^exit:
        llvm.return
      }
    }

**Notice:**

- Explicit thread indexing (``tid.x``)
- Loop over per-thread elements
- Individual memory operations
- NVVM intrinsics for GPU built-ins

Stage 4: LLVM IR (Actual)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Translated from LLVM dialect to actual LLVM IR.**

.. code-block:: llvm

    define void @add_kernel(float* %x_ptr, float* %y_ptr, float* %out_ptr) {
    entry:
      %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
      br label %loop

    loop:
      %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
      %cond = icmp slt i32 %i, 4
      br i1 %cond, label %body, label %exit

    body:
      %offset = add i32 %tid, %i

      %x_gep = getelementptr float, float* %x_ptr, i32 %offset
      %x_val = load float, float* %x_gep, align 4

      %y_gep = getelementptr float, float* %y_ptr, i32 %offset
      %y_val = load float, float* %y_gep, align 4

      %sum = fadd float %x_val, %y_val

      %out_gep = getelementptr float, float* %out_ptr, i32 %offset
      store float %sum, float* %out_gep, align 4

      %i.next = add i32 %i, 1
      br label %loop

    exit:
      ret void
    }

    declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

**This is standard LLVM IR** that the NVPTX backend can compile to PTX.

MLIR Tools and Ecosystem
-------------------------

Command-Line Tools
~~~~~~~~~~~~~~~~~~

**mlir-opt** - Optimize and transform MLIR

.. code-block:: bash

    # Run canonicalization pass
    mlir-opt --canonicalize input.mlir -o output.mlir

    # Run custom pass
    mlir-opt --triton-gpu-pipeline input.mlir

    # Lower to LLVM dialect
    mlir-opt --convert-triton-to-llvm input.mlir

**mlir-translate** - Translate between MLIR and other formats

.. code-block:: bash

    # MLIR → LLVM IR
    mlir-translate --mlir-to-llvmir input.mlir -o output.ll

    # LLVM IR → MLIR
    mlir-translate --import-llvm input.ll -o output.mlir

**mlir-cpu-runner** - JIT execute MLIR on CPU

.. code-block:: bash

    mlir-cpu-runner input.mlir --entry-point=main

TableGen for Defining Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**TableGen** is a domain-specific language for defining MLIR operations.

*Example from Triton:* `TritonOps.td <https://github.com/triton-lang/triton/tree/v3.5.1/include/triton/Dialect/Triton/IR/TritonOps.td>`_

.. code-block:: tablegen

    // Define tt.load operation
    def TT_LoadOp : TT_Op<"load", [MemoryEffects<[MemRead]>]> {
      let summary = "Load from memory";

      let arguments = (ins
        TT_PtrLike:$ptr,        // Pointer operand
        Optional<I1Tensor>:$mask,  // Optional mask
        Optional<AnyType>:$other   // Optional default value
      );

      let results = (outs
        AnyType:$result           // Loaded data
      );

      let assemblyFormat = [{
        $ptr (`,` $mask^ (`,` $other^)?)? attr-dict `:` type($result)
      }];
    }

**TableGen generates:**

- C++ class for the operation
- Parsing and printing code
- Type inference
- Verification

Debugging MLIR
~~~~~~~~~~~~~~

**Print IR at each stage:**

.. code-block:: bash

    # Set environment variable
    export MLIR_ENABLE_DUMP=1

    # Triton will dump IR at each pass
    python kernel.py

**Use ``--debug`` flag:**

.. code-block:: bash

    mlir-opt --debug input.mlir

**Print specific pass output:**

.. code-block:: python

    # In Python
    import triton

    @triton.jit
    def kernel(...):
        pass

    # Compile with debug
    kernel[grid](..., debug=True)

MLIR Resources
--------------

Official Documentation
~~~~~~~~~~~~~~~~~~~~~~

- `MLIR Website <https://mlir.llvm.org/>`_ - Official documentation
- `MLIR Dialects <https://mlir.llvm.org/docs/Dialects/>`_ - Built-in dialects
- `MLIR Language Reference <https://mlir.llvm.org/docs/LangRef/>`_ - Syntax and semantics
- `TableGen Reference <https://mlir.llvm.org/docs/OpDefinitions/>`_ - Defining operations

Tutorials
~~~~~~~~~

- `MLIR Toy Tutorial <https://mlir.llvm.org/docs/Tutorials/Toy/>`_ - Build a compiler from scratch
- `MLIR Talks <https://mlir.llvm.org/talks/>`_ - Conference presentations
- `MLIR Community <https://discourse.llvm.org/c/mlir/>`_ - Discussion forum

Triton-Specific
~~~~~~~~~~~~~~~

- `Triton MLIR Dialects <https://github.com/triton-lang/triton/tree/v3.5.1/include/triton/Dialect>`_ - Operation definitions
- `Triton Passes <https://github.com/triton-lang/triton/tree/v3.5.1/lib/Dialect/TritonGPU/Transforms>`_ - Transformation passes
- `Triton IR Examples <https://github.com/triton-lang/triton/tree/v3.5.1/test/Triton>`_ - Test cases with IR

Summary
-------

Key Concepts Recap
~~~~~~~~~~~~~~~~~~

1. **MLIR = Multi-Level Intermediate Representation**

   - Multiple dialects coexist
   - Gradual lowering preserves semantics
   - Reusable infrastructure

2. **Dialects** - Namespaces for operations, types, attributes

   - Triton has ``tt``, ``ttg``, ``ttng`` dialects
   - Standard dialects: ``arith``, ``scf``, ``llvm``

3. **Operations** - Fundamental computation units

   - SSA form (single assignment)
   - Have operands, results, attributes, types

4. **Types** - Flexible type system

   - Built-in: integers, floats, tensors
   - Custom: Triton pointers, layouts

5. **Passes** - IR transformations

   - Analysis, optimization, lowering
   - Triton has GPU-specific passes

Why MLIR Matters for Triton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Without MLIR:**

- ❌ Would need to build entire compiler infrastructure
- ❌ Hard to support multiple GPU vendors
- ❌ Difficult to add new optimizations
- ❌ Can't reuse existing tools and passes

**With MLIR:**

- ✅ Reuse robust infrastructure (parsing, printing, pass management)
- ✅ Easy multi-backend support (NVIDIA, AMD, Intel)
- ✅ Modular, extensible design
- ✅ Benefit from MLIR ecosystem improvements
- ✅ Gradual lowering preserves optimization opportunities

The Big Picture
~~~~~~~~~~~~~~~

.. code-block:: text

    Triton Compiler Pipeline:

    Python AST
         ↓
    [Code Generator] ← Converts AST to MLIR
         ↓
    TTIR (tt dialect) ← High-level block operations
         ↓
    [MLIR Passes] ← Optimization, coalescing
         ↓
    TTGIR (ttg dialect) ← Add GPU layouts
         ↓
    [MLIR Passes] ← Pipelining, prefetch, tensor cores
         ↓
    LLVM Dialect ← Still MLIR, but LLVM operations
         ↓
    [mlir-translate] ← Convert MLIR → LLVM IR
         ↓
    LLVM IR ← Standard LLVM
         ↓
    [NVPTX Backend] ← LLVM's PTX generator
         ↓
    PTX Assembly

**MLIR enables this multi-stage, optimizing pipeline** with reusable, modular components.
