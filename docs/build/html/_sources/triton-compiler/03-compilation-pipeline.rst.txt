Compilation Pipeline: AST to GPU Binary

This document covers the complete compilation pipeline from Python AST through MLIR transformations to GPU binary.

Code Generation: Python AST -> Triton IR

The code generator (`code_generator.py <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/compiler/code_generator.py>`_) converts Python AST to Triton IR (TTIR).

CodeGenerator Class
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class CodeGenerator(ast.NodeVisitor):
        def __init__(self, context, prototype, gscope, function_name, jit_fn, *, options, codegen_fns,
                     module_map, is_kernel=False):
            self.context = context
            self.builder = ir.builder(context)  # MLIR IR builder
            self.semantic = TritonSemantic(self.builder)  # Triton semantics

            # Scope management
            self.lscope = {}  # Local variables
            self.gscope = gscope  # Global variables

            # Type inference
            self.function_ret_types = {}
            self.last_ret_type = None


AST Visitor Pattern
~~~~~~~~~~~~~~~~~~~

The generator uses the **Visitor pattern** to walk the Python AST:

.. code-block:: python

    def visit_FunctionDef(self, node):
        """Generate IR for function definition."""
        # Extract arguments
        arg_types = []
        for arg in node.args.args:
            arg_types.append(self.visit_annotation(arg.annotation))

        # Generate function signature
        fn_ty = self.builder.get_function_ty(arg_types_ir, ret_types_ir)
        fn = self.builder.create_function(fn_ty, self.fn_name)

        # Generate function body
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node):
        """Generate IR for assignment: x = expr"""
        # Visit right-hand side
        value = self.visit(node.value)

        # Store in local scope
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.lscope[target.id] = value

    def visit_BinOp(self, node):
        """Generate IR for binary operation: a + b"""
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)

        # Map Python operator to Triton operation
        if isinstance(node.op, ast.Add):
            return self.builder.create_add(lhs, rhs)
        elif isinstance(node.op, ast.Mul):
            return self.builder.create_mul(lhs, rhs)
        # ... more operators

~~~~~~~~~~~~~~~~~~~~~~~~~~~

Special handling for ``tl.*`` functions:

.. code-block:: python

    def visit_Call(self, node):
        """Handle function calls."""
        fn = self.visit(node.func)

        # Handle tl.load
        if fn == tl.load:
            ptr = self.visit(node.args[0])
            mask = self.visit(node.keywords['mask']) if 'mask' in node.keywords else None
            return self.builder.create_load(ptr, mask)

        # Handle tl.store
        elif fn == tl.store:
            ptr = self.visit(node.args[0])
            value = self.visit(node.args[1])
            mask = self.visit(node.keywords['mask']) if 'mask' in node.keywords else None
            return self.builder.create_store(ptr, value, mask)

        # Handle tl.program_id
        elif fn == tl.program_id:
            axis = self.visit(node.args[0])
            return self.builder.create_get_program_id(axis)


Python                  Triton Operation             MLIR Op
``tl.load(ptr)``        Load from memory             ``tt.load``
``tl.store(ptr, val)``  Store to memory              ``tt.store``
``tl.program_id(0)``    Get block ID                 ``tt.get_program_id``
``tl.arange(0, N)``     Create range                 ``tt.make_range``
``tl.dot(a, b)``        Matrix multiply              ``tt.dot``
``a + b``               Addition                     ``arith.addi``
``a < b``               Comparison                   ``arith.cmpi``

Type Inference
~~~~~~~~~~~~~~

Triton infers types automatically:

.. code-block:: python

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)

        # Infer result type from operands
        lhs_ty = lhs.type
        rhs_ty = rhs.type

        # Promote types if needed (e.g., int32 + float32 -> float32)
        result_ty = self.promote_types(lhs_ty, rhs_ty)

        # Cast operands if necessary
        lhs = self.cast(lhs, result_ty)
        rhs = self.cast(rhs, result_ty)

        return self.builder.create_add(lhs, rhs, result_ty)

~~~~~~~~~~~~~~~~~~~

For this kernel:

.. code-block:: python

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * 128 + tl.arange(0, 128)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


.. code-block:: mlir

    module {
      tt.func @add_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>,
                          %out_ptr: !tt.ptr<f32>) {
        %c128 = arith.constant 128 : i32
        %pid = tt.get_program_id x : i32
        %offset_base = arith.muli %pid, %c128 : i32

        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
        %offset_base_splat = tt.splat %offset_base : i32 -> tensor<128xi32>
        %offs = arith.addi %offset_base_splat, %range : tensor<128xi32>

        %N_splat = tt.splat %N : i32 -> tensor<128xi32>
        %mask = arith.cmpi slt, %offs, %N_splat : tensor<128xi32>

        %x_ptr_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %x_ptrs = tt.addptr %x_ptr_splat, %offs : tensor<128x!tt.ptr<f32>>
        %x = tt.load %x_ptrs, %mask : tensor<128xf32>

        %y_ptr_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %y_ptrs = tt.addptr %y_ptr_splat, %offs : tensor<128x!tt.ptr<f32>>
        %y = tt.load %y_ptrs, %mask : tensor<128xf32>

        %result = arith.addf %x, %y : tensor<128xf32>

        %out_ptrs = tt.addptr %out_ptr_splat, %offs : tensor<128x!tt.ptr<f32>>
        tt.store %out_ptrs, %result, %mask : tensor<128xf32>

        tt.return
      }
    }


The MLIR pipeline applies a series of transformation passes.

Compilation Orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Main compilation function (`compiler.py:226 <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/compiler/compiler.py#L226>`_):

.. code-block:: python

    def compile(src, target=None, options=None):
        # Get backend for target (CUDA, ROCm, etc.)
        backend = make_backend(target)

        # Add compilation stages
        stages = {}
        backend.add_stages(stages, options, src.language)

        # Apply each stage in order
        for ext, compile_ir in stages.items():
            module = compile_ir(module, metadata)

        # Return compiled kernel
        return CompiledKernel(src, metadata_group, hash)


.. code-block:: python

    stages = {
        "ttir": lambda module, metadata: module,  # Input
        "ttgir": ttir_to_ttgir,    # GPU-specific lowering
        "llir": ttgir_to_llir,      # LLVM IR generation
        "ptx": llir_to_ptx,         # PTX assembly
        "cubin": ptx_to_cubin,      # Binary
    }

~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Location:* `third_party/nvidia/backend/compiler.py <https://github.com/triton-lang/triton/tree/v3.5.1/third_party/nvidia/backend/compiler.py>`_

**Key transformations:**

1. **Add GPU layout information** - Specify how data is distributed across threads
2. **Allocate shared memory** - Insert ``tt.alloc_tensor`` operations
3. **Insert synchronization** - Add ``tt.barrier`` instructions
4. **Coalesce memory accesses** - Optimize memory access patterns
5. **Use tensor cores** - Lower ``tt.dot`` to tensor core operations

.. code-block:: python

    def ttir_to_ttgir(module, metadata):
        # Apply MLIR passes
        passes.ttir_to_ttgir(module,
                             num_warps=metadata["num_warps"],
                             num_ctas=metadata["num_ctas"],
                             capability=metadata["capability"])
        return module


- ``TritonGPUCoalesce`` - Coalesce memory accesses
- ``TritonGPURemoveLayoutConversions`` - Optimize layout changes
- ``TritonGPUAccelerateMatmul`` - Use tensor cores for matmul
- ``TritonGPUPipeline`` - Software pipelining for memory/compute overlap
- ``TritonGPUPrefetch`` - Insert prefetch instructions

TTGIR -> LLVM IR
~~~~~~~~~~~~~~~

.. code-block:: python

    def ttgir_to_llir(module, metadata):
        # Convert to LLVM IR
        passes.convert_ttgir_to_llir(module,
                                     target=metadata["target"],
                                     capability=metadata["capability"])
        return module


- ``tt.load`` -> LLVM load instructions
- ``tt.store`` -> LLVM store instructions
- ``tt.dot`` -> NVVM intrinsics for tensor cores
- Thread indexing -> ``llvm.nvvm.read.ptx.sreg.*`` calls
- Barriers -> ``llvm.nvvm.barrier0()``

Backend Compilation

LLVM IR -> PTX
~~~~~~~~~~~~~

.. code-block:: python

    def llir_to_ptx(module, metadata):
        # Use LLVM backend to generate PTX
        ptx = llvm.translate_to_asm(
            module,
            target=metadata["target"],
            features=get_features(metadata)
        )
        return ptx


.. code-block:: ptx

    .version 8.0
    .target sm_80
    .address_size 64

    .visible .entry add_kernel(
        .param .u64 x_ptr,
        .param .u64 y_ptr,
        .param .u64 out_ptr
    )
    {
        .reg .pred %p<2>;
        .reg .f32 %f<4>;
        .reg .b32 %r<8>;
        .reg .b64 %rd<10>;

        mov.u32 %r1, %tid.x;
        mov.u32 %r2, %ctaid.x;
        mul.lo.s32 %r3, %r2, 128;
        add.s32 %r1, %r1, %r3;

        ld.param.u64 %rd1, [x_ptr];
        cvt.s64.s32 %rd2, %r1;
        shl.b64 %rd3, %rd2, 2;
        add.s64 %rd4, %rd1, %rd3;
        ld.global.f32 %f1, [%rd4];

        // ... more PTX ...
    }

~~~~~~~~~~~

.. code-block:: python

    def ptx_to_cubin(ptx, metadata):
        # Call ptxas (NVIDIA's PTX assembler)
        cubin = nvidia.compile_ptx_to_cubin(
            ptx,
            arch=metadata["arch"],
            options=metadata["ptx_options"]
        )
        return cubin


.. code-block:: bash

    ptxas --gpu-name=sm_80 --output-file kernel.cubin kernel.ptx


Caching Strategy

Triton caches compiled kernels to avoid recompilation:

Cache Key Components
~~~~~~~~~~~~~~~~~~~~

1. **Source hash** - SHA-256 of source code
2. **Specialization** - Constexpr values, alignments
3. **Options** - num_warps, num_stages, etc.
4. **Environment** - CUDA version, PTX version
5. **Backend** - CUDA vs ROCm, compute capability

Cache Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ~/.triton/cache/

~~~~~~~~~~~~

.. code-block:: python

    def compile(src, target, options):
        # Compute cache key
        key = get_cache_key(src, backend, options, env_vars)
        hash = hashlib.sha256(key.encode()).hexdigest()

        # Check cache
        fn_cache_manager = get_cache_manager(hash)
        metadata_path = fn_cache_manager.get("metadata.json")

        if metadata_path is not None:
            # Cache hit!
            return CompiledKernel(src, metadata_group, hash)

        # Cache miss - compile
        # ...


Typical compilation times:

Stage               Time            Notes
AST Parsing         < 1ms           Very fast
Code Generation     5-20ms          Python AST -> TTIR
MLIR Passes         50-200ms        Optimization passes
LLVM Backend        100-500ms       PTX generation
PTX Assembly        100-300ms       ptxas invocation
**Total (cold)**    **250-1020ms**  First compilation
**Total (cached)**  **< 1ms**       Subsequent runs

Summary
-------

The compilation pipeline:

1. **Python AST -> TTIR** - Code generator visits AST nodes
2. **TTIR -> TTGIR** - Add GPU layout and memory hierarchy
3. **TTGIR -> LLIR** - Lower to LLVM IR with GPU intrinsics
4. **LLIR -> PTX** - LLVM backend generates assembly
5. **PTX -> CUBIN** - ptxas assembles binary
6. **Caching** - Store all intermediate representations

**Key files:**

- `code_generator.py <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/compiler/code_generator.py>`_ - AST -> TTIR
- `compiler.py <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/compiler/compiler.py>`_ - Pipeline orchestration
- `third_party/nvidia/backend/compiler.py <https://github.com/triton-lang/triton/tree/v3.5.1/third_party/nvidia/backend/compiler.py>`_ - NVIDIA backend
- `lib/Dialect/TritonGPU/ <https://github.com/triton-lang/triton/tree/v3.5.1/lib/Dialect/TritonGPU>`_ - MLIR passes (C++)

The compiler's architecture allows for:

- **Extensibility** - Easy to add new backends
- **Optimization** - Multiple passes at different levels
- **Debugging** - Can inspect IR at each stage
- **Performance** - Aggressive caching and specialization
