The @triton.jit Decorator
=========================

The ``@triton.jit`` decorator is the entry point to Triton compilation. This document explains how it works, from Python function decoration to compilation triggering.

Overview
--------

When you write a Triton kernel, you decorate it with ``@triton.jit``:

.. code-block:: python

    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

The ``@triton.jit`` decorator transforms the function into a ``JITFunction`` object that handles compilation and execution.

How the Decorator Works
------------------------

Implementation
~~~~~~~~~~~~~~

The decorator is defined in `jit.py <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/jit.py>`_:

.. code-block:: python

    def jit(fn):
        """
        Decorator for triton kernels.

        This decorator transforms a Python function into a JITFunction object
        that can be compiled and executed on the GPU.
        """
        if isinstance(fn, JITFunction):
            return fn
        if not callable(fn):
            raise TypeError("triton.jit requires a callable argument")

        # Create JITFunction wrapper
        return JITFunction(fn)

When you apply ``@triton.jit``, it creates a ``JITFunction`` instance wrapping your original function.

The JITCallable Base Class
---------------------------

All JIT-compiled Triton code inherits from ``JITCallable``, which provides core functionality:

Source Code Extraction
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class JITCallable:
        def __init__(self, fn):
            self.fn = fn
            self.signature = inspect.signature(fn)

            # Extract source code
            self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)

            # Remove decorators and dedent
            src = textwrap.dedent("".join(self.raw_src))
            src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
            self._src = src

*Location:* `jit.py:455-470 <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/jit.py#L455-L470>`_

**Why extract source?**

- Triton needs the Python AST to compile
- Source is hashed for caching
- Enables runtime code inspection

AST Parsing
~~~~~~~~~~~

.. code-block:: python

    def parse(self):
        """Parse the source code into an AST."""
        tree = ast.parse(self._src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

*Location:* `jit.py:527-532 <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/jit.py#L527-L532>`_

This converts the source string into a Python AST that the code generator can process.

Cache Key Generation
~~~~~~~~~~~~~~~~~~~~

Every kernel has a **cache key** based on its source code and dependencies:

.. code-block:: python

    @property
    def cache_key(self) -> str:
        with self._hash_lock:
            if self.hash is not None:
                return self.hash

            # Find all dependencies
            nonlocals = inspect.getclosurevars(self.fn).nonlocals
            dependencies_finder = DependenciesFinder(
                name=self._fn_name,
                globals=self.__globals__,
                nonlocals=nonlocals,
                src=self.src
            )
            dependencies_finder.visit(self.parse())

            # Hash = source + dependencies + line number
            self.hash = dependencies_finder.ret + str(self.starting_line_number)
            self.used_global_vals = dict(sorted(dependencies_finder.used_global_vals.items()))

            # Add constexpr values to hash
            from triton.language.core import constexpr
            self.hash += str([
                (name, val)
                for (name, _), (val, _) in self.used_global_vals.items()
                if isinstance(val, constexpr)
            ])

            self.hash = hashlib.sha256(self.hash.encode("utf-8")).hexdigest()
        return self.hash

*Location:* `jit.py:498-519 <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/jit.py#L498-L519>`_

**Cache key includes:**

- Source code (SHA-256 hash)
- Line number (for duplicate function names)
- Global variable values used by the function
- Constexpr values
- All transitively called functions

Dependencies Tracking
---------------------

The ``DependenciesFinder`` class walks the AST to find all dependencies:

.. code-block:: python

    class DependenciesFinder(ast.NodeVisitor):
        """
        Finds dependencies of a JITFunction.

        Tracks:
        1. Global variables accessed by the function
        2. Other JITCallable functions called
        3. Their values at compilation time
        """

        def __init__(self, name, globals, nonlocals, src):
            super().__init__()
            self.name = name
            self.hasher = hashlib.sha256(src.encode("utf-8"))
            self.globals = globals
            self.nonlocals = nonlocals

            # Map: (var_name, id(__globals__)) -> (var_value, __globals__)
            self.used_global_vals = {}

*Location:* `jit.py:34-86 <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/jit.py#L34-L86>`_

Tracking Global Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def visit_Name(self, node):
        """Visit a variable name in the AST."""
        if type(node.ctx) is ast.Store:
            return node.id  # Writing to variable

        if node.id in self.local_names:
            return None  # Local variable shadows global

        # Look up in globals/nonlocals
        def name_lookup(name):
            val = self.globals.get(name, None)
            if val is not None:
                return val, self.globals
            val = self.nonlocals.get(name, None)
            if val is not None:
                return val, self.nonlocals
            return None, None

        val, var_dict = name_lookup(node.id)

        # Record the value for cache invalidation
        self.record_reference(val, var_dict, node.id)
        return val

*Location:* `jit.py:156-178 <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/jit.py#L156-L178>`_

**Why track globals?**

If you do this:

.. code-block:: python

    CONSTANT = 42

    @triton.jit
    def kernel():
        x = CONSTANT  # Uses global

    kernel[grid](...)  # Compiles with CONSTANT=42

    CONSTANT = 100
    kernel[grid](...)  # ERROR: CONSTANT changed!

Triton detects the change and raises an error, preventing silent bugs.

Handling Nested Function Calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a kernel calls another JIT function:

.. code-block:: python

    @triton.jit
    def helper(x):
        return x * 2

    @triton.jit
    def main_kernel():
        y = helper(5)

The dependency finder:

1. Recognizes ``helper`` as a ``JITCallable``
2. Includes ``helper``'s cache key in ``main_kernel``'s hash
3. Merges ``helper``'s global variables into ``main_kernel``'s

.. code-block:: python

    def _update_hash(self, func):
        assert isinstance(func, JITCallable)

        # Check for conflicts in global variable values
        for k in self.used_global_vals.keys() & func.used_global_vals.keys():
            var_name, _ = k
            v1, _ = self.used_global_vals[k]
            v2, _ = func.used_global_vals[k]
            if v1 != v2:
                raise RuntimeError(
                    f"Global variable {var_name} has value {v1} in {self.name}, "
                    f"but {func.__name__} has conflicting value {v2}"
                )

        # Merge dependencies
        self.used_global_vals.update(func.used_global_vals)

        # Update hash with called function's hash
        func_key = func.cache_key
        func_key += str(getattr(func, "noinline", False))
        self.hasher.update(func_key.encode("utf-8"))

*Location:* `jit.py:99-115 <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/jit.py#L99-L115>`_

The JITFunction Class
---------------------

``JITFunction`` extends ``JITCallable`` to add kernel launching:

Initialization
~~~~~~~~~~~~~~

.. code-block:: python

    class JITFunction(JITCallable):
        def __init__(self, fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None,
                     debug=None, noinline=None, repr=None, launch_metadata=None):
            super().__init__(fn)

            # Cache of compiled kernels
            # Key: (specialization, options) -> CompiledKernel
            self.cache = defaultdict(dict)
            self.kernel_key_cache = {}

            # Attributes
            self.fn.arg_names = [arg.name for arg in self.signature.parameters.values()]
            self.fn.divisibility = 16
            self.do_not_specialize = do_not_specialize or []
            self.do_not_specialize_on_alignment = do_not_specialize_on_alignment or []
            self.debug = debug
            self.noinline = noinline
            self.launch_metadata = launch_metadata

*Location:* `jit.py:608-625 <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/jit.py#L608-L625>`_

The ``cache`` dictionary stores compiled binaries, avoiding recompilation.

Kernel Call Handling
~~~~~~~~~~~~~~~~~~~~

When you call a kernel like ``kernel[grid](...)``:

.. code-block:: python

    def __getitem__(self, grid):
        """
        Returns a callable that can launch the kernel with the given grid.

        Usage: kernel[grid](args)
        """
        return lambda *args, **kwargs: self.run(
            grid=grid,
            *args,
            **kwargs
        )

Then ``run()`` handles:

1. Argument processing
2. Specialization (converting runtime values to compile-time constants)
3. Cache lookup
4. Compilation (if needed)
5. Kernel launch

Argument Specialization
~~~~~~~~~~~~~~~~~~~~~~~

Triton can **specialize** kernels based on runtime argument values:

.. code-block:: python

    # Dynamic specialization function
    def _make_specialization_fn(sig, kparams):
        specialization = []

        for name, kp in zip(sig.parameters.keys(), kparams):
            if kp.is_constexpr:
                # Constexpr: always specialized
                specialization.append(f'("constexpr", {name})')
            else:
                # Regular parameter: may specialize based on attributes
                is_const = 'True' if kp.is_const else 'False'
                specialize = 'False' if kp.do_not_specialize else 'True'
                align = 'False' if kp.do_not_specialize_on_alignment else 'True'

                ret = f"specialize_impl(backend, {name}, {is_const}, {specialize}, {align})"

                if kp.annotation_type:
                    specialization.append(f'("{kp.annotation_type}",) + {ret}[1:]')
                else:
                    specialization.append(f"{ret}")

        # Generate dynamic function
        func_body = f"""
    def dynamic_func({", ".join(arg_names)}):
        params = {{{', '.join([f"'{name}': {name}" for name in sig.parameters.keys()])}}}
        specialization = [{','.join(specialization)}]
        return params, specialization, options
    """
        exec(func_body, func_namespace)
        return func_namespace['dynamic_func']

*Location:* `jit.py:393-448 <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/jit.py#L393-L448>`_

**What gets specialized?**

- ``tl.constexpr`` parameters (always)
- Integer alignment (if ``data_ptr() % 16 == 0``)
- Tensor shapes and dtypes
- Divisibility by 16

**Example:**

.. code-block:: python

    @triton.jit
    def kernel(ptr, N: tl.constexpr):
        pass

    kernel[grid](x, N=1024)  # Compiles with N=1024
    kernel[grid](x, N=2048)  # Recompiles with N=2048 (different binary!)
    kernel[grid](y, N=1024)  # Reuses first binary (cached)

Compilation Triggering
----------------------

If no cached kernel exists, ``JITFunction`` triggers compilation:

.. code-block:: python

    def run(self, *args, grid, **kwargs):
        # ... argument processing ...

        # Compute specialization
        params, specialization, options = specialization_fn(*args, **kwargs)

        # Compute cache key
        cache_key = compute_cache_key(self.kernel_key_cache, specialization, options)

        # Look up in cache
        if cache_key in self.cache:
            kernel = self.cache[cache_key]
        else:
            # COMPILE!
            from ..compiler import compile, ASTSource

            # Create source representation
            src = ASTSource(
                fn=self,
                signature=specialized_signature,
                constexprs=constexprs,
                attrs=attrs
            )

            # Trigger compilation
            kernel = compile(
                src=src,
                target=target,
                options=options
            )

            # Cache result
            self.cache[cache_key] = kernel

        # Launch kernel
        kernel.run(grid, ...)

Compilation creates an ``ASTSource`` object and passes it to the main ``compile()`` function (covered in next sections).

Kernel Metadata
---------------

Launch Metadata
~~~~~~~~~~~~~~~

You can attach metadata to kernels for profiling:

.. code-block:: python

    def _matmul_launch_metadata(grid, kernel, args):
        M, N, K = args["M"], args["N"], args["K"]
        return {
            "name": f"matmul [M={M}, N={N}, K={K}]",
            "flops": 2 * M * N * K,
            "bytes": 2 * (M*K + K*N + M*N)
        }

    @triton.jit(launch_metadata=_matmul_launch_metadata)
    def matmul_kernel(...):
        pass

This metadata is passed to profilers like Triton Proton.

Attributes and Hints
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @triton.jit(
        noinline=True,  # Prevent inlining
        debug=True,     # Enable debug mode
    )
    def kernel(...):
        pass

- ``noinline``: Forces function to be a separate call (not inlined)
- ``debug``: Enables debugging features (assertions, bounds checking)
- ``repr``: Custom repr() for the function

Constexpr Parameters
--------------------

``tl.constexpr`` marks compile-time constants:

.. code-block:: python

    @triton.jit
    def kernel(x, BLOCK_SIZE: tl.constexpr):
        # BLOCK_SIZE is known at compile time
        # Can be used for array bounds, loop limits, etc.
        for i in range(BLOCK_SIZE):  # Unrolled!
            pass

**Constexpr features:**

- Must be literal values or constexpr expressions
- Can be used in control flow
- Can be used for type annotations
- Enables loop unrolling and constant folding

**Implementation:**

.. code-block:: python

    from triton.language.core import constexpr

    class constexpr:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return f"constexpr[{self.value}]"

Constexpr values are stored separately from regular arguments and directly embedded in the generated code.

Summary
-------

The ``@triton.jit`` decorator:

1. **Wraps** the function in a ``JITFunction`` object
2. **Extracts** source code and parses it into AST
3. **Tracks** dependencies (global variables, called functions)
4. **Generates** cache keys for compiled kernels
5. **Specializes** on runtime values (constexpr, alignment, types)
6. **Triggers** compilation when needed
7. **Caches** compiled binaries
8. **Launches** kernels on the GPU

**Key files:**

- `jit.py <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/jit.py>`_ - Main JIT implementation
- `cache.py <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/runtime/cache.py>`_ - Kernel caching
- `compiler.py <https://github.com/triton-lang/triton/tree/v3.5.1/python/triton/compiler/compiler.py>`_ - Compilation orchestration

Next: :doc:`03-code-generation` - How Python AST becomes Triton IR
