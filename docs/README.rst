Fast Concurrent Programming Guide
===================================

A comprehensive guide to concurrent and parallel programming, covering both CPU-based concurrency
(threading, asyncio, multiprocessing) and GPU-based parallelism (Triton, CUDA).

Building Locally
----------------

Install requirements::

    pip install -r requirements.txt

Build HTML documentation::

    make html

View the built documentation::

    open build/html/index.html  # macOS
    xdg-open build/html/index.html  # Linux
    start build/html/index.html  # Windows

Other build formats::

    make latexpdf  # PDF
    make epub      # ePub
    make clean     # Clean build files

Structure
---------

::

    docs/
    ├── source/
    │   ├── conf.py                 # Sphinx configuration
    │   ├── index.rst               # Main entry point
    │   ├── concepts/               # Core GPU/Triton concepts
    │   │   ├── gpu-fundamentals.rst
    │   │   ├── memory-hierarchy.rst
    │   │   ├── execution-model.rst
    │   │   └── performance-optimization.rst
    │   ├── tutorials/              # Step-by-step tutorials
    │   │   ├── 01-vector-add.rst
    │   │   ├── 02-fused-softmax.rst
    │   │   ├── 03-matrix-multiplication.rst
    │   │   ├── 04-low-memory-dropout.rst
    │   │   ├── 05-layer-norm.rst
    │   │   ├── 06-fused-attention.rst
    │   │   └── 07-extern-functions.rst
    │   ├── learning-paths.rst      # Curated learning sequences
    │   ├── troubleshooting.rst     # Common issues and solutions
    │   └── references.rst          # Papers, tools, resources
    ├── triton/                     # Original markdown files
    ├── requirements.txt            # Python dependencies
    ├── Makefile                    # Build commands
    └── README.rst                  # This file

ReadTheDocs Integration
-----------------------

This documentation is configured for ReadTheDocs deployment.

Configuration file: ``.readthedocs.yaml`` (in repository root)

To deploy:

1. Push to GitHub/GitLab
2. Connect repository to ReadTheDocs
3. Documentation builds automatically on commits

Local preview matches ReadTheDocs output.

Content Organization
--------------------

**Concepts** - Foundational knowledge

* GPU architecture and execution model
* Memory hierarchy and optimization
* Performance analysis

**Tutorials** - Hands-on learning

* Beginner: Vector add, fused softmax
* Intermediate: Matrix multiplication, dropout
* Advanced: Layer norm, Flash Attention

**Resources**

* Learning paths for different goals
* Troubleshooting guide
* References and links

Contributing
------------

To add or modify documentation:

1. Edit RST files in ``source/``
2. Build locally to test: ``make html``
3. Check for warnings: ``make html`` output
4. Preview: Open ``build/html/index.html``
5. Commit changes

RST Syntax Reference
~~~~~~~~~~~~~~~~~~~~

Headers::

    Title
    =====

    Section
    -------

    Subsection
    ~~~~~~~~~~

    Subsubsection
    ^^^^^^^^^^^^^

Code blocks::

    .. code-block:: python

        def example():
            pass

Inline code::

    Use ``code`` for inline code.

Links::

    `Link text <https://example.com>`_
    :doc:`other-document`

Lists::

    * Bullet point
    * Another point

    1. Numbered
    2. List

Tables::

    .. list-table::
       :header-rows: 1

       * - Column 1
         - Column 2
       * - Data 1
         - Data 2

More: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

License
-------

Documentation licensed under CC-BY-4.0.
Code examples follow Triton's MIT license.
