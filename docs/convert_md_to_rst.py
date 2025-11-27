#!/usr/bin/env python3
"""
Convert markdown tutorial files to RST format.
This script helps automate the conversion of markdown to restructuredtext.
"""

import re
from pathlib import Path

def convert_headers(text):
    """Convert markdown headers to RST format."""
    # H1: # Title -> Title\n=====
    text = re.sub(r'^# (.+)$', lambda m: f"{m.group(1)}\n{'=' * len(m.group(1))}", text, flags=re.M)

    # H2: ## Title -> Title\n-----
    text = re.sub(r'^## (.+)$', lambda m: f"{m.group(1)}\n{'-' * len(m.group(1))}", text, flags=re.M)

    # H3: ### Title -> Title\n~~~~~
    text = re.sub(r'^### (.+)$', lambda m: f"{m.group(1)}\n{'~' * len(m.group(1))}", text, flags=re.M)

    # H4: #### Title -> Title\n^^^^^
    text = re.sub(r'^#### (.+)$', lambda m: f"{m.group(1)}\n{'^' * len(m.group(1))}", text, flags=re.M)

    return text

def convert_code_blocks(text):
    """Convert markdown code blocks to RST format."""
    # ```python -> .. code-block:: python
    text = re.sub(r'```(\w+)\n', r'.. code-block:: \1\n\n', text)
    text = re.sub(r'```\n', r'::\n\n', text)  # Generic code blocks
    text = re.sub(r'```$', '', text, flags=re.M)

    # Inline code: `code` -> ``code``
    text = re.sub(r'`([^`]+)`', r'``\1``', text)

    return text

def convert_lists(text):
    """Convert markdown lists to RST format."""
    # Unordered lists are similar, but need proper indentation
    # Ordered lists: 1. -> 1.
    return text

def convert_bold_italic(text):
    """Convert markdown bold/italic to RST format."""
    # **bold** -> **bold** (same in RST)
    # *italic* -> *italic* (same in RST, but check for conflicts)

    # Fix: _italic_ -> *italic*
    text = re.sub(r'_([^_]+)_', r'*\1*', text)

    return text

def convert_links(text):
    """Convert markdown links to RST format."""
    # [text](url) -> `text <url>`_
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'`\1 <\2>`_', text)

    return text

def convert_tables(text):
    """Convert markdown tables to RST format (basic)."""
    # This is complex, leaving as-is for manual conversion
    return text

def convert_markdown_to_rst(md_content):
    """Main conversion function."""
    rst_content = md_content

    # Apply conversions in order
    rst_content = convert_headers(rst_content)
    rst_content = convert_code_blocks(rst_content)
    rst_content = convert_bold_italic(rst_content)
    rst_content = convert_links(rst_content)

    return rst_content

def main():
    """Convert all markdown files in triton directory."""
    triton_dir = Path('/home/kzhoulatte/Experiments/fast-concurrent-programs/docs/triton')
    output_dir = Path('/home/kzhoulatte/Experiments/fast-concurrent-programs/docs/source/gpu-tutorials')

    for md_file in sorted(triton_dir.glob('*.md')):
        if md_file.name == 'README.md':
            continue

        print(f"Converting {md_file.name}...")

        # Read markdown content
        md_content = md_file.read_text()

        # Convert to RST
        rst_content = convert_markdown_to_rst(md_content)

        # Write RST file
        rst_file = output_dir / md_file.name.replace('.md', '.rst')
        rst_file.write_text(rst_content)

        print(f"  -> {rst_file.name}")

if __name__ == '__main__':
    main()