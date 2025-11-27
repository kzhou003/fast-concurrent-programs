#!/usr/bin/env python3
"""
Convert basics markdown files to RST format.
"""

import re
from pathlib import Path

def convert_headers(text):
    """Convert markdown headers to RST format."""
    text = re.sub(r'^# (.+)$', lambda m: f"{m.group(1)}\n{'=' * len(m.group(1))}", text, flags=re.M)
    text = re.sub(r'^## (.+)$', lambda m: f"{m.group(1)}\n{'-' * len(m.group(1))}", text, flags=re.M)
    text = re.sub(r'^### (.+)$', lambda m: f"{m.group(1)}\n{'~' * len(m.group(1))}", text, flags=re.M)
    text = re.sub(r'^#### (.+)$', lambda m: f"{m.group(1)}\n{'^' * len(m.group(1))}", text, flags=re.M)
    return text

def convert_code_blocks(text):
    """Convert markdown code blocks to RST format."""
    text = re.sub(r'```(\w+)\n', r'.. code-block:: \1\n\n', text)
    text = re.sub(r'```\n', r'::\n\n', text)
    text = re.sub(r'```$', '', text, flags=re.M)
    text = re.sub(r'`([^`]+)`', r'``\1``', text)
    return text

def convert_bold_italic(text):
    """Convert markdown bold/italic to RST format."""
    text = re.sub(r'_([^_]+)_', r'*\1*', text)
    return text

def convert_links(text):
    """Convert markdown links to RST format."""
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'`\1 <\2>`_', text)
    return text

def convert_markdown_to_rst(md_content):
    """Main conversion function."""
    rst_content = md_content
    rst_content = convert_headers(rst_content)
    rst_content = convert_code_blocks(rst_content)
    rst_content = convert_bold_italic(rst_content)
    rst_content = convert_links(rst_content)
    return rst_content

def main():
    """Convert all markdown files in basics directory."""
    basics_dir = Path('/home/kzhoulatte/Experiments/fast-concurrent-programs/docs/basics')
    output_dir = Path('/home/kzhoulatte/Experiments/fast-concurrent-programs/docs/source/cpu-concurrency')

    for md_file in sorted(basics_dir.glob('*.md')):
        print(f"Converting {md_file.name}...")

        md_content = md_file.read_text()
        rst_content = convert_markdown_to_rst(md_content)

        rst_file = output_dir / md_file.name.replace('.md', '.rst')
        rst_file.write_text(rst_content)

        print(f"  -> {rst_file.name}")

if __name__ == '__main__':
    main()
