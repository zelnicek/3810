#!/usr/bin/env python3
"""
One-shot patch for waveform_catalog_evolved.py produced by the current
waveform_evolver.py.

The evolver writes EVOLVED_META using JSON syntax (true/false/null), but
Python expects True/False/None when importing the file. This script
fixes that in-place.

USAGE
-----
  python3 patch_evolved_catalog.py path/to/waveform_catalog_evolved.py

After running this once, the catalog is importable. To prevent the
problem permanently, see PATCH_NOTES_FOR_EVOLVER.txt for the fix to
apply to waveform_evolver.py itself.
"""
import re
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("usage: patch_evolved_catalog.py <path_to_evolved_catalog.py>")
    sys.exit(1)

target = Path(sys.argv[1])
if not target.exists():
    print(f"[ERROR] {target} not found")
    sys.exit(1)

with open(target) as f:
    src = f.read()

m_start = src.find('EVOLVED_META = {')
m_end = src.find('SHAPES = {')
if m_start < 0 or m_end < 0 or m_start > m_end:
    print("[ERROR] couldn't locate EVOLVED_META block — is this an evolved catalog?")
    sys.exit(1)

before = src[:m_start]
meta = src[m_start:m_end]
after = src[m_end:]

n_true  = len(re.findall(r'\btrue\b', meta))
n_false = len(re.findall(r'\bfalse\b', meta))
n_null  = len(re.findall(r'\bnull\b', meta))

meta = re.sub(r'\btrue\b',  'True',  meta)
meta = re.sub(r'\bfalse\b', 'False', meta)
meta = re.sub(r'\bnull\b',  'None',  meta)

# Backup, then write
backup = target.with_suffix('.py.bak')
with open(backup, 'w') as f:
    f.write(src)
with open(target, 'w') as f:
    f.write(before + meta + after)

print(f"✓ Patched {target}")
print(f"  Backup at: {backup}")
print(f"  Replaced: true→True ({n_true}x), false→False ({n_false}x), null→None ({n_null}x)")
print()
print("Verify by running:")
print(f"  python3 -c 'import importlib.util; "
      f"s = importlib.util.spec_from_file_location(\"c\", \"{target}\"); "
      f"m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
      f"print(\"OK,\", len(m.SHAPES), \"shapes\")'")