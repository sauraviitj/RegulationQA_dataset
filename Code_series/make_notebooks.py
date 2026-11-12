"""Converts the numbered `# %%` .py step files into .ipynb notebooks (no extra packages needed).
Usage: python make_notebooks.py        -> writes notebooks/NN_name.ipynb
Rules: `# %% [markdown]` blocks become markdown cells (leading '# ' removed); `#!pip ...` becomes a real `!pip ...` line;
a bootstrap cell is added first so `from adsir_common import *` works on Kaggle when adsir_common.py is in an added dataset."""
import json, re, glob, os
BOOT = ("import sys, glob\n"
        "# make adsir_common.py importable: current folder, or any Kaggle input dataset that contains it\n"
        "for p in ['.'] + [os.path.dirname(x) for x in glob.glob('/kaggle/input/**/adsir_common.py', recursive=True)]:\n"
        "    if p not in sys.path: sys.path.insert(0, p)\n")
os.makedirs("notebooks", exist_ok=True)
def cell(kind, lines):
    src = "\n".join(lines).strip("\n")
    base = {"cell_type": kind, "metadata": {}, "source": src.splitlines(True)}
    return {**base, "outputs": [], "execution_count": None} if kind == "code" else base
for f in sorted(glob.glob("[0-9][0-9]_*.py")):
    parts, cur, kind = [], [], None
    for line in open(f, encoding="utf-8").read().splitlines():
        m = re.match(r"# %%( \[markdown\])?\s*$", line)
        if m:
            if kind: parts.append((kind, cur))
            kind, cur = ("markdown" if m.group(1) else "code"), []
        else:
            cur.append(line)
    parts.append((kind, cur))
    cells = [cell("code", ["import os"] + BOOT.splitlines())]
    for k, ls in parts:
        if k == "markdown": ls = [re.sub(r"^# ?", "", l) for l in ls]
        else: ls = [re.sub(r"^#!pip", "!pip", l) for l in ls]
        cells.append(cell(k, ls))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join("notebooks", f.replace(".py", ".ipynb")); json.dump(nb, open(out, "w"), indent=1); print("wrote", out, len(cells), "cells")
