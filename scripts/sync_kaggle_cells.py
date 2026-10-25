"""Pull the latest saved Kaggle notebook and write its cells to notebooks/kaggle_cells.py.

Usage:  python scripts/sync_kaggle_cells.py [owner/notebook]
Needs the Kaggle CLI and a token in ~/.kaggle/access_token. Commit the result:
git history of notebooks/kaggle_cells.py is then the notebook's version history.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

NOTEBOOK = sys.argv[1] if len(sys.argv) > 1 else "zyanislam/oct-calibration"
OUT = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_cells.py"


def to_percent(nb: dict) -> str:
    lines = [
        "# ---",
        f"# Kaggle notebook: {NOTEBOOK}",
        f"# https://www.kaggle.com/code/{NOTEBOOK}",
        f"# Synced from the latest saved Kaggle version on {datetime.date.today():%a %-d %b %Y}.",
        "# Every code and markdown cell, in order. Git history of this file = the notebook's history:",
        "#   git log -p notebooks/kaggle_cells.py",
        '# Opens as a notebook in VS Code / JupyterLab ("# %%" marks a cell).',
        "# ---",
        "",
    ]
    n = 0
    for cell in nb["cells"]:
        src = "".join(cell["source"]).rstrip()
        if cell["cell_type"] == "markdown":
            lines.append("# %% [markdown]")
            lines += ["# " + line if line else "#" for line in src.splitlines()]
        else:
            n += 1
            lines.append(f"# %%  [code cell {n}]")
            lines.append(src)
        lines.append("")
    return "\n".join(lines)


with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["kaggle", "kernels", "pull", NOTEBOOK, "-p", tmp], check=True, capture_output=True)
    nb = json.loads(next(Path(tmp).glob("*.ipynb")).read_text())

OUT.write_text(to_percent(nb))
print(f"{len(nb['cells'])} cells -> {OUT}")
