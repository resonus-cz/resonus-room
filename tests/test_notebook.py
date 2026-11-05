from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = Path(__file__).parent.parent / "examples" / "resonus_room.ipynb"


def test_example_notebook_runs():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(nb, timeout=120, kernel_name="python3").execute()
    outputs = [o for cell in nb.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    assert any("image/png" in o.get("data", {}) for o in outputs)
