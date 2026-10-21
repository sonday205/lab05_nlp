import nbformat
from nbclient import NotebookClient

from preprocessing import ROOT


path = ROOT / "sequence_classification.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code_cells)
assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
print(f"Executed {len(code_cells)} code cells; 0 errors: {path}")
