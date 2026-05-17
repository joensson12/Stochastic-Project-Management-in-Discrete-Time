# SPM

Prototype object-oriented framework for stochastic project-management time simulation.

The first implementation focuses on integer work packages, shifted Poisson duration
models, and vectorized NetworkX-based project completion simulation.

## LibreProject/MS Project XML Prototype

Notebook-facing import functions live in `spm.interfaces`, separate from the
simulator core.

```python
from pathlib import Path
import sys

cwd = Path.cwd().resolve()
repo_root = next(path for path in [cwd, *cwd.parents] if (path / "src" / "spm").exists())
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from spm.interfaces import (
    draw_project_network,
    load_project_from_xml,
    print_project_diagnostics,
)

imported = load_project_from_xml(
    repo_root / "Example project" / "Building Garage.pod.xml",
    sample_count=1000,
    rng_seed=1,
)

draw_project_network(imported)
print_project_diagnostics(imported)
result = imported.simulator_project.simulate_project_time()
```

This importer is a specific prototype convention: it reads the task duration as
the smallest mode and reads `a_i` from task notes of the form `Min: a_i`.
