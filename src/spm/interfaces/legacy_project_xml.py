"""Legacy XML interface kept for older notebooks.

The current notebook-facing interface is ``spm.interfaces.project_excel``.
This module intentionally re-exports the older XML helpers so existing
experiments can migrate gradually.
"""

from spm.interfaces.project_xml import (
    ImportedProject,
    ImportedTask,
    draw_project_network,
    load_project_from_xml,
    project_diagnostics_text,
    print_project_diagnostics,
    read_project_xml,
)

__all__ = [
    "ImportedProject",
    "ImportedTask",
    "draw_project_network",
    "load_project_from_xml",
    "project_diagnostics_text",
    "print_project_diagnostics",
    "read_project_xml",
]
