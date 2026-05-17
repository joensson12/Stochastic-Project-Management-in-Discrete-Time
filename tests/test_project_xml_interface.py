from pathlib import Path

import pytest

from spm.distributions import ShiftedPoissonDuration
from spm.interfaces import load_project_from_xml, project_diagnostics_text, read_project_xml


EXAMPLE_FILE = Path("Example project") / "Building Garage.xml"


def test_read_project_xml_extracts_tasks_metadata_and_edges() -> None:
    imported = read_project_xml(EXAMPLE_FILE)

    assert imported.name == "Building Garage"
    assert len(imported.tasks) == 11
    assert imported.tasks[1].name == "Prepare site"
    assert imported.tasks[1].estimated_duration == 2
    assert imported.tasks[1].minimum_duration == 1
    assert imported.tasks[1].lambda_ == 2.0
    assert imported.tasks[3].estimated_duration == 4
    assert imported.tasks[3].minimum_duration == 2
    assert imported.tasks[3].lambda_ == 3.0
    assert imported.tasks[3].critical is True
    assert imported.graph.has_edge(1, 2)
    assert imported.graph.has_edge(4, 8)


def test_load_project_from_xml_builds_shifted_poisson_simulator_project() -> None:
    imported = load_project_from_xml(EXAMPLE_FILE, sample_count=5, rng_seed=123)
    project = imported.simulator_project

    assert set(project.work_packages) == set(imported.tasks)
    assert project.topological_order()[0] == 1
    assert isinstance(project.work_packages[3].duration_model, ShiftedPoissonDuration)
    assert project.work_packages[3].duration_model.a == 2
    assert project.work_packages[3].duration_model.lambda_ == 3.0
    assert project.graph.has_edge(10, 11)


def test_imported_project_can_run_time_simulation() -> None:
    imported = load_project_from_xml(EXAMPLE_FILE, sample_count=10, rng_seed=123)

    result = imported.simulator_project.simulate_project_time()

    assert result.completion_times.shape == (10,)
    assert len(result.critical_paths) == 10


def test_project_diagnostics_text_lists_parameters_distributions_and_predecessors() -> None:
    imported = load_project_from_xml(EXAMPLE_FILE, sample_count=10, rng_seed=123)

    diagnostics = project_diagnostics_text(imported)

    assert "ShiftedPoi(lambda=2, a=1)" in diagnostics
    assert "Prepare site | empty" in diagnostics
    assert "Erect frame | 2 | ShiftedPoi(lambda=3, a=2)" in diagnostics
    assert "modes" in diagnostics


def test_load_project_from_xml_rejects_missing_min_note(tmp_path: Path) -> None:
    xml_file = tmp_path / "missing-min.xml"
    xml_file.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<Project xmlns="http://schemas.microsoft.com/project">
  <Name>Bad Project</Name>
  <MinutesPerDay>480</MinutesPerDay>
  <Tasks>
    <Task>
      <UID>1</UID>
      <ID>1</ID>
      <Name>Task without minimum</Name>
      <IsNull>0</IsNull>
      <Summary>0</Summary>
      <Milestone>0</Milestone>
      <Duration>PT8H0M0S</Duration>
      <Notes>No minimum here</Notes>
    </Task>
  </Tasks>
</Project>
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Min"):
        load_project_from_xml(xml_file)
