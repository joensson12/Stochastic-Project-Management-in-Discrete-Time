"""LibreProject/MS Project XML interface for the time-simulation prototype."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

import networkx as nx

from spm.distributions import ShiftedPoissonDuration
from spm.project import Project


MICROSOFT_PROJECT_NAMESPACE = "http://schemas.microsoft.com/project"  # XML namespace used by MS Project files.
NAMESPACE = {"msp": MICROSOFT_PROJECT_NAMESPACE}  # Prefix map for ElementTree XPath queries.
MIN_NOTE_PATTERN = re.compile(r"\bmin\s*:\s*(\d+)\b", re.IGNORECASE)  # Parses notes like "Min: 1".


@dataclass(frozen=True)
class ImportedTask:
    """Human-facing task metadata kept outside the mathematical simulator."""

    uid: int  # Original task UID from the project file; this becomes the simulator activity ID i.
    project_id: int  # Original visible project ID, kept only for interface/reference use.
    name: str  # Human-readable task name; not used by the simulator math.
    estimated_duration: int  # Estimated duration, interpreted as the smallest mode of D_i.
    minimum_duration: int  # a_i from the note field "Min: a_i".
    lambda_: float  # Prototype lambda_i chosen so estimated_duration is the smallest mode.
    critical: bool  # Critical flag from the planning file, used only for display.


@dataclass(frozen=True)
class ImportedProject:
    """Result of importing a planning file into the simulator-facing objects."""

    name: str  # Project title/name from the external file.
    simulator_project: Project  # Pure mathematical simulator object.
    tasks: dict[int, ImportedTask]  # Interface metadata keyed by task UID/activity ID i.
    network_graph: nx.DiGraph  # Interface copy of the imported schedule graph S.

    @property
    def task_names(self) -> dict[int, str]:
        """Return labels i -> task name for notebook display."""
        return {uid: task.name for uid, task in self.tasks.items()}  # UI labels stay outside Project.

    @property
    def graph(self) -> nx.DiGraph:
        """Expose the imported network graph for notebook visualization."""
        return self.network_graph  # The DAG S from the external planning file.


def load_project_from_xml(
    file_path: str | Path,
    *,
    sample_count: int | None = None,
    rng_seed: int | None = None,
) -> ImportedProject:
    """Load a LibreProject/MS Project XML file as a shifted-Poisson prototype.

    This is intentionally a narrow interface prototype. It assumes:
    - each imported work package has notes of the form "Min: a_i";
    - the scheduled duration is the smallest mode of D_i;
    - D_i follows ShiftedPoi(lambda_i, a_i);
    - dependency links are zero-lag finish-to-start edges.
    """
    imported = read_project_xml(file_path)  # Parse human-facing metadata and prototype parameters.
    project = Project(sample_count=sample_count, rng_seed=rng_seed)  # Create the pure simulator object.

    for task in imported.tasks.values():
        project.add_work_package(  # Add activity i with D_i ~ ShiftedPoi(lambda_i, a_i).
            task.uid,  # Activity ID i.
            ShiftedPoissonDuration(lambda_=task.lambda_, a=task.minimum_duration),  # Prototype law for D_i.
        )

    for predecessor, successor in imported.graph.edges:
        project.add_dependency(predecessor, successor)  # Encode predecessor in Pred(successor).

    return ImportedProject(
        name=imported.name,  # External project name.
        simulator_project=project,  # Mathematical Project object.
        tasks=imported.tasks,  # Notebook/interface metadata.
        network_graph=project.graph,  # Same DAG S, exposed for interface visualization.
    )


def read_project_xml(file_path: str | Path) -> ImportedProject:
    """Read task metadata and dependencies from LibreProject/MS Project XML.

    The returned ImportedProject contains a graph and task metadata, but the
    simulator_project field is an empty Project. Use load_project_from_xml when
    you want the simulator Project populated with shifted-Poisson durations.
    """
    path = Path(file_path)  # Accept both notebook strings and Path objects.
    root = ET.parse(path).getroot()  # Parse the XML document.
    project_name = _text(root, "msp:Name", default=path.stem)  # Project name for display.
    minutes_per_day = _int_text(root, "msp:MinutesPerDay", default=480)  # Converts work duration to days.

    tasks: dict[int, ImportedTask] = {}  # Metadata keyed by UID/activity ID i.
    graph = nx.DiGraph()  # External schedule graph S before it is loaded into Project.

    for task_element in root.findall("msp:Tasks/msp:Task", NAMESPACE):
        if not _is_importable_work_package(task_element):
            continue  # Skip null, summary, resource, and milestone rows in the external file.

        uid = _required_int(task_element, "msp:UID")  # Task UID becomes activity ID i.
        project_id = _int_text(task_element, "msp:ID", default=uid)  # Visible row ID from the planning tool.
        name = _text(task_element, "msp:Name", default=f"Task {uid}")  # Human label for notebook graphs.
        duration_text = _required_text(task_element, "msp:Duration")  # Scheduled estimated duration.
        notes = _required_text(task_element, "msp:Notes")  # Must contain "Min: a_i" in this prototype.

        estimated_duration = _duration_to_project_days(duration_text, minutes_per_day)  # Smallest mode estimate.
        minimum_duration = _minimum_duration_from_notes(notes)  # a_i from Notes.
        lambda_ = _lambda_from_smallest_mode(estimated_duration, minimum_duration)  # lambda_i.
        critical = _int_text(task_element, "msp:Critical", default=0) == 1  # Planning-tool critical flag.

        tasks[uid] = ImportedTask(
            uid=uid,  # Activity ID i.
            project_id=project_id,  # External row ID.
            name=name,  # Human-readable label.
            estimated_duration=estimated_duration,  # Smallest modal value of D_i.
            minimum_duration=minimum_duration,  # a_i.
            lambda_=lambda_,  # Prototype Poisson intensity.
            critical=critical,  # Critical/noncritical display metadata.
        )
        graph.add_node(uid)  # Add activity i to S.

    for task_element in root.findall("msp:Tasks/msp:Task", NAMESPACE):
        if not _is_importable_work_package(task_element):
            continue  # Only add edges between imported work packages.

        successor = _required_int(task_element, "msp:UID")  # Current task is successor i.
        if successor not in tasks:
            continue  # Defensive skip if task was excluded.
        for link in task_element.findall("msp:PredecessorLink", NAMESPACE):
            predecessor = _required_int(link, "msp:PredecessorUID")  # j in Pred(i).
            if predecessor in tasks:
                graph.add_edge(predecessor, successor)  # Encode j -> i.

    return ImportedProject(
        name=project_name,  # Display name.
        simulator_project=Project(),  # Empty placeholder; use load_project_from_xml for simulation.
        tasks=tasks,  # Parsed metadata.
        network_graph=graph,  # Parsed schedule graph S.
    )


def draw_project_network(
    imported_project: ImportedProject,
    *,
    ax=None,
    layout: str = "multipartite",
    with_names: bool = True,
    show_mode: bool = True,
):
    """Draw the imported project network as LibreProject-like task boxes.

    Matplotlib is imported only inside this function so the simulator and parser
    do not depend on notebook visualization packages.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.patches import Rectangle

    graph = imported_project.graph  # DAG S.
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))  # Wide planning-chart shape for notebooks.

    positions = _network_positions(graph, layout=layout)  # Coordinates for task-box centers.
    box_width = 1.65  # Width of each work-package rectangle.
    box_height = 0.62 if show_mode else 0.42  # Extra room for the mode line.

    for predecessor, successor in graph.edges:
        start = positions[predecessor]  # Tail activity j.
        end = positions[successor]  # Head activity i.
        arrow = FancyArrowPatch(
            (start[0] + box_width / 2, start[1]),  # Leave the right side of box j.
            (end[0] - box_width / 2, end[1]),  # Enter the left side of box i.
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color="black",
            shrinkA=2,
            shrinkB=2,
            connectionstyle="angle3,angleA=0,angleB=90",
        )
        ax.add_patch(arrow)  # Draw precedence relation j -> i.

    for node in graph.nodes:
        task = imported_project.tasks[node]  # Interface metadata for activity i.
        x, y = positions[node]  # Center of the task box.
        edge_color = "red" if task.critical else "blue"  # Planning-tool critical/noncritical colors.
        rectangle = Rectangle(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            facecolor="white",
            edgecolor=edge_color,
            linewidth=2.0,
        )
        ax.add_patch(rectangle)  # Draw work-package box.

        title = task.name if with_names else str(node)  # Human name or mathematical ID i.
        label = f"{title}\nmode: {task.estimated_duration}" if show_mode else title  # Display smallest mode.
        ax.text(
            x - box_width / 2 + 0.08,
            y,
            label,
            ha="left",
            va="center",
            fontsize=8,
            family="monospace",
        )  # Text inside the task box.

    ax.set_title(imported_project.name)  # External project name.
    ax.axis("off")  # Graph view, not a coordinate plot.
    ax.autoscale_view()  # Include manually drawn boxes and arrows.
    return ax  # Allows notebook users to keep customizing the figure.


def project_diagnostics_text(imported_project: ImportedProject) -> str:
    """Return a text table of imported parameters and predecessors."""
    lines = [f"Project: {imported_project.name}", ""]  # Notebook-readable heading.
    lines.append("id | task | predecessors | distribution | a_i | lambda_i | modes | E[D_i]")
    lines.append("-" * 88)

    for node in imported_project.simulator_project.topological_order():
        task = imported_project.tasks[node]  # Human-facing metadata for activity i.
        model = imported_project.simulator_project.work_packages[node].duration_model  # Distribution of D_i.
        predecessors = sorted(imported_project.graph.predecessors(node))  # Pred(i).
        predecessor_text = ", ".join(str(pred) for pred in predecessors) or "empty"  # Empty predecessor set.

        if isinstance(model, ShiftedPoissonDuration):
            distribution = f"ShiftedPoi(lambda={model.lambda_:.3g}, a={model.a})"  # D_i law.
            modes = ", ".join(str(mode) for mode in model.modes)  # Modal values of D_i.
            expected = f"{model.expected_value:.3g}"  # E[D_i].
        else:
            distribution = type(model).__name__  # Future non-Poisson distribution name.
            modes = "?"  # Unknown modal values for generic distributions.
            expected = "?"  # Unknown expected value for generic distributions.

        lines.append(
            f"{node} | {task.name} | {predecessor_text} | {distribution} | "
            f"{task.minimum_duration} | {task.lambda_:.3g} | {modes} | {expected}"
        )

    return "\n".join(lines)  # Plain text prints cleanly in notebooks.


def print_project_diagnostics(imported_project: ImportedProject) -> None:
    """Print work-package distributions, parameters, and predecessor sets."""
    print(project_diagnostics_text(imported_project))  # Notebook-friendly diagnostic output.


def _is_importable_work_package(task_element: ET.Element) -> bool:
    """Return whether an XML task should become a simulator activity."""
    uid = _int_text(task_element, "msp:UID", default=0)  # UID 0 is often the unassigned/resource row.
    if uid <= 0:
        return False  # Project simulator activity IDs are positive.
    if _int_text(task_element, "msp:IsNull", default=0) == 1:
        return False  # Null rows are not mathematical activities.
    if _int_text(task_element, "msp:Summary", default=0) == 1:
        return False  # Summary tasks aggregate work packages; they are not D_i here.
    if _int_text(task_element, "msp:Milestone", default=0) == 1:
        return False  # Milestones are zero-duration schedule markers, not work packages in this prototype.
    return True  # Treat as work package/activity i.


def _duration_to_project_days(duration_text: str, minutes_per_day: int) -> int:
    """Convert an MS Project duration like PT16H0M0S into integer project days."""
    total_minutes = _duration_to_minutes(duration_text)  # Work duration in minutes.
    if total_minutes < 0:
        raise ValueError("duration cannot be negative.")
    return ceil(total_minutes / minutes_per_day)  # Discrete-time activity estimate.


def _duration_to_minutes(duration_text: str) -> int:
    """Parse the subset of ISO-8601 durations used by MS Project XML."""
    pattern = re.compile(
        r"^P"
        r"(?:(?P<days>\d+)D)?"
        r"(?:T"
        r"(?:(?P<hours>\d+)H)?"
        r"(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+)S)?"
        r")?$"
    )
    match = pattern.match(duration_text.strip())  # Example: PT16H0M0S.
    if match is None:
        raise ValueError(f"unsupported duration format: {duration_text!r}.")

    days = int(match.group("days") or 0)  # Calendar days if present.
    hours = int(match.group("hours") or 0)  # Hours component.
    minutes = int(match.group("minutes") or 0)  # Minutes component.
    seconds = int(match.group("seconds") or 0)  # Seconds component.
    return days * 24 * 60 + hours * 60 + minutes + ceil(seconds / 60)  # Round seconds up to a minute.


def _minimum_duration_from_notes(notes: str) -> int:
    """Extract a_i from notes of the form 'Min: a_i'."""
    match = MIN_NOTE_PATTERN.search(notes)  # Look for "Min: 1" anywhere in the notes text.
    if match is None:
        raise ValueError(f"task notes must contain 'Min: a_i', got {notes!r}.")
    return int(match.group(1))  # a_i in N_0.


def _lambda_from_smallest_mode(estimated_duration: int, minimum_duration: int) -> float:
    """Choose lambda_i so estimated_duration is the smallest mode of D_i.

    For integer lambda_i, the shifted Poisson has modes
    a_i + lambda_i - 1 and a_i + lambda_i. Therefore choosing
    lambda_i = estimated_duration - a_i + 1 makes the planned duration the
    smallest modal value. This is a specific prototype convention.
    """
    if estimated_duration < minimum_duration:
        raise ValueError("estimated duration must be at least the minimum duration.")
    return float(estimated_duration - minimum_duration + 1)  # Positive integer lambda_i.


def _network_positions(graph: nx.DiGraph, *, layout: str) -> dict[int, tuple[float, float]]:
    """Return node positions suitable for notebook drawing."""
    if layout == "multipartite":
        generations = list(nx.topological_generations(graph))  # Layers in the DAG.
        positions: dict[int, tuple[float, float]] = {}  # Centers of task boxes.
        for layer, generation in enumerate(generations):
            ordered_generation = sorted(generation)  # Deterministic vertical order in each layer.
            offset = (len(ordered_generation) - 1) / 2  # Center stacked boxes vertically.
            for row, node in enumerate(ordered_generation):
                positions[node] = (layer * 2.1, -(row - offset) * 1.05)  # x=layer, y=stacked tasks.
        return positions  # LibreProject-like left-to-right DAG layout.
    if layout == "spring":
        raw_positions: dict[Any, Any] = nx.spring_layout(graph, seed=1)  # General-purpose fallback layout.
        return {
            int(node): (float(position[0]) * 8, float(position[1]) * 4)
            for node, position in raw_positions.items()
        }  # Scale spring coordinates for rectangular boxes.
    raise ValueError("layout must be 'multipartite' or 'spring'.")


def _text(element: ET.Element, path: str, *, default: str) -> str:
    """Read optional text from an XML child."""
    child = element.find(path, NAMESPACE)  # Namespace-aware lookup.
    if child is None or child.text is None:
        return default  # Use caller default when absent.
    return child.text.strip()  # Trim whitespace common in XML files.


def _required_text(element: ET.Element, path: str) -> str:
    """Read required text from an XML child."""
    child = element.find(path, NAMESPACE)  # Namespace-aware lookup.
    if child is None or child.text is None:
        raise ValueError(f"missing required XML field {path}.")
    return child.text.strip()  # Trim whitespace common in XML files.


def _int_text(element: ET.Element, path: str, *, default: int) -> int:
    """Read optional integer text from an XML child."""
    text = _text(element, path, default=str(default))  # Reuse optional text reader.
    return int(text)  # Convert to integer field.


def _required_int(element: ET.Element, path: str) -> int:
    """Read required integer text from an XML child."""
    return int(_required_text(element, path))  # Convert required XML field to int.
