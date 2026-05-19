"""Final notebook-facing Excel interface for the simulation framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import networkx as nx

from spm.distributions import (
    DeterministicContinuousDistribution,
    DiscreteUniformDuration,
    ShiftedBinomialDuration,
    UniformContinuousDistribution,
)
from spm.project import Project
from spm.work_package import CostRisk, ScheduleRisk


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"main": MAIN_NS}


@dataclass(frozen=True)
class ImportedExcelActivity:
    """Notebook-facing metadata for one activity imported from the workbook."""

    activity_id: str  # External activity label, for example A12.
    node_id: int  # Simulator activity ID i.
    phase: str | None  # Optional source phase label.
    predecessor_ids: tuple[int, ...]  # Parsed predecessor activity IDs.
    raw: dict[str, object]  # Original row values for notebook diagnostics.


@dataclass(frozen=True)
class ImportedExcelRisk:
    """Notebook-facing metadata for one risk imported from the workbook."""

    risk_id: str  # External risk label, for example R4.
    risk_type: str  # Schedule or Cost.
    affected_node_id: int  # Activity ID i affected by this risk.
    raw: dict[str, object]  # Original row values for notebook diagnostics.


@dataclass(frozen=True)
class ImportedExcelProject:
    """Workbook import result containing metadata and the simulator project."""

    name: str  # Workbook/project display name.
    simulator_project: Project  # Pure simulation object.
    activities: dict[int, ImportedExcelActivity]  # Activity metadata keyed by simulator ID i.
    risks: dict[str, ImportedExcelRisk]  # Risk metadata keyed by risk ID.
    network_graph: nx.DiGraph  # Interface copy of the imported schedule graph.

    @property
    def graph(self) -> nx.DiGraph:
        """Expose the imported network graph for notebook visualization."""
        return self.network_graph


def load_project_from_excel(
    file_path: str | Path,
    *,
    sample_count: int | None = None,
    rng_seed: int | None = None,
) -> ImportedExcelProject:
    """Load a cleaned Activities/Risks workbook into the simulation framework."""
    path = Path(file_path)  # Accept notebook strings and Path objects.
    activities_rows = _read_xlsx_table(path, "Activities")  # Clean activity table.
    risk_rows = _read_xlsx_table(path, "Risks")  # Clean risk table.
    project = Project(sample_count=sample_count, rng_seed=rng_seed)  # Simulator object.
    activities: dict[int, ImportedExcelActivity] = {}  # Metadata keyed by simulator node ID.

    for row in activities_rows:
        activity_id = _required_text(row, "Activity_ID")  # External label A_i.
        node_id = _activity_node_id(activity_id)  # Convert A_i to integer i.
        duration_model = _duration_model_from_activity(row)  # Baseline duration law Z_i.
        project.add_work_package(node_id, duration_model)  # Register activity i and D_i law.
        project.set_work_package_cost_model(
            node_id,
            DeterministicContinuousDistribution(_number(row, "Variable_Cost_per_Day_x1000", default=0.0)),
            fixed_costs=[_number(row, "Fixed_Cost_x1000", default=0.0)],
        )  # Attach K_i and H_i^j from the cleaned activity row.

        predecessor_ids = _parse_predecessors(row.get("Predecessors"))  # Parse Pred(i).
        activities[node_id] = ImportedExcelActivity(
            activity_id=activity_id,  # External activity label.
            node_id=node_id,  # Simulator activity ID i.
            phase=_optional_text(row.get("Phase")),  # Human-facing phase metadata.
            predecessor_ids=tuple(predecessor_ids),  # Parsed predecessor set.
            raw=row,  # Preserve the cleaned row for diagnostics.
        )

    for activity in activities.values():
        for predecessor in activity.predecessor_ids:
            if predecessor in activities:
                project.add_dependency(predecessor, activity.node_id)  # Encode predecessor -> successor.

    risks: dict[str, ImportedExcelRisk] = {}  # Metadata keyed by risk ID.
    for row in risk_rows:
        risk_id = _required_text(row, "Risk_ID")  # External risk label R_j.
        risk_type = _required_text(row, "Risk_Type").strip().lower()  # Schedule or Cost.
        affected_node_id = _activity_node_id(_required_text(row, "Affected_Activity_ID"))  # Activity i.
        probability_model = _probability_model_from_risk(row)  # Law of P_i^j.

        if affected_node_id not in activities:
            continue  # Ignore risks that point outside the imported activity set.
        if risk_type == "schedule":
            project.add_schedule_risk(
                affected_node_id,
                ScheduleRisk(
                    probability_model=probability_model,  # Occurrence probability law.
                    severity_model=_schedule_impact_model_from_risk(row),  # Discrete time impact U_i^j.
                ),
            )
        elif risk_type == "cost":
            project.add_cost_risk(
                affected_node_id,
                CostRisk(
                    probability_model=probability_model,  # Occurrence probability law.
                    severity_model=_cost_impact_model_from_risk(row),  # Continuous cost impact U_i^j.
                ),
            )
        else:
            raise ValueError(f"unsupported risk type {risk_type!r} for {risk_id}.")

        risks[risk_id] = ImportedExcelRisk(
            risk_id=risk_id,  # External risk label.
            risk_type=risk_type.title(),  # Notebook-readable risk type.
            affected_node_id=affected_node_id,  # Activity i affected by the risk.
            raw=row,  # Preserve the cleaned row for diagnostics.
        )

    return ImportedExcelProject(
        name=path.stem,  # Workbook stem is the display name.
        simulator_project=project,  # Fully wired Project object.
        activities=activities,  # Activity metadata.
        risks=risks,  # Risk metadata.
        network_graph=project.graph.copy(),  # Notebook-facing schedule graph.
    )


def read_excel_tables(file_path: str | Path) -> dict[str, list[dict[str, object]]]:
    """Return the cleaned workbook tables without building a Project."""
    path = Path(file_path)
    return {
        "Activities": _read_xlsx_table(path, "Activities"),
        "Risks": _read_xlsx_table(path, "Risks"),
    }


def excel_project_summary(imported_project: ImportedExcelProject) -> str:
    """Return a compact notebook-friendly summary of an imported Excel project."""
    project = imported_project.simulator_project
    lines = [
        f"Project: {imported_project.name}",
        f"Activities: {len(imported_project.activities)}",
        f"Risks: {len(imported_project.risks)}",
        "",
        "id | predecessors | schedule risks | cost risks | E[D_i]",
        "-" * 64,
    ]
    risks_by_activity: dict[int, dict[str, list[str]]] = {}
    for risk in imported_project.risks.values():
        bucket = risks_by_activity.setdefault(risk.affected_node_id, {"Schedule": [], "Cost": []})
        bucket[risk.risk_type].append(risk.risk_id)

    for node in project.topological_order():
        activity = imported_project.activities[node]
        model = project.work_packages[node].duration_model
        expected_duration = getattr(model, "expected_value", "?")
        risk_bucket = risks_by_activity.get(node, {"Schedule": [], "Cost": []})
        predecessors = ", ".join(f"A{pred}" for pred in activity.predecessor_ids) or "empty"
        schedule_risks = ", ".join(risk_bucket["Schedule"]) or "-"
        cost_risks = ", ".join(risk_bucket["Cost"]) or "-"
        lines.append(f"A{node} | {predecessors} | {schedule_risks} | {cost_risks} | {expected_duration}")
    return "\n".join(lines)


def _duration_model_from_activity(row: dict[str, object]):
    distribution = _normalize_label(row.get("Duration_Distribution"))
    minimum = _integer(row, "Duration_Min_Days")
    maximum = _integer(row, "Duration_Max_Days")
    most_likely = _integer(row, "Duration_Most_Likely_Days", default=(minimum + maximum) // 2)

    if distribution in {"triangular", "pert", "shiftedbinomial", "shifted binomial", "binomial"}:
        return ShiftedBinomialDuration(minimum=minimum, maximum=maximum, most_likely=most_likely)
    if distribution in {"uniform", "uniforme", "discreteuniform", "discrete uniform"}:
        return DiscreteUniformDuration(minimum=minimum, maximum=maximum)
    raise ValueError(f"unsupported duration distribution {row.get('Duration_Distribution')!r}.")


def _probability_model_from_risk(row: dict[str, object]) -> UniformContinuousDistribution:
    return UniformContinuousDistribution(
        minimum=_number(row, "Probability_Min"),
        maximum=_number(row, "Probability_Max"),
    )  # The cleaned dataset currently stores probability uncertainty as a bounded uniform interval.


def _schedule_impact_model_from_risk(row: dict[str, object]) -> DiscreteUniformDuration:
    return DiscreteUniformDuration(
        minimum=_integer(row, "Impact_Min"),
        maximum=_integer(row, "Impact_Max"),
    )  # Schedule risk impacts are discrete day effects.


def _cost_impact_model_from_risk(row: dict[str, object]) -> UniformContinuousDistribution:
    return UniformContinuousDistribution(
        minimum=_number(row, "Impact_Min"),
        maximum=_number(row, "Impact_Max"),
    )  # Cost risk impacts are continuous x1000 monetary units.


def _read_xlsx_table(path: Path, sheet_name: str) -> list[dict[str, object]]:
    """Read one simple worksheet table from an .xlsx file using the stdlib."""
    with ZipFile(path) as archive:
        shared_strings = _shared_strings(archive)  # Text table used by xlsx cells.
        sheet_path = _sheet_path(archive, sheet_name)  # Locate the worksheet XML.
        root = ET.fromstring(archive.read(sheet_path))  # Parse worksheet XML.

    raw_rows: list[list[object]] = []
    for row in root.findall("main:sheetData/main:row", NS):
        values: list[object] = []
        for cell in row.findall("main:c", NS):
            column_index = _column_index(cell.attrib["r"])  # Convert A1 -> 0.
            while len(values) <= column_index:
                values.append(None)
            values[column_index] = _cell_value(cell, shared_strings)  # Store sparse cell value.
        raw_rows.append(values)

    header = next((row for row in raw_rows if any(value not in (None, "") for value in row)), None)
    if header is None:
        return []
    headers = [str(value).strip() if value is not None else "" for value in header]
    table: list[dict[str, object]] = []
    for raw_row in raw_rows[raw_rows.index(header) + 1 :]:
        if not any(value not in (None, "") for value in raw_row):
            continue
        row_dict = {
            header_name: raw_row[index] if index < len(raw_row) else None
            for index, header_name in enumerate(headers)
            if header_name
        }
        table.append(row_dict)
    return table


def _shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings: list[str] = []
    for item in root.findall("main:si", NS):
        strings.append("".join(text.text or "" for text in item.findall(".//main:t", NS)))
    return strings


def _sheet_path(archive: ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    for sheet in workbook.find("main:sheets", NS):
        if sheet.attrib["name"] == sheet_name:
            rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
            target = targets[rel_id].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError(f"workbook does not contain sheet {sheet_name!r}.")


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> object:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//main:t", NS))
    value = cell.find("main:v", NS)
    if value is None or value.text is None:
        return None
    if cell_type == "s":
        return shared_strings[int(value.text)]
    if cell_type == "b":
        return bool(int(value.text))
    if cell_type == "str":
        return value.text
    number = float(value.text)
    return int(number) if number.is_integer() else number


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    column = 0
    for letter in letters:
        column = column * 26 + ord(letter) - ord("A") + 1
    return column - 1


def _parse_predecessors(value: object) -> list[int]:
    text = _optional_text(value)
    if not text or text.lower() in {"empty", "none", "nan", "-"}:
        return []
    return [_activity_node_id(match.group(0)) for match in re.finditer(r"A?\d+", text)]


def _activity_node_id(value: object) -> int:
    text = _required_plain_text(value)
    match = re.search(r"\d+", text)
    if match is None:
        raise ValueError(f"cannot parse activity ID from {value!r}.")
    node_id = int(match.group(0))
    if node_id <= 0:
        raise ValueError("activity IDs must be positive.")
    return node_id


def _required_text(row: dict[str, object], column: str) -> str:
    return _required_plain_text(row.get(column))


def _required_plain_text(value: object) -> str:
    text = _optional_text(value)
    if not text:
        raise ValueError("required text value is missing.")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(row: dict[str, object], column: str, *, default: float | None = None) -> float:
    value = row.get(column)
    if value in (None, ""):
        if default is None:
            raise ValueError(f"missing required numeric column {column}.")
        return default
    return float(value)


def _integer(row: dict[str, object], column: str, *, default: int | None = None) -> int:
    value = row.get(column)
    if value in (None, ""):
        if default is None:
            raise ValueError(f"missing required integer column {column}.")
        return default
    return int(round(float(value)))


def _normalize_label(value: object) -> str:
    text = _optional_text(value) or ""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return normalized.strip().lower().replace("_", " ")
