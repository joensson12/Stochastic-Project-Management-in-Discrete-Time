# Stochastic Project Management in Discrete Time

**Stochastic Project Management in Discrete Time is a transparent simulation and analysis platform for practitioners who want to understand project time, cost, and risk under uncertainty.**

The project is aimed especially at practitioners of project risk management, including those working with integrated cost and schedule risk analysis. Most project plans are presented as one deterministic schedule: one duration, one finish date, one budget. Real projects do not behave that cleanly. Activity durations vary, risks occur or do not occur, costs move with time, and the critical path can change from one scenario to another.

This project gives an introduction to project risk management in discrete time and rephrases familiar planning concepts in the language of probability theory. It models a project as a network of work packages, simulates many possible project outcomes, and reports where time, cost, and risk exposure concentrate.

The purpose is not only to produce simulation outputs, but to make the underlying algorithms transparent to practitioners. It is less polished than commercial tools with simple point-and-click interfaces, but it is easier to inspect, adapt, and build upon. Simulation data can be exported or handled directly in Python notebooks, processed for the exact needs of a project team, and moved into Excel when that is the right medium for reporting or further analysis.

## Why This Matters

Traditional project risk tools often separate the conversation into different boxes:

- schedule analysis asks which activities affect completion time
- cost analysis asks where budget exposure sits
- risk registers rank events using probability-impact scores

This project connects those views in one probabilistic framework. The goal is to help project risk practitioners see how the analysis is constructed: which assumptions enter the model, how they are sampled, how the project network is evaluated, and how cost and schedule exposure emerge from the simulated scenarios.

For a project manager, this helps answer questions like:

- What is the distribution of possible project completion times?
- Which work packages are often on a critical path?
- Which activities consume the largest share of total project cost?
- Which risks drive schedule impact, cost impact, or both?
- Where do cost exposure and schedule-critical exposure overlap?
- Are the simulated expected values consistent with analytical checks?

## Transparency, Not a Black Box

Many project risk workflows rely on software that hides important analytical choices behind the interface. That can be convenient, but it can also make it difficult to understand why a result appears, how sensitive it is to assumptions, or how to adapt the analysis to a particular decision.

This repository takes a different approach. The mathematical definitions, simulation logic, validation checks, and import/export workflows are kept close together. Practitioners can trace how discrete-time probability models are translated into project outcomes, then modify or extend the process when their project requires a different treatment of cost, schedule, dependencies, risks, or reporting.

This is especially relevant with modern AI-assisted analysis. Large simulation outputs can be processed together with AI through code, for example in Python notebooks. That gives AI access to detailed scenario data while preserving transparency: the suggested analysis, transformations, plots, and calculations can be inspected as code instead of disappearing into a black-box conversation.

### AI Transparency Disclaimer

The mathematical framework, including pseudocode, was developed by the author. AI has only supported the work by assisting in cleaning up and fixing details. The code was developed together with AI tools based on the mathematical framework.

## What Has Been Built

This repository contains both the mathematical foundation and a working Python implementation.

The current prototype supports:

- Monte Carlo simulation of project duration through a dependency network
- stochastic work-package duration models
- deterministic or sampled dependency lags
- schedule risks attached to individual work packages
- daily cost, fixed cost, and cost-risk simulation
- project-level completion-time samples
- sampled critical-path reconstruction
- activity-level cost and critical-time exposure diagnostics
- exact shifted-Poisson completion-time CDF validation for small networks
- LibreProject/MS Project XML import
- cleaned Excel workbook import for activities and risks

The implementation is intentionally inspectable. The simulator core is separate from notebook-facing importers, so project-management data can be translated into a clean model without hiding the assumptions inside a black box. The result is a platform that is useful for learning, experimentation, validation, and building customized risk-analysis workflows.

## Example Diagnostic

One of the main diagnostics developed in the technical report combines cost exposure and schedule exposure at the work-package level.

![Activity cost ratio and critical-time ratio metric](Report%20files/report_outputs/figures/activity_cost_ratio_vs_critical_time_ratio.png)

The horizontal axis measures how much of project completion time an activity contributes in scenarios where it is critical. The vertical axis measures the activity's share of total project cost. Point size and color represent how often the activity appears on a critical path.

In practical terms:

- upper-right activities deserve attention for both cost and schedule control
- high-cost but low-critical-time activities matter mainly for budget control
- high-critical-time but low-cost activities matter mainly for schedule control
- frequently critical activities are natural candidates for closer monitoring

This is a screening metric, not a promise that reducing one activity by one day will always reduce the whole project by one day. Its value is that it gives project risk practitioners a structured way to decide where deeper analysis is worth doing.

## Validation Example

The report also validates the Monte Carlo timing simulation against an exact completion-time calculation for a small shifted-Poisson network.

![Exact CDF versus Monte Carlo CDF](Report%20files/mc_cdf_validation_outputs_poisson_exp/figures/01_discrete_cdf_exact_vs_monte_carlo.png)

This matters because project-network simulation can be subtle: parallel paths, merge activities, shared work packages, and changing critical paths can all create errors if the algorithm is wrong. The validation benchmark checks that the simulated completion-time distribution behaves as expected and helps make the algorithmic behavior visible.

## Technical Report

The full technical background is available here:

[Open the technical report](Report%20files/Stochastic_Project_Management_in_Discrete_Time__A_Probability_Based_Simulation_Framework_for_Project_Time__Cost__and_Risk.pdf)

The report develops:

- a probabilistic foundation for project time, cost, and risk
- work-package start and finish time definitions over a project network
- cost models connected to activity duration
- risk occurrence and impact models
- Monte Carlo algorithms for project time and cost
- validation against exact completion-time distributions
- an activity-level cost and critical-time exposure metric for project control

## Software Structure

The core package is organized around a small set of concepts:

```text
src/spm/
  distributions.py       Probability, duration, and cost distribution models
  work_package.py        Activity-level duration, cost, and risk simulation
  project.py             Project network, timing recursion, critical paths, and cost totals
  interfaces/            XML and Excel importers for project-management data
```

For a developer-oriented explanation of the package structure, simulation flow, and extension points, see [`docs/software-architecture.md`](docs/software-architecture.md).

Practitioners can use this structure as a starting point for notebook-based analysis: import a project, run simulations, extract detailed scenario-level or activity-level outputs, create custom plots, compare assumptions, and export selected results to Excel for communication with the wider project team.

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Run the tests:

```bash
pytest
```

Create a small stochastic project:

```python
from spm import Project, ShiftedPoissonDuration

project = Project(sample_count=10_000, rng_seed=1)

project.add_work_package(1, ShiftedPoissonDuration(lambda_=2.0, a=3))
project.add_work_package(2, ShiftedPoissonDuration(lambda_=3.0, a=2))
project.add_work_package(3, ShiftedPoissonDuration(lambda_=1.5, a=4))

project.add_dependency(1, 3)
project.add_dependency(2, 3)

result = project.simulate_project_time()

print(result.completion_times.mean())
print(result.critical_paths[:5])
```

## Importing Project Data

Notebook-facing import functions live in `spm.interfaces`, separate from the simulator core.

### LibreProject/MS Project XML

```python
from spm.interfaces import (
    draw_project_network,
    load_project_from_xml,
    print_project_diagnostics,
)

imported = load_project_from_xml(
    "Time only poisson experiment/Example project/Building Garage_tmp0.xml",
    sample_count=1000,
    rng_seed=1,
)

draw_project_network(imported)
print_project_diagnostics(imported)
result = imported.simulator_project.simulate_project_time()
```

The XML importer is a specific prototype convention:

- it reads the task duration as the smallest modal duration
- it reads `a_i` from task notes of the form `Min: a_i`
- it models each duration as shifted Poisson
- it treats imported predecessor links as finish-to-start dependencies

### Cleaned Excel Workbook

```python
from spm.interfaces import excel_project_summary, load_project_from_excel

imported = load_project_from_excel(
    "Time and Cost experiment/Experiment/cleaned_project_risk_dataset_UPDATED.xlsx",
    sample_count=10_000,
    rng_seed=1,
)

print(excel_project_summary(imported))

project = imported.simulator_project
time_result = project.simulate_project_time_with_critical_activities()
project.simulate_all_costs()
print(project.expected_project_cost())
```

## Current Status

This is a prototype, not a commercial scheduling product. Its strength is that it connects project-management concepts, probability assumptions, simulation algorithms, validation checks, and software implementation in one inspectable framework.

For practitioners, the tradeoff is deliberate: the interface is less simple than a commercial risk tool, but the analysis is more open. You can see the probability model, inspect the simulation data, process the results in notebooks, export data to Excel, and build customized workflows for integrated cost and schedule risk analysis.
