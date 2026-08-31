# Predicting Pathway Left Shifts with Outcome-Oriented Predictive Process Monitoring

MSc Advanced Computer Science (Data Analytics), University of Leeds — COMP5200M.

Unplanned escalation of care ("left shifts") in MIMIC-IV v3.1, framed as outcome-oriented
predictive process monitoring. Two cohorts (ICU and surgical ward), five label variants
plus a restricted cohort, thirteen model families across five feature conditions.

## Data access

**No patient data is included in this repository**, in line with the PhysioNet
Credentialed Health Data Use Agreement.

MIMIC-IV v3.1 requires a credentialed PhysioNet account, CITI "Data or Specimens Only
Research" training, and a signed data use agreement. The scripts in `sql/` reconstruct
every derived table from the source database. Nothing in `results/` contains a patient
identifier.

## Repository layout

| Folder | Contents |
|---|---|
| `sql/` | Cohort construction, feature extraction and outcome labelling, numbered in execution order |
| `python-scripts/` | Feature assembly, partitioning, training, ablation, error analysis, and the figure scripts |
| `aire-hpc-scripts/` | Slurm submission scripts for the University of Leeds Aire cluster |
| `references/` | Lookup tables: Charlson and ICD chapter maps, care-unit intensity mapping |
| `results/` | Aggregate metrics and analysis outputs |
| `deliverables/` | The artefacts listed in section 1.3 of the dissertation |

## Deliverables

Section 1.3 of the dissertation lists five artefacts.

| # | Artefact | Location |
|---|---|---|
| 1 | Disco process maps of both cohorts | `deliverables/01-disco-process-maps/` |
| 2 | Training code for each model | `deliverables/02-training-code/` |
| 3 | The care-intensity mapping | `deliverables/03-care-intensity-mapping/` |
| 4 | Codebase repository, no patient data | this repository |
| 5 | MSc report | `deliverables/04-msc-report/` |

Model weights were retained on the Aire cluster and are not published here, as stated in
section 1.3.

## Environment

Python 3.11+ with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `torch`; PostgreSQL
for `sql/`. Two environments were used and are reported in Appendix C.4 of the
dissertation: a local workstation for the tabular families, and the Aire cluster for the
sequence architectures. No model family was fitted in both.
