# Dataset Card — OSMI Mental Health in Tech Survey

---

## Source

**Provider:** Open Sourcing Mental Illness (OSMI)  
**URL:** [https://osmihelp.org/research](https://osmihelp.org/research)  
**License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)

---

## Collection

| Property | Value |
|:---|:---|
| Collection method | Annual voluntary online survey |
| Collection period | 2016–2022 (7 annual waves) |
| Population | Technology-sector workers worldwide |
| Response format | Multiple choice, Likert scale, free text |
| Raw files | `Mental_Health_Survey_2016.csv` through `Mental_Health_Survey_2022.csv` |

---

## Composition

| Metric | Value |
|:---|:---|
| Total responses (merged) | 3,433 |
| Raw columns (union across years) | 193 |
| Columns retained after pruning | 19 (18 features + 1 target) |
| Target variable | `treatment` (binary: 1 = sought professional help, 0 = did not) |
| Class balance | 58.05% positive (treatment-seeking) / 41.95% negative |
| Train set size | 2,746 (80%) |
| Test set size | 687 (20%) |

---

## Preprocessing Pipeline

1. **Merge:** Load all 7 CSVs, tag each with a `year` column, concatenate with `pd.concat`.
2. **Column normalization:** Rename variant question wordings across survey years to consistent short names (`treatment`, `work_interfere`, `seek_help`, `remote_work`, `no_employees`).
3. **Target encoding:** Map `Yes`/`No` string values (2016 format) to `1`/`0` integers.
4. **Column pruning:** Drop columns with > 50% missing values. Drop free-text columns (identified by > 50 unique values).
5. **Missing value imputation:** Categorical columns → fill with `'Unknown'`. Numeric columns → fill with column median.
6. **Label encoding:** Map each categorical column to integer codes via sorted unique value mapping.
7. **Normalization:** Z-score standardization on all features: `(x - mean) / (std + 1e-8)`.
8. **Split:** 80/20 stratified train/test split with `random_state=42`.

---

## Feature Summary

After preprocessing, 18 features remain:

- **17 categorical** (label-encoded): employer size, remote work status, employer mental health benefits, comfort discussing mental health with supervisors/coworkers, family history, previous diagnosis, work interference level, etc.
- **1 numerical:** survey year

---

## Known Biases and Limitations

| Bias Type | Description |
|:---|:---|
| Self-selection | Only tech workers who voluntarily chose to participate. Likely overrepresents those with strong opinions about mental health. |
| Language | English-only survey. Excludes non-English-speaking tech workers. |
| Geographic | Respondent pool skews heavily toward US and Western Europe. |
| Temporal | Survey questions changed across years. Column union produces sparse matrices for year-specific questions. |
| Class imbalance | 58/42 split is mild but means a majority-class baseline already achieves 58% accuracy. |
| Free-text exclusion | Dropping free-text columns discards potentially informative qualitative data about workplace conditions. |

---

## Ethical Considerations

- **Sensitivity:** Mental health survey data carries inherent sensitivity. No personally identifiable information (PII) is present in the published dataset.
- **Consent:** All respondents opted in to the OSMI survey voluntarily.
- **Intended use:** Research and education only. Not suitable for clinical decision-making or individual risk assessment.
