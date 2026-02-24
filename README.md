# Predicting Natural Disaster Economic Impact from County Demographics and Housing Conditions

**CSPC 5071 — Data Management for Data Science**
**Professor Yueting Chen | Seattle University, Winter 2026**
**Team SAD:** **S**ravya, **A**nushka, **D**uy

---

## Research Question

> Can we predict the level of federal disaster assistance allocated to a county based on its pre-disaster demographic characteristics and housing cost conditions?

This is a **cross-sectional study**, so for each observation represents a county characterized by point-in-time demographic and housing snapshots, not a time series. We link three federal data sources at the county level using standardized 5-digit FIPS codes.

---

## Data Sources

| Source | Type | Scope | Raw Records |
|--------|------|-------|-------------|
| [FEMA Disaster Declarations v2](https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries) | REST API (public, no auth) | FY 2020–2024 federally declared disasters | 18,718 |
| [Census ACS 5-Year Estimates (2022)](https://api.census.gov/data/2022/acs/acs5) | REST API (no key required) | All U.S. counties — population, income, poverty, unemployment | 3,222 |
| [HUD Fair Market Rents FY2025](https://www.huduser.gov/portal/datasets/fmr.html) | CSV download | Fair Market Rents (0–4 BR), metro status, population | 4,764 (3,228 after county-level aggregation) |

---

## Repository Structure

```
.
├── README.md                           # This file
├── LICENSE                             # MIT License
├── code/
│   └── initlal_data_exploratory.ipynb  # Data exploration & challenge discovery
│   └── database_pipeline  # database management pedagogical reasoing & implementation (analysis-readiness)
├── data/
│   ├── FY25_FMRs-Table 1.csv          # HUD Fair Market Rents (raw)
│   ├── Field_Descriptions-Table 1.csv  # HUD field documentation
│   ├── fema_sample_data.csv            # 10-record FEMA API sample
│   └── census_exploration.csv          # 100-row Census sample
└── report/
    └── project1_SAD_DuyNguyen_Sravya_Anushka.pdf  # Project proposal
```

---

## Key Challenges Discovered During Exploration

The data exploration notebook (`code/initlal_data_exploratory.ipynb`) documents five data integration challenges and how we reasoned through each one:

### 1. FIPS Code Standardization

Each source represents county identifiers differently:

| Source | Format | Example |
|--------|--------|---------|
| FEMA | Two separate string fields | `fipsStateCode: '41'` + `fipsCountyCode: '067'` |
| Census | Two separate string fields | `state: '01'` + `county: '001'` |
| HUD | Single 10-digit integer | `100199999` → first 5 digits = `01001` |

**Solution:** Zero-pad and concatenate (FEMA/Census) or zero-pad and slice (HUD) to produce a standard 5-digit string (e.g., `'01001'`). We rejected integer conversion because it would drop leading zeros from codes like `'01001'`.

### 2. HUD New England Town-Level Subdivisions

HUD reports FMR at the town level for New England states, producing 4,764 rows for ~3,228 counties. Multiple towns share the same county FIPS, creating duplicate keys that would inflate disaster counts in a naive join.

**Solution:** Population-weighted aggregation — each town's FMR contribution is weighted by its population share within the county. Metro flag set to 1 if any sub-area is metro.

### 3. Census Missing-Value Sentinel

The Census API uses `-666666666` as a sentinel for suppressed data (1 county for median household income). This is not a real value but corrupts summary statistics if left in place.

**Solution:** Replace sentinel with `NULL`/`NaN`. Use pandas nullable integer type (`Int64`) to preserve NaN alongside integer values.

### 4. Temporal Alignment Across Sources

FEMA covers FY2020–2024, Census is a 2022 snapshot, HUD is FY2025. Initial concern: temporal mismatch invalidates joins.

**Resolution:** This is a cross-sectional study, not a time series. County demographics and housing costs are sufficiently stable year-to-year to serve as proxies for persistent county characteristics across the disaster window. This framing is standard in cross-sectional disaster impact research.

### 5. FEMA API Pagination

The API caps responses at 1,000 records per request, which was not apparent during initial schema design.

**Solution:** Pagination loop using OData `$skip`/`$top` parameters, incrementing by 1,000 until an empty response is returned (19 pages total).

---

## Database Design

The exploration findings informed a relational schema with 4 tables in MySQL 8.0:

```
                    ┌──────────────────┐
                    │     county       │
                    │──────────────────│
                    │ PK: fips CHAR(5) │
                    │ county_name      │
                    │ state_abbr       │
                    └──────┬───────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
   ┌────────▼───────┐ ┌───▼──────────┐ ┌─▼──────────────┐
   │   disasters    │ │ demographics │ │ housing_costs  │
   │────────────────│ │──────────────│ │────────────────│
   │ PK: id         │ │ PK/FK: fips  │ │ PK/FK: fips    │
   │ FK: fips       │ │ total_pop    │ │ metro          │
   │ disaster_number│ │ med_income   │ │ pop2022        │
   │ declaration_..│ │ poverty_cnt  │ │ fmr_0..fmr_4   │
   │ incident_type  │ │ unemp_cnt   │ └────────────────┘
   │ program flags  │ │ labor_force  │
   └────────────────┘ └──────────────┘
```

- **county** (3,292 rows) — central reference table, union of all FIPS from all 3 sources
- **disasters** (18,718 rows) — one row per disaster declaration per county (1:N with county)
- **demographics** (3,222 rows) — Census ACS 2022 snapshot (1:1 with county, partial)
- **housing_costs** (3,228 rows) — HUD FY2025 FMR data (1:1 with county, partial)

Two SQL views support analysis:
- `county_disaster_summary` — aggregates disasters to one row per county
- `analysis_ready` — LEFT JOINs all 4 tables into a flat 3,292 × 17 DataFrame

---

## How to Reproduce

### Prerequisites

- Python 3.x with `requests`, `pandas`, `numpy`, `mysql-connector-python`, `sqlalchemy`
- MySQL 8.0 (Docker recommended)
- Jupyter Notebook

### Steps

1. **Start MySQL:**
   ```bash
   docker run --name mysql-container -e MYSQL_ROOT_PASSWORD='fill in your own local password' \
     -e MYSQL_DATABASE='choose your own local database' -p 3306:3306 -d mysql:8.0
   ```

2. **Run the exploration notebook**, read through the output and come up with your own interpretation before reading the notebook interpretation to understand the data:
   ```
   code/initlal_data_exploratory.ipynb
   ```

3. **Run the pipeline notebook** (in local `group_project/code/`) to build the database:
   - Fetches all data from APIs and CSV
   - Cleans, transforms, and standardizes FIPS codes
   - Creates tables and inserts data into `disaster_impact_db`
   - Creates analysis views

4. **Verify** the database with the verification queries documented in the verification notebook [extra reading alert but feel free to check it out: ](https://drive.google.com/file/d/1OSIwjfpD2X4gO4Ed4qCL-0lqwpepmJz7/view?usp=sharing).

---

## Coverage Summary

| Metric | Count |
|--------|-------|
| Total counties in database | 3,292 |
| Counties with ≥1 disaster | 3,277 (99.5%) |
| Counties with zero disasters | 15 |
| Counties missing demographics | 70 |
| Counties missing housing data | 64 |
| Counties missing median income | 1 |

---

## Tech Stack

- **Language:** Python 3, Jupyter Notebook
- **Database:** MySQL 8.0 (Docker)
- **Libraries:** `requests`, `pandas`, `numpy`, `mysql-connector-python`, `sqlalchemy`
- **Planned modeling:** scikit-learn, XGBoost/LightGBM, matplotlib/seaborn

---

## License

MIT License — see [LICENSE](LICENSE) for details.


***Thank you for reading. Hope these make sense and interesting to all the readers!***

