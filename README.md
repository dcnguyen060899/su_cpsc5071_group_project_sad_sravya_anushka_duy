# Predicting Natural Disaster Economic Impact from County Demographics and Housing Conditions

**CSPC 5071 — Data Management for Data Science**
**Professor Yueting Chen | Seattle University, Winter 2026**
**Team SAD:** **S**ravya, **A**nushka, **D**uy

---

## Research Question

> Can we predict the level of federal disaster assistance allocated to a county based on its pre-disaster demographic characteristics and housing cost conditions?

This is a **cross-sectional study**, so for each observation represents a county characterized by point-in-time demographic and housing snapshots, not a time series. We link three federal data sources at the county level using standardized 5-digit FIPS codes. Now that we had worked through all three phases of this project, our understanding of this question has been refined in important ways as modeling confirmed the most important finding: disaster frequency is overwhelmingly driven by geography, not demographics. Bottom line is that this shifts the interpretation of our research question because we are not predicting which counties are inherently more vulnerable, but rather which geographic regions are historically more exposed to federally declared disasters.

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
│   ├── initlal_data_exploratory.ipynb  # Data exploration & challenge discovery
│   ├── database_pipeline.ipynb         # Database management pedagogical reasoning & implementation (analysis-readiness)
│   ├── eda.ipynb                       # Exploratory data analysis — distributions, correlations, geographic patterns
│   └── modeling.ipynb                  # Statistical modeling — OLS, logistic regression, decision tree, state fixed effects
├── data/
│   ├── FY25_FMRs-Table 1.csv          # HUD Fair Market Rents (raw)
│   ├── Field_Descriptions-Table 1.csv  # HUD field documentation
│   ├── fema_sample_data.csv            # 10-record FEMA API sample
│   └── census_exploration.csv          # 100-row Census sample
└── report/
    ├── project1_SAD_DuyNguyen_Sravya_Anushka.pdf       # Phase 1: Project proposal
    ├── project2_groupSAD_Sravya_Anushka_Duy.pdf        # Phase 2: Progress report (EDA results + plan)
    └── project3_groupSAD_Sravya_Anushka_Duy.pdf        # Phase 3: Final report (modeling results + conclusions)
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

**Solution:** Zero-pad and concatenate (FEMA/Census) or zero-pad and slice (HUD) to produce a standard 5-digit string (e.g., `'01001'`). We rejected integer conversion because it silently drops leading zeros and that is not something we can afford in a join key. Imagine if we had not caught this, every county starting with a zero would have joined to the wrong row entirely.

### 2. HUD New England Town-Level Subdivisions

HUD reports FMR at the town level for New England states, producing 4,764 rows for ~3,228 counties. Multiple towns share the same county FIPS, creating duplicate keys that would inflate disaster counts in a naive join. Think about it, without resolving this, we would be inflating disaster counts simply because of how New England jurisdictions are structured, not because of anything real about those counties.

**Solution:** We suggest calcualte the population-weighted aggregation, meaning that each town's FMR contribution is weighted by its population share within the county. Metro flag set to 1 if any sub-area is metro. This is a core data integrity decision that made the subsequent analysis trustworthy.

### 3. Census Missing-Value Sentinel

The Census API uses `-666666666` as a sentinel for suppressed data (1 county for median household income). This is not a real value but corrupts summary statistics if left in place. Imagine if we had not caught this, that would have been a catastrophically wrong income value feeding directly into our models.

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

We chose the `LEFT JOIN` design intentionally for information preservation — without it, those 80 missing counties would have silently disappeared instead of being explicitly acknowledged. And as we diagnose later in modeling, those 80 counties are not random, they are systematically the most disaster-prone counties (mean 10.8 declarations vs. 5.4 for the rest). So what do we do? We document them explicitly and carry that limitation forward transparently.

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

4. **Verify** the database with the verification queries documented in the verification notebook [extra reading alert but feel free to check it out!](https://drive.google.com/file/d/1OSIwjfpD2X4gO4Ed4qCL-0lqwpepmJz7/view?usp=sharing).

5. **Run the EDA notebook** to explore distributions, correlations, and geographic patterns in the analysis-ready dataset:
   ```
   code/eda.ipynb
   ```
   This notebook connects to your local MySQL instance, loads the `analysis_ready` view, engineers features (poverty rate, unemployment rate, log population), and walks through univariate, bivariate, and multivariate analysis. Make sure to update the MySQL config at the top of the notebook with your own credentials and so the same as Steps 2 and 3.

6. **Run the modeling notebook** to see the full statistical analysis of OLS regression, logistic regression, decision tree, and the state fixed effects analysis that ties everything together:
   ```
   code/modeling.ipynb
   ```
   This notebook picks up directly from the `analysis_ready` view, drops the 80 counties with any NULL (2.4% of rows), does an 80/20 stratified train/test split, and walks through all three models in pedagogical order. Keep in mind, however, the goal here is interpretation and transparency, not maximum predictive accuracy, that is why we chose these models specifically because they teach us something different at each step. Make sure your MySQL credentials match what you set in Steps 2 and 3.

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

## EDA Key Findings

After building the database, we ran a full exploratory analysis (`code/eda.ipynb`) on the `analysis_ready` view. Here is a summary of what we learned and why it matters for modeling: the details and visualizations are all in the notebook itself, so we encourage you to run it and explore. And we want to be upfront that EDA is what raised the questions we then used modeling to answer.

### Target variable is heavily right-skewed

`total_disasters` has a mean of 5.7 but a median of only 4, with a long right tail stretching to 149 declarations. Most counties cluster at low counts while a small number accumulate very high totals. As we diagnose in EDA thoroughly, raw counts are not suitable for ordinary linear regression — so what do we do for modeling to be interpretable? We log transform them. A log transformation `log(1 + total_disasters)` produces a much more symmetric distribution. For classification, we defined a binary `high_impact` flag as counties above the median (more than 4 declarations), producing a near-balanced 48.9% / 51.1% split. Bottom line is that our models can be evaluated straightforwardly without any class imbalance corrections.

### Raw counts are misleading — rates are necessary

Think about it: a county with 50,000 people in poverty sounds alarming, but if that county is Los Angeles (population ~10 million) it is a 0.5% rate. A rural county with 1,000 in poverty out of 5,000 people has a 20% rate. These two counties are not comparable using raw counts and this is why we need to be thoughtful here. We engineered three derived features to make counties comparable:

| Feature | Formula | Why |
|---------|---------|-----|
| `poverty_rate` | `poverty_count / total_population × 100` | Normalizes poverty by county size |
| `unemployment_rate` | `unemployment_count / labor_force_count × 100` | Normalizes unemployment by labor force |
| `log_population` | `log(1 + total_population)` | Compresses a 50-to-10-million range into a manageable scale |

### FMR columns are almost perfectly correlated

All five Fair Market Rent columns (`fmr_0` through `fmr_4`) are correlated above 0.94 with each other. This makes sense — a county where studios are expensive also has expensive 4-bedrooms. Including all five in a regression would just introduce collinearity without adding any new information.

**Decision:** We use `fmr_2` (2-bedroom FMR) as the single housing cost representative, since it is HUD's primary published affordability benchmark and the most widely cited FMR in housing research.

### Geography dominates — demographics alone may not be enough

A state-level aggregation shows that disaster frequency is heavily concentrated in specific regions. For example, Maine leads with ~40 mean declarations per county, followed by Gulf Coast states (Louisiana, Florida) and tornado-alley states (Kentucky, South Carolina). Meanwhile, western mountain states and the Pacific Northwest have far fewer declarations.

This is the biggest takeaway from EDA, and we want to be honest about this: a county's location on the Gulf Coast hurricane track or in tornado alley is **not encoded** in its poverty rate, median income, or housing cost. Demographics alone may underpredict disaster frequency because they cannot capture geographic exposure. We designed the modeling phase specifically to test this — comparing models with and without state-level indicator variables (fixed effects). This is a question we were already suspecting from EDA, but we needed the models to actually prove it.

### 80 counties dropped due to missing data — and they are not random

The `LEFT JOIN` design preserved all counties, but 80 rows have at least one NULL across Census or HUD columns. Profiling these counties reveals they are systematically different: higher mean disaster count (10.8 vs 5.4), concentrated in US territories (American Samoa, CNMI, USVI), Alaska non-standard jurisdictions, and Connecticut's restructured county-equivalents. This is a data infrastructure limitation, as three independently maintained federal datasets do not share uniform geographic coverage, thereby not a data entry error. Listwise deletion drops only 2.4% of rows but introduces a mild selection bias that we note as a model limitation.

### Proposed feature set for modeling

Based on everything we learned in EDA, our candidate predictors going into the modeling phase are:

| Feature | Role |
|---------|------|
| `log_population` | County size (log-transformed) |
| `median_household_income` | Economic capacity |
| `poverty_rate` | Economic vulnerability (derived) |
| `unemployment_rate` | Labor market health (derived) |
| `fmr_2` | Housing cost proxy (2-bedroom FMR) |
| `metro` | Urban/rural classification (binary) |

---

## Modeling Key Findings

Cool, so after EDA raised all the right questions, we built three models in `code/modeling.ipynb` (OLS regression, logistic regression, and a decision tree), in order to actually answer them. We chose these not for maximum predictive power but for transparency and pedagogical value. We anticipated that each model would teach us something different about the data. Let's recap what we found:

### Geography dominates demographics — by a factor of 3.5x

Not to our surprise, the demographics-only OLS model explains only 16.7% of the variance in log-disaster counts ($R^2 = 0.167$). But when we add 50 state dummy variables (state fixed effects), $R^2$ jumps to 0.742, that is a 57.5 percentage point jump. The AIC drops from 4,035 to 369. That is the smoking gun. Geography explains 3.5 times more variance in disaster frequency than all six demographic and housing features combined. Knowing which state a county is in is far more predictive than knowing its poverty rate, income, housing cost, or unemployment rate.

### Poverty's association with disasters is a geographic confound

This is the most policy-relevant finding of the entire study. Without state controls, poverty rate appears to strongly predict more disasters ($p < 0.001$, $r = +0.218$). A naive interpretation would conclude that poverty increases disaster vulnerability. But as we add state dummies and hold geography constant, `poverty_rate` collapses from highly significant ($p < 0.001$) to completely non-significant ($p = 0.82$). Same happens to `fmr_2`. What this tells us is that poor counties and high-rent counties aren't disaster-prone because of their demographics but rather they just happen to co-locate in the same high-risk geographies like the Gulf Coast and Deep South. Hence, it is more like correlations than causation. Any study of disaster vulnerability that does not control for geography risks this confound, and that is not something we can afford if the goal is real policy insight.

### Population size is the only robust demographic predictor

`log_population` remains highly significant ($p < 0.001$) before and after controlling for geography, with a stable coefficient (~0.027 in both models). This is the one genuine demographic predictor: larger counties cross the federal declaration threshold more often because they have more infrastructure, more people, and more area exposed to any given hazard. Everything else either disappears or weakens once state is controlled for.

### All three models tell the same story — same coin, different sides

As we observe across OLS, logistic regression, and the decision tree, the pattern converges every single time. Logistic regression and the decision tree achieve nearly identical accuracy (both 0.656) and AUC (0.711 and 0.700 respectively). Poverty rate is the strongest positive predictor in all three. `fmr_2` and `median_household_income` consistently rank next. `metro` status and unemployment contribute little to nothing. Same story (analogy to coin), different side of the coin that whether we are reading OLS coefficients, logistic log-odds, or Gini importance from the decision tree, the same six features tell us the same thing. The near-identical performance between a linear model (logistic regression) and a non-linear model (decision tree) further implies that the relationship between demographics and disaster classification is primarily linear, and switching to a more complex algorithm won't help until we add geographic features.

### Conclusion

Let's revisit our research question from the very beginning: Can we predict the level of federal disaster assistance allocated to a county based on its pre-disaster demographic characteristics and housing cost conditions?                                                                                                  

Honestly, the answer is more nuanced than a flat yes or no and we want to be upfront about this. It is not that demographic and housing features were utterly useless; it is that geography accounts for more than half the story. Think about it: demographics alone explain 16.7% of variance in disaster frequency, but the moment we add state fixed effects, that number jumps to 74.2%. That is the smoking gun. Bottom line is that we answer our research question with complete scientific honesty and rigor, meaning that it is not a clean yes, not a clean no, but rather: yes, demographics carry a real but limited signal, and the dominant driver is where a county sits on the map, not what its poverty rate or income level looks like.

---

## Tech Stack

- **Language:** Python 3, Jupyter Notebook
- **Database:** MySQL 8.0 (Docker)
- **ETL & Analysis:** `requests`, `pandas`, `numpy`, `mysql-connector-python`, `sqlalchemy`
- **Visualization:** `matplotlib`, `seaborn`
- **Modeling:** `scikit-learn` (logistic regression, decision tree), `statsmodels` (OLS with full inference, state fixed effects)

---

## License

MIT License — see [LICENSE](LICENSE) for details.


***Thank you for reading. Hope these make sense and are interesting to all the readers! Special thanks to our group work on the database engineering pipeline, because without the referential integrity on FIPS codes and the LEFT JOIN design, the state fixed effects analysis would not have been credible, and neither would the modeling notebook that builds on top of it.***
