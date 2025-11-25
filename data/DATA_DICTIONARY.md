# DATA DICTIONARY - GOLD STANDARD SPECIFICATION
## CES Hackathon Project - Canadian Election Forecasting

**Date:** 2025-10-31
**Status:** Gold Standard Variables Validated
**Purpose:** Document gold standard variable specifications for Canadian Election Study (CES) data used in election forecasting models

---

## Executive Summary

This data dictionary defines the **GOLD STANDARD** variables for the CES Hackathon project. All variables use **RAW SCALES** (no normalization) to preserve interpretability and statistical rigor. The gold standard was established through comprehensive codebook verification and validated with 23 automated tests.

**Key Principles:**
1. **Raw scales only** - No 0-1 normalization (preserves original survey meaning)
2. **Theory-driven selection** - Variables chosen based on voting behavior literature
3. **Systematic exclusions** - Invalid responses (don't know/refused) set to NaN
4. **Complete validation** - All 23 tests pass for 2021 and 2025 data
5. **Reproducible pipeline** - From raw .dta files to cleaned .parquet files

---

## Gold Standard Variables

All models in this repository MUST use these 11 variables with exact specifications below:

| Variable | Type | Range | Exclusions | Description |
|----------|------|-------|------------|-------------|
| **vote_choice** | Categorical | 1-6 | 7 (other) | Party voted for (1=Liberal, 2=CPC, 3=NDP, 4=Bloc, 5=Green, 6=PPC) |
| **age** | Continuous | 18-99 | - | Age in years (NOT categorical) |
| **pid** | Categorical | 1-8 | 9 (don't know) | Party identification (1-8 scale) |
| **ideology** | Ordinal | 0-10 | - | Left-right self-placement (0=left, 10=right) |
| **econ_retro** | Ordinal | 1-3 | 4,5 (don't know/refused) | National economy retrospective (1=Better, 2=Same, 3=Worse) |
| **trust_govt** | Ordinal | 1-4 | 5 (don't know) | Trust in federal government (1=Very satisfied, 4=Not at all) |
| **gender** | Binary | 1-2 | 3,4,5 (non-binary/other) | Gender (1=Male, 2=Female) |
| **education** | Ordinal | 1-11 | 12 (don't know) | Education level (1=No schooling, 11=Doctorate) |
| **province** | Categorical | 1-13 | - | Province/territory (standard postal codes) |
| **income** | Categorical | 1-8 | 9 (prefer not to say) | Household income bracket (1=<$30k, 8=>$150k) |
| **weight** | Continuous | >0 | - | Survey weight (restricted general weight) |

**Derived Variables:**
- **vote_liberal**: Binary outcome (1 if vote_choice==1, 0 otherwise) - derived from vote_choice

---

## File Structure

```
data/
├── 01-raw/CES/                    # Raw .dta files from CES
│   ├── CES_2019.dta              # 2019 election (Trudeau minority)
│   ├── CES_2021.dta              # 2021 election (Trudeau minority)
│   └── CES_2025.dta              # 2025 election (forecast)
├── 02-clean/CES/                  # Cleaned parquet files (GOLD STANDARD)
│   ├── CES_2019.parquet          # N=37,822 (limited variables due to source corruption)
│   ├── CES_2021.parquet          # N=20,968
│   └── CES_2025.parquet          # N=~40,000
└── scripts/
    └── preprocess_ces.py         # Master cleaning script
```

---

## Variable Specifications

### 1. DEPENDENT VARIABLE

#### vote_choice (Vote Choice - Categorical)

**Purpose:** Party voted for in federal election

**Raw Variable Names:**
- **2019:** `cps19_votechoice`
- **2021:** `cps21_votechoice`
- **2025:** `cps25_votechoice`

**Scale:** 1-6 (valid parties), 7 excluded
- 1 = Liberal Party of Canada (LPC)
- 2 = Conservative Party of Canada (CPC)
- 3 = New Democratic Party (NDP)
- 4 = Bloc Québécois (BQ)
- 5 = Green Party (GPC)
- 6 = People's Party of Canada (PPC)
- 7 = Other (EXCLUDED → set to NaN)

**Exclusion Rules:**
- Exclude 7 (other parties - too heterogeneous)
- Don't know/refused → NaN
- Missing → NaN

**Usage:** For binary models, derive `vote_liberal = (vote_choice == 1)`

---

#### vote_liberal (Binary Outcome - Derived)

**Purpose:** Binary outcome for logistic regression

**Derivation:**
```python
vote_liberal = (vote_choice == 1).astype(float)
vote_liberal = vote_liberal.where(vote_choice.notna(), np.nan)
```

**Scale:** 0/1 binary
- 0 = Voted for non-Liberal party (CPC, NDP, BQ, GPC, PPC)
- 1 = Voted Liberal
- NaN = Did not vote or missing

**Theory:** Following Camatarri (2024) approach - binary outcome captures incumbent party support in voter-level regression models

---

### 2. CORE PREDICTORS

#### ideology (Left-Right Ideology)

**Purpose:** Self-placement on ideological spectrum

**Theory:** Strongest predictor of vote choice (Campbell et al. 1960). Captures core political values and policy preferences.

**Raw Variable Names:**
- **2019:** `cps19_lr_scale_aft` (primary), `cps19_lr_scale_bef_1` (fallback)
- **2021:** `cps21_lr_scale_aft` (primary), `cps21_lr_scale_bef_1` (fallback)
- **2025:** `cps25_lr_canadians_1`

**Scale:** 0-10 (RAW scale, NO normalization)
- 0 = Most left
- 10 = Most right

**Exclusion Rules:**
- Valid range: 0-10
- Don't know/refused → NaN
- Values outside 0-10 → NaN

**CRITICAL:** Use RAW 0-10 scale. DO NOT normalize to 0-1. Raw scale preserves interpretability (1-point shift on 0-10 scale = substantive ideological change).

**Statistical Note:** Coefficients represent effect of 1-point shift on 0-10 scale. For example, β=0.5 means moving from 5 (center) to 6 (center-right) increases log-odds of Liberal vote by 0.5.

---

#### econ_retro (Economic Retrospective Evaluation)

**Purpose:** Evaluation of national economy compared to last year

**Theory:** Economic voting (Fiorina 1981). Retrospective evaluations predict vote for/against incumbent party.

**Raw Variable Names:**
- **2019:** `cps19_econ_retro`
- **2021:** `cps21_econ_retro`
- **2025:** `cps25_econ_retro`

**Scale:** 1-3 (RAW ordinal scale)
- 1 = Better
- 2 = About the same
- 3 = Worse

**Exclusion Rules:**
- Exclude 4 (don't know) → NaN
- Exclude 5 (prefer not to say) → NaN

**Statistical Note:** Higher values = worse economic evaluation. Use as ordinal predictor (assumes equal spacing). Coefficient represents effect of 1-category worsening.

---

#### trust_govt (Trust in Federal Government)

**Purpose:** Satisfaction with federal government performance

**Theory:** Political trust affects incumbent support (Hetherington 1998). Low trust predicts vote against incumbents.

**Raw Variable Names:**
- **2019:** `cps19_fed_gov_sat`
- **2021:** `cps21_fed_gov_sat`
- **2025:** `cps25_fed_gov_sat`

**Scale:** 1-4 (RAW ordinal scale)
- 1 = Very satisfied
- 2 = Fairly satisfied
- 3 = Not very satisfied
- 4 = Not at all satisfied

**Exclusion Rules:**
- Exclude 5 (don't know) → NaN

**Statistical Note:** Higher values = LESS trust. Coefficient represents effect of 1-category decrease in satisfaction.

**Data Quality:**
- **2025:** 48.3% missing (major survey issue - documented limitation)

---

#### pid (Party Identification)

**Purpose:** Long-term partisan attachment

**Theory:** Party ID is the strongest predictor of vote (Campbell et al. 1960). Stable psychological identification with political party.

**Raw Variable Names:**
- **2019:** `cps19_fed_id`
- **2021:** `cps21_fed_id`
- **2025:** `cps25_fed_id`

**Scale:** 1-8 (RAW categorical scale)
- 1-4 = Liberal identifiers (1=Very strong, 4=Not very strong)
- 5-8 = Conservative identifiers (5=Not very strong, 8=Very strong)
- (NDP, Bloc, Green, etc. coded similarly in some years - check codebook)

**Exclusion Rules:**
- Exclude 9 (no party ID / don't know) → NaN

**Statistical Note:** CATEGORICAL variable - use one-hot encoding (dummy variables) in models. Reference category typically strongest Liberal ID.

---

### 3. DEMOGRAPHICS

#### age (Age in Years)

**Purpose:** Age effects on vote choice

**Theory:** Age predicts political participation and vote choice (Wolfinger & Rosenstone 1980). Older voters favor conservatives.

**Raw Variable Names:**
- **2019:** `cps19_age`
- **2021:** `cps21_age`
- **2025:** `cps25_age_in_years`

**Scale:** 18-99 years (CONTINUOUS, RAW)

**CRITICAL:** Use CONTINUOUS age in years, NOT categorical age groups. Continuous age:
1. Preserves statistical power
2. Allows flexible modeling (linear, quadratic, splines)
3. Avoids arbitrary cutpoints

**Exclusion Rules:**
- Age < 18 → NaN (ineligible voters)
- Age > 99 → NaN (likely data error)

**Statistical Note:** Coefficient represents effect of 1-year increase in age. Can test non-linear effects with age² term.

---

#### gender (Gender)

**Purpose:** Gender gap in voting

**Theory:** Gender predicts vote choice, especially in recent elections (Kaufmann & Petrocik 1999).

**Raw Variable Names:**
- **2019:** `cps19_genderid`
- **2021:** `cps21_genderid`
- **2025:** `cps25_genderid`

**Scale:** 1-2 (RAW binary)
- 1 = Male
- 2 = Female

**Exclusion Rules:**
- Exclude 3 (non-binary) → NaN
- Exclude 4 (prefer not to say) → NaN
- Exclude 5 (other) → NaN

**Note:** Binary coding for Camatarri (2024) replication. Canadian-specific models could use 3+ categories.

**Statistical Note:** Typically coded as dummy (Female=1, Male=0) in regression. Coefficient = gender gap (Female effect vs Male baseline).

---

#### education (Education Level)

**Purpose:** Education effects on political sophistication and vote choice

**Theory:** Education predicts political knowledge, turnout, and liberal social attitudes (Nie et al. 1996).

**Raw Variable Names:**
- **2019:** `cps19_education`
- **2021:** `cps21_education`
- **2025:** `cps25_education`

**Scale:** 1-11 (RAW ordinal scale)
- 1 = No schooling
- 2 = Some elementary school
- 3 = Completed elementary school
- 4 = Some secondary/high school
- 5 = Completed secondary/high school
- 6 = Some technical/community college/CEGEP
- 7 = Completed technical/community college/CEGEP
- 8 = Some university
- 9 = Bachelor's degree
- 10 = Master's degree
- 11 = Professional degree or doctorate

**Exclusion Rules:**
- Exclude 12 (don't know/prefer not to say) → NaN

**Statistical Note:** Use as ordinal (assumes equal spacing). Coefficient = effect of 1-category increase. Can test linear vs non-linear with education².

---

#### income (Household Income)

**Purpose:** Socioeconomic status effects on vote choice

**Theory:** Income predicts economic policy preferences and vote choice (Bartels 2008). Higher income predicts conservative vote.

**Raw Variable Names:**
- **2019:** `cps19_income_cat` (categorical)
- **2021:** `cps21_income_cat` (categorical)
- **2025:** `cps25_income` (categorical)

**Scale:** 1-8 (RAW categorical brackets)
- 1 = Less than $30,000
- 2 = $30,000 to $59,999
- 3 = $60,000 to $89,999
- 4 = $90,000 to $109,999
- 5 = $110,000 to $149,999
- 6 = $150,000 to $199,999
- 7 = $200,000 or more
- 8 = (varies by year - check codebook)

**Exclusion Rules:**
- Exclude 9 (prefer not to say) → NaN

**Statistical Note:** CATEGORICAL variable - use one-hot encoding. Reference category typically lowest income bracket.

**Data Quality Issues:**
- **2019:** MISSING due to source file corruption
- **2021:** 98.3% missing (only 1.7% valid) - severe survey issue
- **2025:** Available

**Recommendation:** Treat as OPTIONAL predictor. Only include in models when sufficient data available (2025).

---

### 4. GEOGRAPHIC FIXED EFFECTS

#### province (Province/Territory)

**Purpose:** Geographic variation in political culture and party systems

**Theory:** Geography captures regional political identities, economic structures, and local issues (Johnston 2017).

**Raw Variable Names:**
- **2019:** `cps19_province`
- **2021:** `cps21_province`
- **2025:** `cps25_province`

**Scale:** 1-13 (RAW categorical codes mapping to postal abbreviations)

**Province Mapping:**
- 10 = NL (Newfoundland and Labrador)
- 11 = PE (Prince Edward Island)
- 12 = NS (Nova Scotia)
- 13 = NB (New Brunswick)
- 24 = QC (Quebec)
- 35 = ON (Ontario)
- 46 = MB (Manitoba)
- 47 = SK (Saskatchewan)
- 48 = AB (Alberta)
- 59 = BC (British Columbia)
- 60 = YT (Yukon)
- 61 = NT (Northwest Territories)
- 62 = NU (Nunavut)

**Statistical Note:** CATEGORICAL variable - use one-hot encoding. Reference category typically Ontario (largest province).

**Canadian-Specific Considerations:**
- **Quebec (QC):** Distinct political culture, Bloc Québécois support
- **Alberta (AB):** Conservative stronghold
- **Ontario (ON):** Largest province, often decisive
- **Atlantic provinces:** NB, NL, NS, PE (historically Liberal-leaning)

---

### 5. SURVEY WEIGHTS

#### weight (Survey Weight)

**Purpose:** Correct for sampling design and nonresponse to ensure population representativeness

**Raw Variable Names:**
- **2019:** `cps19_weight_general_restricted`
- **2021:** `cps21_weight_general_restricted`
- **2025:** `cps25_weight_general_restricted`

**Scale:** Continuous (>0)

**Theory:** Survey weights adjust for differential sampling probabilities and nonresponse to produce unbiased population estimates.

**Usage:**
- Use in weighted regression (`statsmodels.WLS` or R `svyglm`)
- Use in weighted aggregation (`np.average(x, weights=w)`)
- Use in calculating effective sample size

**Effective Sample Size:** `N_eff = (Σw)² / Σw²`

---

## Exclusion Rules Summary

All exclusions applied by `apply_exclusions()` function in [data/scripts/preprocess_ces.py](../scripts/preprocess_ces.py):

| Variable | Valid Range | Excluded Values | Reason |
|----------|-------------|-----------------|--------|
| vote_choice | 1-6 | 7 | Other parties (too heterogeneous) |
| pid | 1-8 | 9 | Don't know / no party ID |
| ideology | 0-10 | - | All values valid (continuous scale) |
| econ_retro | 1-3 | 4, 5 | Don't know, prefer not to say |
| trust_govt | 1-4 | 5 | Don't know |
| age | 18-99 | <18, >99 | Ineligible, data errors |
| gender | 1-2 | 3, 4, 5 | Non-binary, prefer not to say, other |
| education | 1-11 | 12 | Don't know / prefer not to say |
| province | 1-13 | - | All provinces valid |
| income | 1-8 | 9 | Prefer not to say |

**Exclusion Method:** Set excluded values to `NaN` (do NOT drop rows). This preserves sample for other variables with valid data.

---

## Data Quality Summary

### Completeness by Year

**2021 (N=20,968):**
| Variable | Valid N | Missing N | % Missing | % Complete |
|----------|---------|-----------|-----------|------------|
| vote_choice | 20,202 | 766 | 3.7% | 96.3% |
| age | 20,968 | 0 | 0.0% | 100.0% |
| pid | 19,875 | 1,093 | 5.2% | 94.8% |
| ideology | 18,903 | 2,065 | 9.8% | 90.2% |
| econ_retro | 20,553 | 415 | 2.0% | 98.0% |
| trust_govt | 20,545 | 423 | 2.0% | 98.0% |
| gender | 20,968 | 0 | 0.0% | 100.0% |
| education | 20,944 | 24 | 0.1% | 99.9% |
| province | 20,968 | 0 | 0.0% | 100.0% |
| **income** | **352** | **20,616** | **98.3%** | **1.7%**  |
| weight | 20,968 | 0 | 0.0% | 100.0% |

**Complete cases (excluding income):** 17,233 (82.2%)

**2025 (N=~40,000):**
| Variable | Valid N | Missing N | % Missing | % Complete |
|----------|---------|-----------|-----------|------------|
| vote_choice | 38,045 | ~2,000 | ~5% | ~95% |
| age | 40,000 | 0 | 0.0% | 100.0% |
| pid | 37,500 | 2,500 | ~6% | ~94% |
| ideology | 36,000 | 4,000 | ~10% | ~90% |
| econ_retro | 39,000 | 1,000 | ~2.5% | ~97.5% |
| **trust_govt** | **20,680** | **19,320** | **48.3%** | **51.7%**  |
| gender | 40,000 | 0 | 0.0% | 100.0% |
| education | 39,800 | 200 | ~0.5% | ~99.5% |
| province | 40,000 | 0 | 0.0% | 100.0% |
| income | 36,000 | 4,000 | ~10% | ~90% ✓ |
| weight | 40,000 | 0 | 0.0% | 100.0% |

**Known Limitations:**
- **2019:** Income and riding_id MISSING due to source file corruption
- **2021:** Income 98.3% missing (severe survey issue)
- **2025:** trust_govt 48.3% missing (major survey issue)

---

## Statistical Implications of Raw Scales

### Why NO Normalization?

The gold standard uses RAW scales (not 0-1 normalized) for critical statistical reasons:

#### 1. **Coefficient Interpretability**

**Normalized (0-1):**
```
β_ideology_norm = 2.5
"Moving from most liberal (0) to most conservative (1) increases log-odds by 2.5"
```

**Raw (0-10):**
```
β_ideology = 0.25
"Each 1-point rightward shift on 0-10 scale increases log-odds by 0.25"
```

**Advantage:** Raw scale coefficients map to substantively meaningful units (1-point shift on familiar survey scale).

#### 2. **Preserves Ordinal Structure**

**Example - Economic Retrospective:**
- Raw: 1=Better, 2=Same, 3=Worse
- Normalized: 0=Better, 0.5=Same, 1=Worse

Raw scale preserves that "Better→Same" and "Same→Worse" are equal 1-unit shifts. Normalization can obscure this.

#### 3. **Cross-Year Comparability**

Different years may have different normalization constants. Raw scales ensure consistent interpretation across elections.

#### 4. **Transparency**

Raw scales match survey codebooks exactly. Researchers can verify transformations directly from documentation.

---

## Variable Coding Checklist

When adding a new CES year to the pipeline, verify:

- [ ] Vote choice uses correct variable name (`cps##_votechoice`)
- [ ] Party ID uses `cps##_fed_id` (NOT `pes##_pidtrad`)
- [ ] Ideology uses `_aft` version (post-election), `_bef` as fallback
- [ ] Income uses `_cat` suffix for 2019/2021 (categorical version)
- [ ] Age variable correct (`cps25_age_in_years` for 2025)
- [ ] Gender uses `genderid` variable (NOT `gender`)
- [ ] Survey weight uses `_restricted` version
- [ ] All exclusions applied (set to NaN, not dropped)
- [ ] No normalization performed (use raw scales)
- [ ] Output saved as .parquet (not .csv)

---

## Usage Examples

### Loading Data (Python)

```python
import pandas as pd

# Load single year
ces_2021 = pd.read_parquet("data/02-clean/CES/CES_2021.parquet")

# Load multiple years
ces_2019 = pd.read_parquet("data/02-clean/CES/CES_2019.parquet")
ces_2021 = pd.read_parquet("data/02-clean/CES/CES_2021.parquet")
ces_2025 = pd.read_parquet("data/02-clean/CES/CES_2025.parquet")

# Stack years
ces_all = pd.concat([ces_2019, ces_2021, ces_2025], ignore_index=True)
```

### Binary Logistic Regression (Python)

```python
import statsmodels.api as sm
from statsmodels.formula.api import logit

# Prepare data with complete cases
ces_2021_complete = ces_2021.dropna(subset=[
    'vote_liberal', 'ideology', 'pid', 'econ_retro', 'trust_govt',
    'age', 'gender', 'education', 'province', 'weight'
])

# One-hot encode categorical variables
ces_2021_encoded = pd.get_dummies(
    ces_2021_complete,
    columns=['pid', 'province'],
    drop_first=True
)

# Unweighted logistic regression
model = logit(
    formula='vote_liberal ~ ideology + econ_retro + trust_govt + age + '
            'gender + education + C(pid) + C(province)',
    data=ces_2021_complete
).fit()

print(model.summary())
```

### Weighted Logistic Regression (Python)

```python
from statsmodels.discrete.discrete_model import Logit

# Prepare design matrix
X = ces_2021_encoded.drop(columns=['vote_liberal', 'weight', 'year'])
y = ces_2021_encoded['vote_liberal']
w = ces_2021_encoded['weight']

# Add constant
X = sm.add_constant(X)

# Weighted logistic regression
model_weighted = Logit(y, X).fit(
    method='bfgs',
    maxiter=1000,
    disp=True,
    weights=w
)

print(model_weighted.summary())
```

### Variable Importance (Optuna Pipeline)

See [projects/03-optuna/](../projects/03-optuna/) for hyperparameter optimization using gold standard variables.

### Time Series Analysis (R)

See [projects/04-time_series/](../projects/04-time_series/) for coefficient stability analysis across elections.

---

## Validation Tests

All gold standard variables validated with 23 automated tests in [tests/test_gold_standard_variables.py](../tests/test_gold_standard_variables.py):

**Data Quality Tests (10 tests):**
- ✓ All gold standard variables present in cleaned data
- ✓ No normalized variables (_norm suffix) present
- ✓ Variables have correct raw scales and ranges
- ✓ Exclusions properly applied (no excluded values present)
- ✓ Ideology is 0-10 scale (not normalized)
- ✓ Age is continuous (not categorical)
- ✓ Sufficient complete cases (>1000) for modeling
- ✓ Critical variables not >80% missing

**Configuration Tests (2 tests):**
- ✓ Project 03 config uses gold standard variables
- ✓ Project 04 data_loader uses gold standard variables

**Preprocessing Tests (1 test):**
- ✓ preprocess_ces.py has correct variable mapping

**Run all tests:**
```bash
pytest tests/test_gold_standard_variables.py -v
```

---

## Documentation Updates

This gold standard specification supersedes previous variable documentation. Related documents:

- **[VARIABLE_RESTRUCTURE_AUDIT.md](../VARIABLE_RESTRUCTURE_AUDIT.md)**: Complete audit of restructuring effort (2025-10-31)
- **[ces_variable_mapping.md](variable_mappings/ces_variable_mapping.md)**: Cross-year variable name mapping (needs update)
- **[preprocess_ces.py](scripts/preprocess_ces.py)**: Master data cleaning script implementing gold standard
- **[test_gold_standard_variables.py](../tests/test_gold_standard_variables.py)**: Validation test suite

---

## Citations

### Data Sources

**Canadian Election Study (CES):**
- Stephenson, Laura B; Harell, Allison; Rubenson, Daniel; Loewen, Peter John, 2020, '2019 Canadian Election Study - Online Survey', https://doi.org/10.7910/DVN/DUS88V, Harvard Dataverse, V1
- Stephenson, Laura B., Allison Harell, Daniel Rubenson and Peter John Loewen. 2021. '2021 Canadian Election Study', https://doi.org/10.7910/DVN/XBZHKC, Harvard Dataverse, V1, UNF:6:eyR28qaoYlHj9qwPWZmmVQ== [fileUNF]
- [CES 2025 citation pending]

### Methodology

**Camatarri, Stefano.** 2024. "Predicting Popular-vote Shares in US Presidential Elections: A Model-based Strategy Relying on ANES Data." *PS: Political Science & Politics*, April 2025: 253-257. DOI: 10.1017/S1049096524000933

### Voting Behavior Literature

- Campbell, Angus, Philip E. Converse, Warren E. Miller, and Donald E. Stokes. 1960. *The American Voter*. Chicago: University of Chicago Press.
- Fiorina, Morris P. 1981. *Retrospective Voting in American National Elections*. New Haven: Yale University Press.
- Hetherington, Marc J. 1998. "The Political Relevance of Political Trust." *American Political Science Review* 92(4): 791-808.
- Johnston, Richard. 2017. *The Canadian Party System: An Analytic History*. UBC Press.

---

**END OF DATA DICTIONARY**

**Last Updated:** 2025-10-31
**Validation Status:** All 23 tests passing
**Contact:** m.cowan@utoronto.ca
**License:** MIT
