# SUD Facility Proximity Analysis
## Comprehensive Geospatial Access to Treatment in Los Angeles County

**Study Area:** Los Angeles County, California
**Generated:** November 27, 2025
**Geographic Coverage:** 2,496 Census Tracts | 313 ZIP Codes
**Facilities Analyzed:** 756 SUD Treatment Facilities
**Treatment Types:** 11 (OTP, MAT, Residential, Outpatient, IOP, Withdrawal Mgmt, Co-Occurring, Men's, Women's, Youth, All)

---

## Overview

This analysis provides **comprehensive geospatial access metrics** for Substance Use Disorder (SUD) treatment facilities across Los Angeles County. The dataset includes proximity measurements to **11 different treatment types** in a single unified file.

### What's Included

For each census tract and ZIP code, we calculate distance and facility counts for:

1. **Treatment Modalities:**
   - **OTP** - Opioid Treatment Program (76 facilities)
   - **MAT** - Medication-Assisted Treatment (102 facilities)
   - **IOP** - Intensive Outpatient (180 facilities)
   - **Outpatient** - Outpatient services (494 facilities)
   - **Residential** - Residential treatment (392 facilities)
   - **Withdrawal Management** - Detox services (90 facilities)
   - **Co-Occurring** - Dual diagnosis capable (103 facilities)

2. **Population-Specific:**
   - **Men's Facilities** (272 facilities)
   - **Women's Facilities** (265 facilities)
   - **Youth/Adolescent** (57 facilities)

3. **All Facilities** (756 facilities - any treatment type)

### Metrics Computed

For each treatment type:
- Distance to nearest facility (km and miles)
- Count of facilities within 1, 5, 10, 15, 25 km
- Count of facilities within 1, 3, 5, 10, 15 miles

---

## 🎯 Quick Start

### Loading the Comprehensive Dataset

```python
import pandas as pd
import geopandas as gpd

# Load comprehensive dataset (all treatment types in one file)
df = pd.read_csv('data/processed/la_county_census_tracts_comprehensive.csv')

# Check OTP access
otp_deserts = df[df['count_otp_within_5km'] == 0]
print(f"Census tracts with no OTP within 5km: {len(otp_deserts)} ({len(otp_deserts)/len(df)*100:.1f}%)")

# Compare MAT vs OTP access
df['mat_advantage'] = df['nearest_facility_mat_distance_km'] - df['nearest_facility_otp_distance_km']
print(f"Tracts closer to MAT than OTP: {(df['mat_advantage'] < 0).sum()}")

# Find comprehensive treatment deserts (no facilities of any type within 5km)
comprehensive_deserts = df[df['count_any_within_5km'] == 0]
print(f"Total treatment deserts: {len(comprehensive_deserts)}")

# Load geospatial data for mapping
gdf = gpd.read_file('data/processed/la_county_census_tracts_comprehensive.geojson')
```

### Example Analyses

```python
# 1. Multi-modality access analysis
df['has_otp_5km'] = df['count_otp_within_5km'] > 0
df['has_mat_5km'] = df['count_mat_within_5km'] > 0
df['has_residential_5km'] = df['count_residential_within_5km'] > 0

multi_access = df[df['has_otp_5km'] & df['has_mat_5km'] & df['has_residential_5km']]
print(f"Tracts with full treatment spectrum: {len(multi_access)} ({len(multi_access)/len(df)*100:.1f}%)")

# 2. Treatment modality comparison
comparison = df[[
    'nearest_facility_any_distance_km',
    'nearest_facility_otp_distance_km',
    'nearest_facility_mat_distance_km',
    'nearest_facility_residential_distance_km'
]].median()
print("\nMedian distance to nearest facility:")
print(comparison)

# 3. Youth treatment access gaps
youth_gaps = df[
    (df['count_any_within_5km'] > 0) &  # Has adult treatment
    (df['count_youth_within_5km'] == 0)  # No youth treatment
]
print(f"\nAreas with adult but no youth treatment: {len(youth_gaps)}")

# 4. Categorize comprehensive access
df['access_score'] = (
    (df['count_otp_within_5km'] > 0).astype(int) +
    (df['count_mat_within_5km'] > 0).astype(int) +
    (df['count_residential_within_5km'] > 0).astype(int) +
    (df['count_iop_within_5km'] > 0).astype(int)
)
print("\nAccess score distribution (0-4 modalities):")
print(df['access_score'].value_counts().sort_index())
```

---

## 📊 Key Findings

### OTP (Opioid Treatment Program) Access
- **76 OTP facilities** across LA County
- Median distance: **4.98 km (3.09 miles)**
- **49.7%** of census tracts are **OTP deserts** (no facility within 5km)
- Average OTP facilities within 5km: **1.78**

### MAT (Medication-Assisted Treatment) Access
- **102 MAT facilities**
- Median distance: **4.78 km (2.97 miles)**
- **44.4%** are MAT deserts
- Often co-located with primary care (better distributed than OTP)

### Residential Treatment Access
- **392 residential facilities** (with beds)
- Median distance: **2.35 km (1.46 miles)**
- **16.9%** are residential deserts
- Critical for housing-insecure populations

### Youth/Adolescent Services (Critical Gap)
- Only **57 youth facilities** (7.5% of total)
- Median distance: **3.93 km** (2× farther than adult facilities)
- **36.7%** of tracts have ZERO youth facilities within 5km
- Severe shortage for adolescent treatment needs

### Overall Access (Any Facility)
- **756 total facilities**
- Median distance: **1.82 km (1.13 miles)**
- **9.1%** are complete treatment deserts
- Urban cores have exceptional density (100+ facilities within 10km)

---

## 📁 Dataset Structure

### Comprehensive Dataset (Recommended)

Single file with **161 columns** covering all treatment types:

**File:** `data/processed/la_county_census_tracts_comprehensive.csv` (2.2 MB)

**Column Structure:**
```
Geographic Columns (8):
- OBJECTID, CT20, LABEL, ShapeSTArea, ShapeSTLength, centroid_lat, centroid_lon, geometry

Proximity Metrics (14 columns × 11 treatment types = 154 columns):
For each treatment type (any, otp, mat, residential, outpatient, iop,
                        withdrawal_mgmt, co_occurring, men, women, youth):

- nearest_facility_{type}_id
- nearest_facility_{type}_name
- nearest_facility_{type}_distance_km
- nearest_facility_{type}_distance_miles
- count_{type}_within_1km, 5km, 10km, 15km, 25km
- count_{type}_within_1mi, 3mi, 5mi, 10mi, 15mi
```

### Available Datasets

| File | Size | Description |
|------|------|-------------|
| `la_county_census_tracts_comprehensive.csv` | 2.2 MB | All treatment types, census tracts |
| `la_county_census_tracts_comprehensive.geojson` | 31 MB | Same, with geometry for mapping |
| `la_county_zip_codes_comprehensive.csv` | 280 KB | All treatment types, ZIP codes |
| `la_county_zip_codes_comprehensive.geojson` | 21 MB | Same, with geometry for mapping |

---

## 🔧 Reproducing the Analysis

### Prerequisites
```bash
pip install pandas numpy geopandas shapely tqdm matplotlib seaborn
```

### Two-Script Workflow

#### 1. Generate Dataset (Run First)
```bash
python scripts/create_dataset.py
```

**What it does:**
- Loads 756 facilities from `data/facilities/sudhelpla_agencies_20251126_233819.csv`
- Calculates distances for all 11 treatment types in **one pass**
- Outputs comprehensive datasets (no intermediate files)
- Runtime: ~3-5 minutes for both census tracts and ZIP codes

**Outputs:**
- `data/processed/la_county_census_tracts_comprehensive.csv`
- `data/processed/la_county_census_tracts_comprehensive.geojson`
- `data/processed/la_county_zip_codes_comprehensive.csv`
- `data/processed/la_county_zip_codes_comprehensive.geojson`

#### 2. Create Visualizations (Optional)
```bash
# Visualize all facilities (general overview)
python scripts/create_visualizations.py

# Visualize specific treatment type
python scripts/create_visualizations.py --treatment otp
python scripts/create_visualizations.py --treatment mat
python scripts/create_visualizations.py --treatment residential

# Create visualizations for ALL treatment types
python scripts/create_visualizations.py --treatment all-types

# List available treatment types
python scripts/create_visualizations.py --list
```

**What it creates:**
- Distance distribution histograms
- Facility count bar plots
- High-resolution maps (facility locations, access quality, density)
- Summary statistics

**Outputs:** `figures/` (for all) or `figures/{treatment_type}/` (for specific types)

---

## 📋 Column Reference

### Geographic Columns
- `CT20` / `ZIPCODE` - Geographic unit identifier
- `LABEL` - Human-readable name (census tracts only)
- `centroid_lat`, `centroid_lon` - Centroid coordinates (WGS84)
- `geometry` - Polygon geometry (GeoJSON only)

### Treatment-Specific Columns

**Format:** `{metric}_{treatment_type}_{measure}`

**Example for OTP:**
```
nearest_facility_otp_id              # Facility ID
nearest_facility_otp_name            # Facility name
nearest_facility_otp_distance_km     # Distance in km
nearest_facility_otp_distance_miles  # Distance in miles
count_otp_within_1km                 # Count within 1 km
count_otp_within_5km                 # Count within 5 km
count_otp_within_10km                # Count within 10 km
count_otp_within_15km                # Count within 15 km
count_otp_within_25km                # Count within 25 km
count_otp_within_1mi                 # Count within 1 mile
count_otp_within_3mi                 # Count within 3 miles
count_otp_within_5mi                 # Count within 5 miles
count_otp_within_10mi                # Count within 10 miles
count_otp_within_15mi                # Count within 15 miles
```

**Treatment Type Suffixes:**
- `any` - Any SUD facility
- `otp` - Opioid Treatment Program
- `mat` - Medication-Assisted Treatment
- `residential` - Residential facilities
- `outpatient` - Outpatient services
- `iop` - Intensive Outpatient
- `withdrawal_mgmt` - Withdrawal management/detox
- `co_occurring` - Co-occurring disorder capable
- `men` - Men's facilities
- `women` - Women's facilities
- `youth` - Youth/adolescent facilities

---

## 🔬 Methodology

### Facility Classification

**Treatment types determined by service flags and bed availability:**

- **OTP:** `service_OTP == True`
- **MAT:** `service_MAT == True`
- **Residential:** Facilities with `available_beds` containing "RS(", "RBH", or "R-WM"
- **Outpatient:** `service_OP == True`
- **IOP:** `service_IOP == True`
- **Withdrawal Mgmt:** Any of `service_A_WM`, `service_R_WM`, `service_I_WM == True`, or "WM"/"Detox" in `available_beds`
- **Co-Occurring:** `service_Co_Occurring == True`
- **Men:** `service_Male == True`
- **Women:** `service_Female == True`
- **Youth:** `service_Youth == True`

### Distance Calculation

1. **Centroid Computation:** Uses NAD83 California State Plane Zone 5 (EPSG:2229) for accurate centroid calculation
2. **Distance Metric:** Haversine formula (geodesic distance) accounts for Earth's curvature
3. **Performance:** Vectorized NumPy operations compute all distances in seconds

### Data Sources

- **SUD Facilities:** LA County SUDHelpLA database (November 26, 2025)
- **Census Tracts:** 2020 US Census boundaries
- **ZIP Codes:** LA County ZIP code boundaries

---

## 📈 Summary Statistics

### Census Tracts (n=2,496)

| Treatment Type | Median Distance | Mean Facilities<br/>within 5km | Treatment Deserts<br/>(0 within 5km) |
|----------------|-----------------|--------------------------------|--------------------------------------|
| **Any Facility** | 1.82 km (1.13 mi) | 22.6 | 9.1% |
| **OTP** | 4.98 km (3.09 mi) | 1.8 | 49.7% |
| **MAT** | 4.78 km (2.97 mi) | 2.2 | 44.4% |
| **Residential** | 2.35 km (1.46 mi) | 11.7 | 16.9% |
| **Outpatient** | 1.47 km (0.91 mi) | 14.3 | 5.6% |
| **IOP** | 2.76 km (1.71 mi) | 5.0 | 28.3% |
| **Withdrawal Mgmt** | 3.61 km (2.24 mi) | 2.8 | 34.7% |
| **Co-Occurring** | 2.77 km (1.72 mi) | 2.9 | 27.6% |
| **Men's** | 2.51 km (1.56 mi) | 7.7 | 20.1% |
| **Women's** | 3.75 km (2.33 mi) | 7.4 | 32.6% |
| **Youth** | 3.93 km (2.44 mi) | 1.5 | 36.7% |

### ZIP Codes (n=313)

| Treatment Type | Median Distance | Mean Facilities<br/>within 5km | Treatment Deserts<br/>(0 within 5km) |
|----------------|-----------------|--------------------------------|--------------------------------------|
| **Any Facility** | 2.24 km (1.39 mi) | 19.4 | 18.2% |
| **OTP** | 5.65 km (3.51 mi) | 1.5 | 56.2% |
| **MAT** | 5.08 km (3.16 mi) | 2.0 | 50.5% |

---

## 🎯 Use Cases

### 1. Overdose Risk Modeling
Use proximity metrics as features in predictive models:
```python
features = [
    'nearest_facility_mat_distance_km',
    'count_otp_within_5km',
    'count_residential_within_10km',
    'nearest_facility_any_distance_km'
]
```

### 2. Health Equity Analysis
Correlate treatment access with socioeconomic indicators:
```python
# Join with census socioeconomic data
equity_df = pd.merge(
    df[['CT20', 'nearest_facility_otp_distance_km', 'count_mat_within_5km']],
    census_demographics,
    on='CT20'
)
```

### 3. Policy Evaluation
Assess impact of new facility locations:
```python
# Before/after comparison
baseline = df[['CT20', 'count_otp_within_5km']].copy()
# Simulate new facility at location X
# Recalculate distances
# Compare coverage improvement
```

### 4. Treatment Desert Identification
Target areas for intervention:
```python
critical_deserts = df[
    (df['count_otp_within_5km'] == 0) &
    (df['count_mat_within_5km'] == 0) &
    (df['count_residential_within_5km'] == 0)
]
```

---

## ⚠️ Limitations

### Distance vs. Accessibility
Geodesic distance does not account for:
- Road networks / actual travel time
- Public transportation routes and schedules
- Traffic patterns
- Physical barriers (highways, mountains, etc.)

### Facility Characteristics Not Included
- Current availability / waitlists
- Insurance acceptance
- Language services / cultural competency
- Specific treatment protocols
- Quality of care metrics

### Temporal Considerations
- Data snapshot from November 2025
- Facility status changes over time
- Does not reflect real-time bed availability

---

## 🗂️ Project Structure

```
overdose-risk-prediction/
├── data/
│   ├── facilities/                    # Source facility data
│   │   └── sudhelpla_agencies_20251126_233819.csv
│   ├── geo/                           # Geographic boundary files
│   │   ├── 2020_Census_Tracts_2025.11.25.geojson
│   │   └── LA_County_ZIP_Codes_2025.11.25.geojson
│   └── processed/                     # Generated comprehensive datasets
│       ├── la_county_census_tracts_comprehensive.csv
│       ├── la_county_census_tracts_comprehensive.geojson
│       ├── la_county_zip_codes_comprehensive.csv
│       └── la_county_zip_codes_comprehensive.geojson
├── figures/                           # Visualizations (all facilities)
│   └── {treatment_type}/              # Treatment-specific visualizations
├── scripts/
│   ├── create_dataset.py              # Generate comprehensive datasets
│   └── create_visualizations.py       # Create all visualizations
└── README.md                          # This file
```

---

## 🚀 What Makes This Comprehensive?

### Single Unified Dataset
- **No need for multiple files** - All treatment types in one place
- **Easy comparisons** - All metrics use same geography
- **Consistent methodology** - Same distance calculations across all types

### 11 Treatment Types
Most analyses only consider "any facility" - this provides:
- **Treatment modality access** (OTP, MAT, IOP, Outpatient, Residential, Detox)
- **Population-specific access** (Men, Women, Youth)
- **Capability filters** (Co-occurring disorders)

### Complete Metrics
- Multiple distance thresholds (1, 5, 10, 15, 25 km/mi)
- Both metric and imperial units
- Facility counts AND distances
- 154 proximity columns total

---

## 📚 Citation

If using this analysis in research or publications:

```bibtex
@dataset{la_sud_proximity_2025,
  title={Comprehensive SUD Facility Proximity Analysis: Los Angeles County},
  author={[Your Name/Institution]},
  year={2025},
  month={November},
  note={756 facilities, 11 treatment types, 2,496 census tracts}
}
```

**Data Sources:**
- LA County SUDHelpLA Facility Database (November 2025)
- U.S. Census Bureau 2020 Boundaries
- California Department of Health Care Services

---

## 🤝 Support

**For questions about:**
- **Methodology:** See code in `scripts/create_dataset.py`
- **Data columns:** See "Column Reference" section above
- **Visualizations:** Run `python scripts/create_visualizations.py --help`

---

**Analysis Complete** ✅
Ready for integration into overdose risk prediction models, health equity research, and policy evaluation.
