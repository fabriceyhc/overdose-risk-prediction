# SUD Facility Proximity Analysis
## Geospatial Access to Treatment in Los Angeles County

**Study Area:** Los Angeles County, California
**Generated:** November 25, 2025
**Geographic Coverage:** 2,496 Census Tracts | 313 ZIP Codes
**Facilities Analyzed:** 764 SUD Treatment Facilities

---

## Overview

This analysis quantifies geographic access to Substance Use Disorder (SUD) treatment facilities across Los Angeles County. For each census tract and ZIP code, we calculate:

1. **Distance to nearest facility** (km and miles)
2. **Count of facilities within distance thresholds** (1, 5, 10, 15, 25 km/mi)

The analysis provides **5 filtered datasets** to address different research questions about facility type (residential vs. outpatient), target population (men, women, youth), and treatment availability.

---

## Key Findings

### Overall Access
- **Median distance:** 1.14 km (0.71 miles) to nearest facility
- **96.5%** of census tracts have ≥1 facility within 5km
- **88 census tracts (3.5%)** are "treatment deserts" with zero facilities within 5km
- Urban cores (West Hollywood, Downtown LA) have exceptional density: 300-350 facilities within 10 miles

### Critical Gaps Identified

#### 1. Residential Treatment Beds (44% Worse Access)
- **44% of facilities are outpatient only** (no beds)
- Median distance to residential facility: **1.64 km** vs **1.14 km** (all facilities)
- **175 census tracts (7%)** have no residential facilities within 5km
- **Impact:** Housing-insecure populations face significantly worse access to intensive care

#### 2. Youth/Adolescent Services (5× Farther)
- Only **22 youth facilities** (3% of total) in entire LA County
- Median distance: **6.11 km** (5× farther than adult facilities)
- **63% of census tracts** have ZERO youth facilities within 5km
- **Impact:** Critical shortage for adolescent treatment needs

#### 3. Gender Equity (Minimal Disparity)
- Men: 746 facilities (98%) | Women: 728 facilities (95%)
- Mean distance difference: **0.012 km** (essentially equal)
- **90% of census tracts** have identical nearest facility for both genders
- **Impact:** Gender access is already equitable; focus efforts elsewhere

---

## Available Datasets

All datasets in `data/processed/` with both CSV and GeoJSON formats:

| Dataset | Facilities | Median Distance | Treatment Deserts¹ | Primary Use Case |
|---------|-----------|-----------------|-------------------|------------------|
| **All** | 764 | 1.14 km (0.71 mi) | 3.5% | General facility availability |
| **Residential** | 437 | 1.64 km (1.02 mi) | 7.0% | Bed availability, housing-insecure |
| **Men** | 746 | 1.16 km (0.72 mi) | 4.0% | Male-specific access |
| **Women** | 728 | 1.17 km (0.73 mi) | 4.2% | Female-specific, pregnancy/childcare |
| **Youth** | 22 | 6.11 km (3.80 mi) | 62.7% | Adolescent treatment access |

¹ % of census tracts with zero facilities within 5km

### File Naming Convention
```
la_county_{geography}_facility_proximity_{filter}.{format}

Examples:
- la_county_census_tracts_facility_proximity_all.csv
- la_county_zip_codes_facility_proximity_residential.geojson
- la_county_census_tracts_facility_proximity_women.csv
```

---

## Quick Start

### Loading Data

```python
import pandas as pd
import geopandas as gpd

# Load census tract data (CSV)
df = pd.read_csv('data/processed/la_county_census_tracts_facility_proximity_all.csv')

# Load geospatial data for mapping (GeoJSON)
gdf = gpd.read_file('data/processed/la_county_census_tracts_facility_proximity_all.geojson')

# Load a filtered dataset (e.g., residential only)
residential_df = pd.read_csv('data/processed/la_county_census_tracts_facility_proximity_residential.csv')
```

### Common Analyses

```python
# Find treatment deserts (zero facilities within 5km)
deserts = df[df['count_within_5km'] == 0]

# Find areas with poor residential access
poor_access = residential_df[residential_df['nearest_facility_distance_km'] > 5]

# Compare men vs women access
men = pd.read_csv('data/processed/la_county_census_tracts_facility_proximity_men.csv')
women = pd.read_csv('data/processed/la_county_census_tracts_facility_proximity_women.csv')
disparity = women['nearest_facility_distance_km'] - men['nearest_facility_distance_km']

# Categorize access quality
df['access_category'] = pd.cut(
    df['nearest_facility_distance_km'],
    bins=[0, 1, 3, 5, 10, 100],
    labels=['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']
)
```

---

## Data Columns

Each dataset includes:

| Column | Description |
|--------|-------------|
| `CT20` / `ZIPCODE` | Census tract ID (2020) or ZIP code |
| `LABEL` | Human-readable label (census tracts only) |
| `centroid_lat` | Latitude of geographic unit centroid (WGS84) |
| `centroid_lon` | Longitude of geographic unit centroid (WGS84) |
| `nearest_facility_id` | OBJECTID of nearest facility |
| `nearest_facility_name` | Name of nearest facility |
| `nearest_facility_distance_km` | Distance to nearest (kilometers) |
| `nearest_facility_distance_miles` | Distance to nearest (miles) |
| `count_within_1km` to `count_within_25km` | Facility counts at 1, 5, 10, 15, 25 km |
| `count_within_1mi` to `count_within_15mi` | Facility counts at 1, 3, 5, 10, 15 miles |


---

## Methodology

### Distance Calculation
- **Centroid computation:** NAD83 California State Plane Zone 5 (EPSG:2229) for accuracy
- **Distance metric:** Geodesic distance (Haversine formula) accounts for Earth's curvature
- **Performance:** Vectorized NumPy operations compute ~2.1M pairwise distances in seconds

### Facility Filtering

**Residential:** RES, RES-DETOX, DPH-DETOX, DSS program codes (facilities with beds)

**Men:** CO-ED, MEN ONLY, MEN/YOUTH, DUAL DIAGNOSIS target populations

**Women:** CO-ED, WOMEN ONLY, WOMEN/CHILDREN, WOMEN/YOUTH, DUAL DIAGNOSIS

**Youth:** YOUTH, ADOLESCENT, CO-ED/YOUTH target populations

### Data Sources
- **SUD Facilities:** California Department of Health Care Services
- **Census Tracts:** 2020 US Census
- **ZIP Codes:** LA County boundaries

---

## Reproducing the Analysis

### Prerequisites
```bash
pip install -r requirements.txt
```

Required packages: `geopandas`, `geopy`, `shapely`, `pandas`, `numpy`, `tqdm`

### Run Analysis
```bash
# Generate all filtered datasets
python3 scripts/create_facility_proximity_dataset_filtered.py

# Create visualizations
python3 scripts/visualize_facility_proximity.py
python3 scripts/compare_filtered_datasets.py

# Run example queries
python3 examples/example_queries.py
```

---

## Use Cases

### Research Applications
1. **Health Equity Analysis** - Correlate with socioeconomic indicators, race/ethnicity, overdose rates
2. **Predictive Modeling** - Use proximity as feature in overdose risk models
3. **Service Utilization** - Model how distance affects treatment engagement
4. **Policy Evaluation** - Assess impact of new facility locations

### Which Dataset to Use?

**General treatment access** → Use "All"

**Housing-insecure populations** → Use "Residential" (bed availability)

**Gender-specific studies** → Use "Men" or "Women" (differences are minimal)

**Youth interventions** → Use "Youth" (note: severe shortage identified)

**Comparative analysis** → Use multiple datasets to contrast access patterns

---

## Visualizations

Generated plots in `figures/`:

- `distance_distributions.png` - Distance to nearest facility histograms
- `facility_counts_by_distance.png` - Average counts by distance threshold
- `comparison_median_distances.png` - Cross-filter median distance comparison
- `comparison_zero_facilities.png` - Treatment desert prevalence by filter
- `comparison_distributions.png` - Distance distribution by filter type
- `comparison_facility_counts.png` - Facility density comparison

---

## Limitations

### Distance vs. Accessibility
Geodesic distance does not account for:
- Road networks / travel time
- Public transportation availability
- Traffic patterns
- Facility hours of operation

### Facility Characteristics
Analysis does not consider:
- Current availability / waitlists
- Insurance acceptance (Medi-Cal, private)
- Language services / cultural competency
- Specific treatment modalities (MAT, CBT, etc.)

### Capacity Considerations
- Treatment_Capacity = 0 indicates outpatient (still provides services)
- Does not reflect current occupancy or waitlist status
- May not include all bed types

### Temporal Snapshot
- Data from November 2025
- Facility status may change (openings, closures, program changes)

---

## Future Enhancements

Potential additional analyses:
- **Capacity-weighted distance** - Weight by number of beds available
- **Service-specific filters** - MAT providers, detox-only, dual diagnosis
- **Insurance filters** - Medi-Cal accepting facilities
- **Travel time analysis** - Integrate road network and transit data
- **Waitlist data** - Actual availability vs. theoretical access

---

## Project Structure

```
overdose-risk-prediction/
├── data/
│   ├── facilities/          # Source facility data
│   ├── geo/                 # Geographic boundary files
│   └── processed/           # Generated datasets (20 files)
├── figures/                 # Visualizations (7 plots)
├── scripts/
│   ├── create_facility_proximity_dataset_filtered.py  # Main analysis
│   ├── visualize_facility_proximity.py                # Basic viz
│   ├── compare_filtered_datasets.py                   # Comparison viz
│   └── example_queries.py                             # Usage examples
├── examples/outputs/        # Example filtered outputs
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## Citation

If using this analysis in research or publications, please cite the data sources:

- **SUD Facilities:** California Department of Health Care Services (DHCS)
- **Geographic Data:** U.S. Census Bureau (2020 boundaries)
- **Analysis:** [Your institution/name], November 2025

---

## Contact & Support

For questions about:
- **Methodology:** See `scripts/create_facility_proximity_dataset_filtered.py`
- **Data columns:** See "Data Columns" section above
- **Usage examples:** See `examples/example_queries.py`

---

## Summary Statistics

### Census Tracts (n=2,496)
```
All Facilities:      1.14 km median | 96.5% have ≥1 within 5km
Residential:         1.64 km median | 93.0% have ≥1 within 5km
Men:                 1.16 km median | 96.0% have ≥1 within 5km
Women:               1.17 km median | 95.8% have ≥1 within 5km
Youth:               6.11 km median | 37.3% have ≥1 within 5km
```

### ZIP Codes (n=313)
```
All Facilities:      1.28 km median | 89.1% have ≥1 within 5km
Residential:         2.07 km median | 85.3% have ≥1 within 5km
Men:                 1.28 km median | 88.8% have ≥1 within 5km
Women:               1.29 km median | 88.8% have ≥1 within 5km
Youth:               7.12 km median | 31.3% have ≥1 within 5km
```

---

**Analysis Complete** - All datasets ready for integration into overdose risk prediction models and health equity research.
