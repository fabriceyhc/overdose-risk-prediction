#!/usr/bin/env python3
"""
STREAMLINED SCRIPT: Create comprehensive facility proximity dataset.

This script directly creates the comprehensive dataset with proximity metrics
for all treatment types in ONE pass, without creating intermediate filtered files.

Usage:
    python create_comprehensive_proximity_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Point
from typing import List, Tuple
import geopandas as gpd
from tqdm import tqdm


# ============================================================================
# TREATMENT TYPE DEFINITIONS
# ============================================================================

TREATMENT_FILTERS = {
    'any': {
        'name': 'Any SUD Facility',
        'filter': lambda df: df  # No filter, all facilities
    },
    'otp': {
        'name': 'Opioid Treatment Program (OTP)',
        'filter': lambda df: df[df['service_OTP'] == True]
    },
    'mat': {
        'name': 'Medication-Assisted Treatment (MAT)',
        'filter': lambda df: df[df['service_MAT'] == True]
    },
    'residential': {
        'name': 'Residential Facilities',
        'filter': lambda df: df[df['available_beds'].notna() &
                               df['available_beds'].str.contains(r'RS\(|RBH|R-WM', case=False, na=False)]
    },
    'outpatient': {
        'name': 'Outpatient Facilities',
        'filter': lambda df: df[df['service_OP'] == True]
    },
    'iop': {
        'name': 'Intensive Outpatient',
        'filter': lambda df: df[df['service_IOP'] == True]
    },
    'withdrawal_mgmt': {
        'name': 'Withdrawal Management (Detox)',
        'filter': lambda df: df[(df['service_A_WM'] == True) |
                               (df['service_R_WM'] == True) |
                               (df['service_I_WM'] == True) |
                               (df['available_beds'].notna() &
                                df['available_beds'].str.contains('WM|Detox', case=False, na=False))]
    },
    'co_occurring': {
        'name': 'Co-Occurring Disorder Capable',
        'filter': lambda df: df[df['service_Co_Occurring'] == True]
    },
    'men': {
        'name': 'Facilities Serving Men',
        'filter': lambda df: df[df['service_Male'] == True]
    },
    'women': {
        'name': 'Facilities Serving Women',
        'filter': lambda df: df[df['service_Female'] == True]
    },
    'youth': {
        'name': 'Youth/Adolescent Facilities',
        'filter': lambda df: df[df['service_Youth'] == True]
    }
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_all_facilities(facilities_path: str) -> pd.DataFrame:
    """Load all SUD facilities."""
    print("Loading SUD facilities...")

    df = pd.read_csv(facilities_path)
    df = df.dropna(subset=['latitude', 'longitude'])

    df = df.rename(columns={
        'agency_name': 'Facility_Name',
        'agency_address': 'Facility_Address',
        'latitude': 'lat',
        'longitude': 'lon'
    })

    print(f"Loaded {len(df)} facilities with valid coordinates")
    return df


def load_geographic_units(geo_path: str, unit_type: str) -> gpd.GeoDataFrame:
    """Load geographic units (census tracts or ZIP codes)."""
    print(f"Loading {unit_type}...")

    gdf = gpd.read_file(geo_path)

    print(f"Calculating centroids for {len(gdf)} {unit_type}...")
    gdf_projected = gdf.to_crs('EPSG:2229')
    centroids_projected = gdf_projected.geometry.centroid
    centroids_wgs84 = centroids_projected.to_crs('EPSG:4326')

    gdf['centroid_lat'] = centroids_wgs84.y
    gdf['centroid_lon'] = centroids_wgs84.x

    return gdf


# ============================================================================
# DISTANCE CALCULATION FUNCTIONS
# ============================================================================

def compute_distances_vectorized(geo_units_gdf: gpd.GeoDataFrame,
                                  facilities_df: pd.DataFrame) -> Tuple:
    """Compute distances from all geographic units to all facilities using vectorized operations."""

    geo_coords = np.array(list(zip(geo_units_gdf['centroid_lat'], geo_units_gdf['centroid_lon'])))
    facility_coords = np.array(list(zip(facilities_df['lat'], facilities_df['lon'])))

    geo_coords_rad = np.radians(geo_coords)
    facility_coords_rad = np.radians(facility_coords)

    lat1 = geo_coords_rad[:, 0:1]
    lon1 = geo_coords_rad[:, 1:2]
    lat2 = facility_coords_rad[:, 0:1].T
    lon2 = facility_coords_rad[:, 1:2].T

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    R_km = 6371.0
    R_miles = 3958.8

    distances_km = R_km * c
    distances_miles = R_miles * c

    return distances_km, distances_miles


def compute_proximity_metrics_for_filter(geo_gdf: gpd.GeoDataFrame,
                                         all_facilities_df: pd.DataFrame,
                                         filter_key: str,
                                         distance_thresholds_km: List[float] = [1, 5, 10, 15, 25],
                                         distance_thresholds_miles: List[float] = [1, 3, 5, 10, 15]) -> pd.DataFrame:
    """
    Compute proximity metrics for a specific facility filter.
    Returns a DataFrame with columns for this treatment type only.
    """
    filter_info = TREATMENT_FILTERS[filter_key]
    treatment_name = filter_info['name']

    print(f"  Computing metrics for: {treatment_name}")

    # Apply filter to get relevant facilities
    filtered_facilities = filter_info['filter'](all_facilities_df.copy())

    if len(filtered_facilities) == 0:
        print(f"    WARNING: No facilities found for {filter_key}")
        # Return empty metrics
        result_df = pd.DataFrame(index=geo_gdf.index)
        suffix = filter_key

        result_df[f'nearest_facility_{suffix}_id'] = np.nan
        result_df[f'nearest_facility_{suffix}_name'] = ''
        result_df[f'nearest_facility_{suffix}_distance_km'] = np.inf
        result_df[f'nearest_facility_{suffix}_distance_miles'] = np.inf

        for threshold in distance_thresholds_km:
            result_df[f'count_{suffix}_within_{int(threshold)}km'] = 0
        for threshold in distance_thresholds_miles:
            result_df[f'count_{suffix}_within_{int(threshold)}mi'] = 0

        return result_df

    print(f"    Found {len(filtered_facilities)} facilities")

    # Compute distances
    distances_km, distances_miles = compute_distances_vectorized(geo_gdf, filtered_facilities)

    # Find nearest facility
    nearest_indices = np.argmin(distances_km, axis=1)
    nearest_distances_km = np.min(distances_km, axis=1)
    nearest_distances_miles = np.min(distances_miles, axis=1)

    # Get facility info
    facility_ids = filtered_facilities.index.values
    facility_names = filtered_facilities['Facility_Name'].values

    # Create result DataFrame with proper suffix
    suffix = filter_key
    result_df = pd.DataFrame(index=geo_gdf.index)

    result_df[f'nearest_facility_{suffix}_id'] = facility_ids[nearest_indices]
    result_df[f'nearest_facility_{suffix}_name'] = facility_names[nearest_indices]
    result_df[f'nearest_facility_{suffix}_distance_km'] = np.round(nearest_distances_km, 3)
    result_df[f'nearest_facility_{suffix}_distance_miles'] = np.round(nearest_distances_miles, 3)

    # Count facilities within distance thresholds
    for threshold_km in distance_thresholds_km:
        counts = np.sum(distances_km <= threshold_km, axis=1)
        result_df[f'count_{suffix}_within_{int(threshold_km)}km'] = counts

    for threshold_mi in distance_thresholds_miles:
        counts = np.sum(distances_miles <= threshold_mi, axis=1)
        result_df[f'count_{suffix}_within_{int(threshold_mi)}mi'] = counts

    # Print summary
    median_dist = result_df[f'nearest_facility_{suffix}_distance_km'].median()
    mean_count_5km = result_df[f'count_{suffix}_within_5km'].mean()
    zero_count = (result_df[f'count_{suffix}_within_5km'] == 0).sum()
    zero_pct = zero_count / len(result_df) * 100

    print(f"    Median distance: {median_dist:.2f} km")
    print(f"    Mean count within 5km: {mean_count_5km:.1f}")
    print(f"    Treatment deserts: {zero_count} ({zero_pct:.1f}%)")

    return result_df


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def create_comprehensive_dataset(geo_type='census_tracts'):
    """
    Create comprehensive dataset with proximity metrics for all treatment types.

    Args:
        geo_type: Either 'census_tracts' or 'zip_codes'
    """
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    data_dir = base_dir / 'data'

    print(f"\n{'='*80}")
    print(f"CREATING COMPREHENSIVE {geo_type.upper().replace('_', ' ')} DATASET")
    print(f"{'='*80}\n")

    # Load data
    facilities_path = data_dir / 'facilities' / 'sudhelpla_agencies_20251126_233819.csv'
    all_facilities = load_all_facilities(str(facilities_path))

    if geo_type == 'census_tracts':
        geo_path = data_dir / 'geo' / '2020_Census_Tracts_2025.11.25.geojson'
    else:
        geo_path = data_dir / 'geo' / 'LA_County_ZIP_Codes_2025.11.25.geojson'

    geo_gdf = load_geographic_units(str(geo_path), geo_type)

    # Start with geographic columns
    print(f"\nStarting with {len(geo_gdf.columns)} geographic columns")

    # Process each treatment type
    print(f"\n{'='*80}")
    print("COMPUTING PROXIMITY METRICS FOR ALL TREATMENT TYPES")
    print(f"{'='*80}\n")

    # Collect all proximity metrics in a list for efficient concatenation
    all_metrics = []
    for filter_key in tqdm(TREATMENT_FILTERS.keys(), desc="Processing treatment types"):
        proximity_metrics = compute_proximity_metrics_for_filter(
            geo_gdf,
            all_facilities,
            filter_key
        )
        all_metrics.append(proximity_metrics)

    # Concatenate all metrics at once to avoid DataFrame fragmentation
    print("\nCombining all metrics into comprehensive dataset...")
    comprehensive_gdf = pd.concat([geo_gdf] + all_metrics, axis=1)

    # Ensure it's still a GeoDataFrame
    comprehensive_gdf = gpd.GeoDataFrame(comprehensive_gdf, geometry='geometry', crs=geo_gdf.crs)

    # Print summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Total rows: {len(comprehensive_gdf):,}")
    print(f"Total columns: {len(comprehensive_gdf.columns)}")
    print(f"\nColumn breakdown:")

    # Count geographic vs proximity columns
    proximity_patterns = ['nearest_facility_', 'count_', '_within_']
    geo_cols = [col for col in comprehensive_gdf.columns
                if not any(pattern in col for pattern in proximity_patterns)]
    proximity_cols = len(comprehensive_gdf.columns) - len(geo_cols)

    print(f"  Geographic columns: {len(geo_cols)}")
    print(f"  Proximity columns: {proximity_cols}")
    print(f"    ({proximity_cols // len(TREATMENT_FILTERS)} metrics × {len(TREATMENT_FILTERS)} treatment types)")

    # Save outputs
    output_dir = data_dir / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    if geo_type == 'census_tracts':
        csv_file = output_dir / 'la_county_census_tracts_comprehensive.csv'
        geojson_file = output_dir / 'la_county_census_tracts_comprehensive.geojson'
    else:
        csv_file = output_dir / 'la_county_zip_codes_comprehensive.csv'
        geojson_file = output_dir / 'la_county_zip_codes_comprehensive.geojson'

    print(f"\n{'='*80}")
    print("SAVING OUTPUTS")
    print(f"{'='*80}")

    # Save CSV
    print(f"\nSaving CSV: {csv_file.name}")
    csv_df = comprehensive_gdf.drop(columns=['geometry'])
    csv_df.to_csv(csv_file, index=False)
    print(f"✓ Saved {len(csv_df):,} rows, {len(csv_df.columns)} columns")

    # Save GeoJSON
    print(f"\nSaving GeoJSON: {geojson_file.name}")
    comprehensive_gdf.to_file(geojson_file, driver='GeoJSON')
    print(f"✓ Saved GeoJSON")

    print(f"\n{'='*80}")
    print(f"{geo_type.upper().replace('_', ' ')} DATASET COMPLETE!")
    print(f"{'='*80}\n")

    return comprehensive_gdf


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("STREAMLINED COMPREHENSIVE PROXIMITY DATASET CREATOR")
    print("="*80)
    print("\nThis script creates comprehensive datasets with proximity metrics")
    print("for all treatment types in a single pass (no intermediate files).")
    print("="*80)

    # Create comprehensive datasets for both census tracts and ZIP codes
    print("\n" + "="*80)
    print("PART 1: CENSUS TRACTS")
    print("="*80)
    census_gdf = create_comprehensive_dataset('census_tracts')

    print("\n" + "="*80)
    print("PART 2: ZIP CODES")
    print("="*80)
    zip_gdf = create_comprehensive_dataset('zip_codes')

    # Final summary
    print("\n" + "="*80)
    print("ALL COMPREHENSIVE DATASETS CREATED!")
    print("="*80)
    print("\nOutput files:")
    print("  - data/processed/la_county_census_tracts_comprehensive.csv")
    print("  - data/processed/la_county_census_tracts_comprehensive.geojson")
    print("  - data/processed/la_county_zip_codes_comprehensive.csv")
    print("  - data/processed/la_county_zip_codes_comprehensive.geojson")
    print("\nNo intermediate files created - all metrics computed in one pass!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
