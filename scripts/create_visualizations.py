#!/usr/bin/env python3
"""
MASTER SCRIPT: Create all facility proximity visualizations.

This script combines the functionality of:
1. visualize_facility_proximity.py - General visualizations for all facilities
2. visualize_treatment_type.py - Treatment-specific visualizations

Usage:
    python visualize_all_proximity_data.py                    # Create general visualizations (all facilities)
    python visualize_all_proximity_data.py --treatment otp    # Create OTP-specific visualizations
    python visualize_all_proximity_data.py --treatment all    # Create visualizations for all treatment types
    python visualize_all_proximity_data.py --list             # List available treatment types
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Treatment type names
TREATMENT_NAMES = {
    'all': 'All SUD',
    'otp': 'OTP (Opioid Treatment Program)',
    'mat': 'MAT (Medication-Assisted Treatment)',
    'residential': 'Residential',
    'outpatient': 'Outpatient',
    'iop': 'Intensive Outpatient',
    'withdrawal_mgmt': 'Withdrawal Management',
    'co_occurring': 'Co-Occurring Disorder',
    'men': 'Men\'s',
    'women': 'Women\'s',
    'youth': 'Youth'
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_data(filter_type='all'):
    """Load the comprehensive datasets and extract columns for specific treatment type."""
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    data_dir = base_dir / 'data' / 'processed'

    # Load comprehensive datasets
    census_df = pd.read_csv(data_dir / 'la_county_census_tracts_comprehensive.csv')
    zip_df = pd.read_csv(data_dir / 'la_county_zip_codes_comprehensive.csv')

    # Map filter_type to column prefix (in comprehensive dataset, 'all' is called 'any')
    col_prefix = 'any' if filter_type == 'all' else filter_type

    # Create simplified dataframes with standardized column names for the treatment type
    census_filtered = extract_treatment_columns(census_df, col_prefix)
    zip_filtered = extract_treatment_columns(zip_df, col_prefix)

    return census_filtered, zip_filtered


def extract_treatment_columns(df, treatment_type):
    """Extract columns for a specific treatment type and rename them to standard names."""
    # Base columns to keep
    if 'CT20' in df.columns:  # Census tracts
        base_cols = ['OBJECTID', 'CT20', 'LABEL', 'ShapeSTArea', 'ShapeSTLength',
                     'centroid_lat', 'centroid_lon', 'geometry']
    else:  # ZIP codes
        base_cols = ['OBJECTID', 'ZIPCODE', 'PO_NAME', 'Shape_Area', 'Shape_Leng',
                     'centroid_lat', 'centroid_lon', 'geometry']

    # Get columns that exist in this dataframe
    existing_base = [col for col in base_cols if col in df.columns]
    result_df = df[existing_base].copy()

    # Add treatment-specific columns with standardized names
    # Distance columns
    result_df['nearest_facility_id'] = df.get(f'nearest_facility_{treatment_type}_id')
    result_df['nearest_facility_name'] = df.get(f'nearest_facility_{treatment_type}_name')
    result_df['nearest_facility_distance_km'] = df.get(f'nearest_facility_{treatment_type}_distance_km')
    result_df['nearest_facility_distance_miles'] = df.get(f'nearest_facility_{treatment_type}_distance_miles')

    # Count columns (km)
    for dist in [1, 5, 10, 15, 25]:
        result_df[f'count_within_{dist}km'] = df.get(f'count_{treatment_type}_within_{dist}km')

    # Count columns (miles)
    for dist in [1, 3, 5, 10, 15]:
        result_df[f'count_within_{dist}mi'] = df.get(f'count_{treatment_type}_within_{dist}mi')

    return result_df


def load_facilities(filter_type='all'):
    """Load facilities filtered by treatment type."""
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    facilities_path = base_dir / 'data' / 'facilities' / 'sudhelpla_agencies_20251126_233819.csv'

    df = pd.read_csv(facilities_path)
    df = df.dropna(subset=['latitude', 'longitude'])

    df = df.rename(columns={
        'agency_name': 'Facility_Name',
        'agency_address': 'Facility_Address',
        'latitude': 'lat',
        'longitude': 'lon'
    })

    # Apply filter
    if filter_type == 'otp':
        df = df[df['service_OTP'] == True].copy()
    elif filter_type == 'mat':
        df = df[df['service_MAT'] == True].copy()
    elif filter_type == 'residential':
        df = df[df['available_beds'].notna() &
                df['available_beds'].str.contains(r'RS\(|RBH|R-WM', case=False, na=False)].copy()
    elif filter_type == 'outpatient':
        df = df[df['service_OP'] == True].copy()
    elif filter_type == 'iop':
        df = df[df['service_IOP'] == True].copy()
    elif filter_type == 'withdrawal_mgmt':
        df = df[((df['service_A_WM'] == True) |
                 (df['service_R_WM'] == True) |
                 (df['service_I_WM'] == True) |
                 (df['available_beds'].notna() &
                  df['available_beds'].str.contains('WM|Detox', case=False, na=False)))].copy()
    elif filter_type == 'co_occurring':
        df = df[df['service_Co_Occurring'] == True].copy()
    elif filter_type == 'men':
        df = df[df['service_Male'] == True].copy()
    elif filter_type == 'women':
        df = df[df['service_Female'] == True].copy()
    elif filter_type == 'youth':
        df = df[df['service_Youth'] == True].copy()

    from shapely.geometry import Point
    geometry = [Point(row['lon'], row['lat']) for _, row in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    return gdf


def load_geographic_units():
    """Load census tracts and ZIP codes GeoJSON."""
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    census_path = base_dir / 'data' / 'geo' / '2020_Census_Tracts_2025.11.25.geojson'
    zip_path = base_dir / 'data' / 'geo' / 'LA_County_ZIP_Codes_2025.11.25.geojson'

    census_gdf = gpd.read_file(census_path)
    zip_gdf = gpd.read_file(zip_path)

    return census_gdf, zip_gdf


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_distance_distribution_plots(census_df, zip_df, output_dir, treatment_name, filter_type):
    """Create histograms showing distance to nearest facility."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Census tracts - km
    axes[0, 0].hist(census_df['nearest_facility_distance_km'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(census_df['nearest_facility_distance_km'].median(),
                       color='red', linestyle='--', label=f'Median: {census_df["nearest_facility_distance_km"].median():.2f} km')
    axes[0, 0].set_xlabel('Distance (km)')
    axes[0, 0].set_ylabel('Number of Census Tracts')
    axes[0, 0].set_title(f'Distance to Nearest {treatment_name} Facility - Census Tracts')
    axes[0, 0].legend()

    # Census tracts - miles
    axes[0, 1].hist(census_df['nearest_facility_distance_miles'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(census_df['nearest_facility_distance_miles'].median(),
                       color='red', linestyle='--', label=f'Median: {census_df["nearest_facility_distance_miles"].median():.2f} mi')
    axes[0, 1].set_xlabel('Distance (miles)')
    axes[0, 1].set_ylabel('Number of Census Tracts')
    axes[0, 1].set_title(f'Distance to Nearest {treatment_name} Facility - Census Tracts')
    axes[0, 1].legend()

    # ZIP codes - km
    axes[1, 0].hist(zip_df['nearest_facility_distance_km'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(zip_df['nearest_facility_distance_km'].median(),
                       color='red', linestyle='--', label=f'Median: {zip_df["nearest_facility_distance_km"].median():.2f} km')
    axes[1, 0].set_xlabel('Distance (km)')
    axes[1, 0].set_ylabel('Number of ZIP Codes')
    axes[1, 0].set_title(f'Distance to Nearest {treatment_name} Facility - ZIP Codes')
    axes[1, 0].legend()

    # ZIP codes - miles
    axes[1, 1].hist(zip_df['nearest_facility_distance_miles'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].axvline(zip_df['nearest_facility_distance_miles'].median(),
                       color='red', linestyle='--', label=f'Median: {zip_df["nearest_facility_distance_miles"].median():.2f} mi')
    axes[1, 1].set_xlabel('Distance (miles)')
    axes[1, 1].set_ylabel('Number of ZIP Codes')
    axes[1, 1].set_title(f'Distance to Nearest {treatment_name} Facility - ZIP Codes')
    axes[1, 1].legend()

    plt.tight_layout()
    filename = 'distance_distributions.png' if filter_type == 'all' else f'distance_distributions_{filter_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / filename}")
    plt.close()


def create_facility_count_plots(census_df, zip_df, output_dir, treatment_name, filter_type):
    """Create bar plots showing facility counts within distance thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    km_cols = ['count_within_1km', 'count_within_5km', 'count_within_10km', 'count_within_15km', 'count_within_25km']
    km_means = [census_df[col].mean() for col in km_cols]
    km_labels = ['1 km', '5 km', '10 km', '15 km', '25 km']

    axes[0, 0].bar(km_labels, km_means, edgecolor='black', alpha=0.7)
    axes[0, 0].set_ylabel('Average Number of Facilities')
    axes[0, 0].set_title(f'Average {treatment_name} Facility Count by Distance - Census Tracts')
    axes[0, 0].grid(axis='y', alpha=0.3)

    mi_cols = ['count_within_1mi', 'count_within_3mi', 'count_within_5mi', 'count_within_10mi', 'count_within_15mi']
    mi_means = [census_df[col].mean() for col in mi_cols]
    mi_labels = ['1 mi', '3 mi', '5 mi', '10 mi', '15 mi']

    axes[0, 1].bar(mi_labels, mi_means, edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('Average Number of Facilities')
    axes[0, 1].set_title(f'Average {treatment_name} Facility Count by Distance - Census Tracts')
    axes[0, 1].grid(axis='y', alpha=0.3)

    km_means_zip = [zip_df[col].mean() for col in km_cols]
    axes[1, 0].bar(km_labels, km_means_zip, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_ylabel('Average Number of Facilities')
    axes[1, 0].set_title(f'Average {treatment_name} Facility Count by Distance - ZIP Codes')
    axes[1, 0].grid(axis='y', alpha=0.3)

    mi_means_zip = [zip_df[col].mean() for col in mi_cols]
    axes[1, 1].bar(mi_labels, mi_means_zip, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_ylabel('Average Number of Facilities')
    axes[1, 1].set_title(f'Average {treatment_name} Facility Count by Distance - ZIP Codes')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filename = 'facility_counts_by_distance.png' if filter_type == 'all' else f'facility_counts_by_distance_{filter_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / filename}")
    plt.close()


def create_facility_locations_map(census_gdf, facilities_gdf, output_dir, treatment_name, filter_type):
    """Create high-resolution map showing facility locations."""
    print(f"\nCreating {treatment_name} facility locations map...")

    fig, ax = plt.subplots(1, 1, figsize=(20, 24))
    census_gdf.plot(ax=ax, color='white', edgecolor='#cccccc', linewidth=0.3, alpha=0.8)
    facilities_gdf.plot(ax=ax, color='#A23B72', markersize=50, alpha=0.8, zorder=5)

    ax.set_xlim(census_gdf.total_bounds[0], census_gdf.total_bounds[2])
    ax.set_ylim(census_gdf.total_bounds[1], census_gdf.total_bounds[3])
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title(f'LA County {treatment_name} Facilities\n(Census Tract Boundaries)',
                 fontsize=18, fontweight='bold', pad=20)

    info_text = f'Total {treatment_name} Facilities: {len(facilities_gdf)}\nCensus Tracts: {len(census_gdf)}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.8))

    ax.set_aspect('equal')
    plt.tight_layout()

    filename = 'map_facilities_by_type.png' if filter_type == 'all' else f'map_{filter_type}_facilities.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_access_quality_map(census_gdf, census_df, output_dir, treatment_name, filter_type):
    """Create map showing treatment access quality by census tract."""
    print(f"\nCreating {treatment_name} access quality map...")

    census_gdf_copy = census_gdf.copy()
    census_df_copy = census_df.copy()
    census_gdf_copy['CT20'] = census_gdf_copy['CT20'].astype(str)
    census_df_copy['CT20'] = census_df_copy['CT20'].astype(str)

    census_gdf_merged = census_gdf_copy.merge(
        census_df_copy[['CT20', 'nearest_facility_distance_km', 'count_within_5km']],
        on='CT20',
        how='left'
    )

    def categorize_access(row):
        distance_km = row['nearest_facility_distance_km']
        if distance_km <= 3:
            return 'Excellent (≤3 km)'
        elif distance_km <= 5:
            return 'Good (3-5 km)'
        elif distance_km <= 10:
            return 'Fair (5-10 km)'
        elif distance_km <= 15:
            return 'Poor (10-15 km)'
        else:
            return 'Very Poor (>15 km)'

    census_gdf_merged['access_category'] = census_gdf_merged.apply(categorize_access, axis=1)

    access_colors = {
        'Excellent (≤3 km)': '#2E7D32',
        'Good (3-5 km)': '#66BB6A',
        'Fair (5-10 km)': '#FDD835',
        'Poor (10-15 km)': '#FB8C00',
        'Very Poor (>15 km)': '#D32F2F'
    }

    fig, ax = plt.subplots(1, 1, figsize=(20, 24))

    for category, color in access_colors.items():
        subset = census_gdf_merged[census_gdf_merged['access_category'] == category]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.2, alpha=0.8)

    ax.set_xlim(census_gdf.total_bounds[0], census_gdf.total_bounds[2])
    ax.set_ylim(census_gdf.total_bounds[1], census_gdf.total_bounds[3])
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title(f'{treatment_name} Access Quality by Census Tract\n(Distance to Nearest Facility)',
                 fontsize=18, fontweight='bold', pad=20)

    legend_elements = []
    for category, color in access_colors.items():
        count = len(census_gdf_merged[census_gdf_merged['access_category'] == category])
        pct = count / len(census_gdf_merged) * 100
        legend_elements.append(Patch(facecolor=color, edgecolor='white',
                                     label=f'{category}: {count} tracts ({pct:.1f}%)',
                                     alpha=0.8))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
              title='Access Quality', title_fontsize=12, framealpha=0.95)

    median_dist = census_gdf_merged['nearest_facility_distance_km'].median()
    treatment_deserts = (census_gdf_merged['count_within_5km'] == 0).sum()
    desert_pct = treatment_deserts / len(census_gdf_merged) * 100

    stats_text = f'Median Distance: {median_dist:.2f} km\n{treatment_name} Deserts: {treatment_deserts} ({desert_pct:.1f}%)\n(0 facilities within 5km)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.9))

    ax.set_aspect('equal')
    plt.tight_layout()

    filename = 'map_access_quality.png' if filter_type == 'all' else f'map_{filter_type}_access_quality.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_facility_density_map(census_gdf, census_df, facilities_gdf, output_dir, treatment_name, filter_type):
    """Create map showing facility density by census tract."""
    print(f"\nCreating {treatment_name} facility density map...")

    census_gdf_copy = census_gdf.copy()
    census_df_copy = census_df.copy()
    census_gdf_copy['CT20'] = census_gdf_copy['CT20'].astype(str)
    census_df_copy['CT20'] = census_df_copy['CT20'].astype(str)

    census_gdf_merged = census_gdf_copy.merge(
        census_df_copy[['CT20', 'count_within_5km']],
        on='CT20',
        how='left'
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 20))

    census_gdf_merged.plot(column='count_within_5km', ax=ax1, cmap='YlOrRd',
                          edgecolor='white', linewidth=0.2, alpha=0.8,
                          legend=True, legend_kwds={'label': f'Number of {treatment_name} Facilities within 5km',
                                                     'orientation': 'vertical',
                                                     'shrink': 0.6})

    ax1.set_xlim(census_gdf.total_bounds[0], census_gdf.total_bounds[2])
    ax1.set_ylim(census_gdf.total_bounds[1], census_gdf.total_bounds[3])
    ax1.set_xlabel('Longitude', fontsize=14)
    ax1.set_ylabel('Latitude', fontsize=14)
    ax1.set_title(f'{treatment_name} Facility Density by Census Tract\n(Count within 5km)',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.set_aspect('equal')

    mean_count = census_gdf_merged['count_within_5km'].mean()
    median_count = census_gdf_merged['count_within_5km'].median()
    max_count = census_gdf_merged['count_within_5km'].max()

    stats_text = f'Mean: {mean_count:.1f} facilities\nMedian: {median_count:.0f} facilities\nMax: {max_count:.0f} facilities'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round',
             facecolor='white', alpha=0.9))

    census_gdf_merged.plot(column='count_within_5km', ax=ax2, cmap='YlOrRd',
                          edgecolor='gray', linewidth=0.2, alpha=0.4)
    facilities_gdf.plot(ax=ax2, color='darkblue', markersize=20, alpha=0.7, zorder=5)

    ax2.set_xlim(census_gdf.total_bounds[0], census_gdf.total_bounds[2])
    ax2.set_ylim(census_gdf.total_bounds[1], census_gdf.total_bounds[3])
    ax2.set_xlabel('Longitude', fontsize=14)
    ax2.set_ylabel('Latitude', fontsize=14)
    ax2.set_title(f'{treatment_name} Facility Locations Overlay\n(All {len(facilities_gdf)} Facilities)',
                  fontsize=16, fontweight='bold', pad=15)
    ax2.set_aspect('equal')

    plt.tight_layout()

    filename = 'map_facility_density.png' if filter_type == 'all' else f'map_{filter_type}_density.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_statistics(census_df, zip_df, treatment_name):
    """Print summary statistics."""
    print("\n" + "="*80)
    print(f"{treatment_name.upper()} - SUMMARY STATISTICS")
    print("="*80)

    print("\nCENSUS TRACTS")
    print("-" * 80)
    print(f"Total: {len(census_df):,}")
    print(f"Distance to nearest (median): {census_df['nearest_facility_distance_km'].median():.2f} km ({census_df['nearest_facility_distance_miles'].median():.2f} mi)")
    print(f"Facilities within 5km (mean): {census_df['count_within_5km'].mean():.1f}")
    print(f"Treatment deserts (0 within 5km): {(census_df['count_within_5km'] == 0).sum()} ({(census_df['count_within_5km'] == 0).sum()/len(census_df)*100:.1f}%)")

    print("\nZIP CODES")
    print("-" * 80)
    print(f"Total: {len(zip_df):,}")
    print(f"Distance to nearest (median): {zip_df['nearest_facility_distance_km'].median():.2f} km ({zip_df['nearest_facility_distance_miles'].median():.2f} mi)")
    print(f"Facilities within 5km (mean): {zip_df['count_within_5km'].mean():.1f}")
    print(f"Treatment deserts (0 within 5km): {(zip_df['count_within_5km'] == 0).sum()} ({(zip_df['count_within_5km'] == 0).sum()/len(zip_df)*100:.1f}%)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def create_visualizations_for_treatment(filter_type):
    """Create all visualizations for a specific treatment type."""
    treatment_name = TREATMENT_NAMES.get(filter_type, filter_type.upper())

    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')

    # Create output directory
    if filter_type == 'all':
        output_dir = base_dir / 'figures'
    else:
        output_dir = base_dir / 'figures' / filter_type

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS FOR: {treatment_name.upper()}")
    print(f"{'='*80}\n")

    print("Loading data...")
    census_df, zip_df = load_data(filter_type)

    print("Loading geographic data...")
    facilities_gdf = load_facilities(filter_type)
    census_gdf, zip_gdf = load_geographic_units()

    print(f"\nFound {len(facilities_gdf)} {treatment_name} facilities")

    print("\nCreating statistical visualizations...")
    create_distance_distribution_plots(census_df, zip_df, output_dir, treatment_name, filter_type)
    create_facility_count_plots(census_df, zip_df, output_dir, treatment_name, filter_type)

    print("\nCreating map visualizations...")
    create_facility_locations_map(census_gdf, facilities_gdf, output_dir, treatment_name, filter_type)
    create_access_quality_map(census_gdf, census_df, output_dir, treatment_name, filter_type)
    create_facility_density_map(census_gdf, census_df, facilities_gdf, output_dir, treatment_name, filter_type)

    print_summary_statistics(census_df, zip_df, treatment_name)

    print(f"\n{'='*80}")
    print(f"{treatment_name.upper()} VISUALIZATIONS COMPLETE!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}/")
    print(f"{'='*80}\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Create facility proximity visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                         # Create visualizations for all facilities
  %(prog)s --treatment otp         # Create OTP-specific visualizations
  %(prog)s --treatment all-types   # Create visualizations for ALL treatment types
  %(prog)s --list                  # List available treatment types
        '''
    )

    parser.add_argument(
        '--treatment',
        type=str,
        default='all',
        help='Treatment type to visualize (default: all)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available treatment types and exit'
    )

    args = parser.parse_args()

    # List treatment types
    if args.list:
        print("\nAvailable treatment types:")
        print("-" * 40)
        for key, name in TREATMENT_NAMES.items():
            print(f"  {key:<20} {name}")
        print("-" * 40)
        print("\nSpecial options:")
        print("  all-types            Create visualizations for ALL treatment types")
        print("-" * 40)
        return

    # Create visualizations
    print("\n" + "="*80)
    print("MASTER SCRIPT: CREATE ALL VISUALIZATIONS")
    print("="*80)

    if args.treatment == 'all-types':
        print("\nCreating visualizations for ALL treatment types...")
        for filter_type in TREATMENT_NAMES.keys():
            try:
                create_visualizations_for_treatment(filter_type)
            except Exception as e:
                print(f"\nERROR processing {filter_type}: {e}")
                print("Continuing with next treatment type...")
    else:
        if args.treatment not in TREATMENT_NAMES:
            print(f"\nERROR: Unknown treatment type '{args.treatment}'")
            print("Use --list to see available treatment types")
            sys.exit(1)

        create_visualizations_for_treatment(args.treatment)

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
