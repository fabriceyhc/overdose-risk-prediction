#!/usr/bin/env python3
"""
Create visualizations of SUD facility proximity analysis.
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

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load the processed datasets."""
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    data_dir = base_dir / 'data' / 'processed'

    census_df = pd.read_csv(data_dir / 'la_county_census_tracts_facility_proximity_all.csv')
    zip_df = pd.read_csv(data_dir / 'la_county_zip_codes_facility_proximity_all.csv')

    return census_df, zip_df


def load_facilities():
    """Load SUD facilities GeoJSON."""
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    facilities_path = base_dir / 'data' / 'facilities' / 'SUD_Recovery_Treatment_Facilities_2025.11.25.geojson'

    with open(facilities_path, 'r') as f:
        facilities_data = json.load(f)

    facilities = []
    for feature in facilities_data['features']:
        props = feature['properties']
        coords = feature['geometry']['coordinates']

        facilities.append({
            'OBJECTID': props.get('OBJECTID'),
            'Facility_Name': props.get('Facility_Name'),
            'CountyName': props.get('CountyName'),
            'lon': coords[0],
            'lat': coords[1],
            'Program_Code': props.get('Program_Code'),
            'Target_Population': props.get('Target_Population'),
            'Treatment_Capacity': props.get('Treatment_Capacity', 0)
        })

    df = pd.DataFrame(facilities)
    df_la = df[df['CountyName'] == 'Los Angeles County'].copy()

    from shapely.geometry import Point
    geometry = [Point(row['lon'], row['lat']) for _, row in df_la.iterrows()]
    gdf = gpd.GeoDataFrame(df_la, geometry=geometry, crs="EPSG:4326")

    return gdf


def load_geographic_units():
    """Load census tracts and ZIP codes GeoJSON."""
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')

    census_path = base_dir / 'data' / 'geo' / '2020_Census_Tracts_2025.11.25.geojson'
    zip_path = base_dir / 'data' / 'geo' / 'LA_County_ZIP_Codes_2025.11.25.geojson'

    census_gdf = gpd.read_file(census_path)
    zip_gdf = gpd.read_file(zip_path)

    return census_gdf, zip_gdf


def create_distance_distribution_plots(census_df, zip_df, output_dir):
    """Create histograms showing distance to nearest facility."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Census tracts - km
    axes[0, 0].hist(census_df['nearest_facility_distance_km'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(census_df['nearest_facility_distance_km'].median(),
                       color='red', linestyle='--', label=f'Median: {census_df["nearest_facility_distance_km"].median():.2f} km')
    axes[0, 0].set_xlabel('Distance (km)')
    axes[0, 0].set_ylabel('Number of Census Tracts')
    axes[0, 0].set_title('Distance to Nearest SUD Facility - Census Tracts')
    axes[0, 0].legend()

    # Census tracts - miles
    axes[0, 1].hist(census_df['nearest_facility_distance_miles'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(census_df['nearest_facility_distance_miles'].median(),
                       color='red', linestyle='--', label=f'Median: {census_df["nearest_facility_distance_miles"].median():.2f} mi')
    axes[0, 1].set_xlabel('Distance (miles)')
    axes[0, 1].set_ylabel('Number of Census Tracts')
    axes[0, 1].set_title('Distance to Nearest SUD Facility - Census Tracts')
    axes[0, 1].legend()

    # ZIP codes - km
    axes[1, 0].hist(zip_df['nearest_facility_distance_km'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(zip_df['nearest_facility_distance_km'].median(),
                       color='red', linestyle='--', label=f'Median: {zip_df["nearest_facility_distance_km"].median():.2f} km')
    axes[1, 0].set_xlabel('Distance (km)')
    axes[1, 0].set_ylabel('Number of ZIP Codes')
    axes[1, 0].set_title('Distance to Nearest SUD Facility - ZIP Codes')
    axes[1, 0].legend()

    # ZIP codes - miles
    axes[1, 1].hist(zip_df['nearest_facility_distance_miles'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].axvline(zip_df['nearest_facility_distance_miles'].median(),
                       color='red', linestyle='--', label=f'Median: {zip_df["nearest_facility_distance_miles"].median():.2f} mi')
    axes[1, 1].set_xlabel('Distance (miles)')
    axes[1, 1].set_ylabel('Number of ZIP Codes')
    axes[1, 1].set_title('Distance to Nearest SUD Facility - ZIP Codes')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'distance_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'distance_distributions.png'}")
    plt.close()


def create_facility_count_plots(census_df, zip_df, output_dir):
    """Create bar plots showing facility counts within distance thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Census tracts - km
    km_cols = ['count_within_1km', 'count_within_5km', 'count_within_10km', 'count_within_15km', 'count_within_25km']
    km_means = [census_df[col].mean() for col in km_cols]
    km_labels = ['1 km', '5 km', '10 km', '15 km', '25 km']

    axes[0, 0].bar(km_labels, km_means, edgecolor='black', alpha=0.7)
    axes[0, 0].set_ylabel('Average Number of Facilities')
    axes[0, 0].set_title('Average Facility Count by Distance - Census Tracts')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Census tracts - miles
    mi_cols = ['count_within_1mi', 'count_within_3mi', 'count_within_5mi', 'count_within_10mi', 'count_within_15mi']
    mi_means = [census_df[col].mean() for col in mi_cols]
    mi_labels = ['1 mi', '3 mi', '5 mi', '10 mi', '15 mi']

    axes[0, 1].bar(mi_labels, mi_means, edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('Average Number of Facilities')
    axes[0, 1].set_title('Average Facility Count by Distance - Census Tracts')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # ZIP codes - km
    km_means_zip = [zip_df[col].mean() for col in km_cols]

    axes[1, 0].bar(km_labels, km_means_zip, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_ylabel('Average Number of Facilities')
    axes[1, 0].set_title('Average Facility Count by Distance - ZIP Codes')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # ZIP codes - miles
    mi_means_zip = [zip_df[col].mean() for col in mi_cols]

    axes[1, 1].bar(mi_labels, mi_means_zip, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_ylabel('Average Number of Facilities')
    axes[1, 1].set_title('Average Facility Count by Distance - ZIP Codes')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'facility_counts_by_distance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'facility_counts_by_distance.png'}")
    plt.close()


def create_summary_statistics_table(census_df, zip_df, output_dir):
    """Create summary statistics table."""

    print("\n" + "="*80)
    print("DETAILED SUMMARY STATISTICS")
    print("="*80)

    print("\nCENSUS TRACTS")
    print("-" * 80)
    print(f"Total census tracts: {len(census_df):,}")
    print(f"\nDistance to nearest facility (km):")
    print(f"  Mean:   {census_df['nearest_facility_distance_km'].mean():.2f} km")
    print(f"  Median: {census_df['nearest_facility_distance_km'].median():.2f} km")
    print(f"  Min:    {census_df['nearest_facility_distance_km'].min():.2f} km")
    print(f"  Max:    {census_df['nearest_facility_distance_km'].max():.2f} km")
    print(f"  Std:    {census_df['nearest_facility_distance_km'].std():.2f} km")

    print(f"\nDistance to nearest facility (miles):")
    print(f"  Mean:   {census_df['nearest_facility_distance_miles'].mean():.2f} mi")
    print(f"  Median: {census_df['nearest_facility_distance_miles'].median():.2f} mi")
    print(f"  Min:    {census_df['nearest_facility_distance_miles'].min():.2f} mi")
    print(f"  Max:    {census_df['nearest_facility_distance_miles'].max():.2f} mi")
    print(f"  Std:    {census_df['nearest_facility_distance_miles'].std():.2f} mi")

    print(f"\nFacility counts within distance thresholds (average):")
    print(f"  Within 1 km:  {census_df['count_within_1km'].mean():.2f} facilities")
    print(f"  Within 5 km:  {census_df['count_within_5km'].mean():.2f} facilities")
    print(f"  Within 10 km: {census_df['count_within_10km'].mean():.2f} facilities")
    print(f"  Within 15 km: {census_df['count_within_15km'].mean():.2f} facilities")
    print(f"  Within 25 km: {census_df['count_within_25km'].mean():.2f} facilities")

    print(f"\nTracts with zero facilities within distance:")
    print(f"  Within 1 km:  {(census_df['count_within_1km'] == 0).sum():,} tracts ({(census_df['count_within_1km'] == 0).sum()/len(census_df)*100:.1f}%)")
    print(f"  Within 5 km:  {(census_df['count_within_5km'] == 0).sum():,} tracts ({(census_df['count_within_5km'] == 0).sum()/len(census_df)*100:.1f}%)")
    print(f"  Within 10 km: {(census_df['count_within_10km'] == 0).sum():,} tracts ({(census_df['count_within_10km'] == 0).sum()/len(census_df)*100:.1f}%)")

    print("\n" + "-"*80)
    print("\nZIP CODES")
    print("-" * 80)
    print(f"Total ZIP codes: {len(zip_df):,}")
    print(f"\nDistance to nearest facility (km):")
    print(f"  Mean:   {zip_df['nearest_facility_distance_km'].mean():.2f} km")
    print(f"  Median: {zip_df['nearest_facility_distance_km'].median():.2f} km")
    print(f"  Min:    {zip_df['nearest_facility_distance_km'].min():.2f} km")
    print(f"  Max:    {zip_df['nearest_facility_distance_km'].max():.2f} km")
    print(f"  Std:    {zip_df['nearest_facility_distance_km'].std():.2f} km")

    print(f"\nDistance to nearest facility (miles):")
    print(f"  Mean:   {zip_df['nearest_facility_distance_miles'].mean():.2f} mi")
    print(f"  Median: {zip_df['nearest_facility_distance_miles'].median():.2f} mi")
    print(f"  Min:    {zip_df['nearest_facility_distance_miles'].min():.2f} mi")
    print(f"  Max:    {zip_df['nearest_facility_distance_miles'].max():.2f} mi")
    print(f"  Std:    {zip_df['nearest_facility_distance_miles'].std():.2f} mi")

    print(f"\nFacility counts within distance thresholds (average):")
    print(f"  Within 1 km:  {zip_df['count_within_1km'].mean():.2f} facilities")
    print(f"  Within 5 km:  {zip_df['count_within_5km'].mean():.2f} facilities")
    print(f"  Within 10 km: {zip_df['count_within_10km'].mean():.2f} facilities")
    print(f"  Within 15 km: {zip_df['count_within_15km'].mean():.2f} facilities")
    print(f"  Within 25 km: {zip_df['count_within_25km'].mean():.2f} facilities")

    print(f"\nZIP codes with zero facilities within distance:")
    print(f"  Within 1 km:  {(zip_df['count_within_1km'] == 0).sum():,} ZIP codes ({(zip_df['count_within_1km'] == 0).sum()/len(zip_df)*100:.1f}%)")
    print(f"  Within 5 km:  {(zip_df['count_within_5km'] == 0).sum():,} ZIP codes ({(zip_df['count_within_5km'] == 0).sum()/len(zip_df)*100:.1f}%)")
    print(f"  Within 10 km: {(zip_df['count_within_10km'] == 0).sum():,} ZIP codes ({(zip_df['count_within_10km'] == 0).sum()/len(zip_df)*100:.1f}%)")

    print("\n" + "="*80)


def create_facility_type_map(census_gdf, facilities_gdf, output_dir):
    """Create high-resolution map of LA County with facilities color-coded by program type."""
    print("\nCreating facility type map...")

    # Define color scheme for facility types
    program_colors = {
        'RES': '#2E86AB',          # Blue - Residential
        'RES-DETOX': '#A23B72',    # Purple - Residential Detox
        'DPH-DETOX': '#F18F01',    # Orange - DPH Detox
        'NON': '#C73E1D',          # Red - Non-residential/Outpatient
        'DSS': '#6A994E',          # Green - DSS
        'Other': '#808080'         # Gray - Other/Unknown
    }

    # Map program codes to colors
    def get_color(program_code):
        if pd.isna(program_code):
            return program_colors['Other']
        return program_colors.get(program_code, program_colors['Other'])

    facilities_gdf['color'] = facilities_gdf['Program_Code'].apply(get_color)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 24))

    # Plot census tracts
    census_gdf.plot(ax=ax, color='white', edgecolor='#cccccc', linewidth=0.3, alpha=0.8)

    # Plot facilities by type
    for program_code, color in program_colors.items():
        if program_code == 'Other':
            mask = ~facilities_gdf['Program_Code'].isin(['RES', 'RES-DETOX', 'DPH-DETOX', 'NON', 'DSS'])
        else:
            mask = facilities_gdf['Program_Code'] == program_code

        subset = facilities_gdf[mask]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, markersize=30, alpha=0.7, zorder=5, label=f'{program_code} ({len(subset)})')

    # Formatting
    ax.set_xlim(census_gdf.total_bounds[0], census_gdf.total_bounds[2])
    ax.set_ylim(census_gdf.total_bounds[1], census_gdf.total_bounds[3])
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title('LA County SUD Treatment Facilities by Program Type\n(Census Tract Boundaries)',
                 fontsize=18, fontweight='bold', pad=20)

    # Create custom legend
    legend_elements = []
    for program_code, color in program_colors.items():
        if program_code == 'Other':
            count = len(facilities_gdf[~facilities_gdf['Program_Code'].isin(['RES', 'RES-DETOX', 'DPH-DETOX', 'NON', 'DSS'])])
        else:
            count = len(facilities_gdf[facilities_gdf['Program_Code'] == program_code])

        if count > 0:
            label_text = f'{program_code}: {count} facilities'
            if program_code == 'RES':
                label_text += ' (Residential)'
            elif program_code == 'RES-DETOX':
                label_text += ' (Residential Detox)'
            elif program_code == 'DPH-DETOX':
                label_text += ' (DPH Detox)'
            elif program_code == 'NON':
                label_text += ' (Non-residential/Outpatient)'
            elif program_code == 'DSS':
                label_text += ' (Dept Social Services)'

            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color, markersize=10,
                                        label=label_text, alpha=0.7))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
              title='Facility Program Type', title_fontsize=12, framealpha=0.95)

    # Add info text
    info_text = f'Total Facilities: {len(facilities_gdf)}\nCensus Tracts: {len(census_gdf)}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.8))

    ax.set_aspect('equal')
    plt.tight_layout()

    # Save high-resolution version
    output_path = output_dir / 'map_facilities_by_type.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_access_quality_map(census_gdf, census_df, output_dir):
    """Create map showing treatment access quality by census tract."""
    print("\nCreating access quality map...")

    # Ensure CT20 columns have matching types
    census_gdf_copy = census_gdf.copy()
    census_df_copy = census_df.copy()
    census_gdf_copy['CT20'] = census_gdf_copy['CT20'].astype(str)
    census_df_copy['CT20'] = census_df_copy['CT20'].astype(str)

    # Merge proximity data with geographic boundaries
    census_gdf_merged = census_gdf_copy.merge(
        census_df_copy[['CT20', 'nearest_facility_distance_km', 'count_within_5km']],
        on='CT20',
        how='left'
    )

    # Categorize access quality
    def categorize_access(row):
        distance_km = row['nearest_facility_distance_km']
        if distance_km <= 1:
            return 'Excellent (≤1 km)'
        elif distance_km <= 3:
            return 'Good (1-3 km)'
        elif distance_km <= 5:
            return 'Fair (3-5 km)'
        elif distance_km <= 10:
            return 'Poor (5-10 km)'
        else:
            return 'Very Poor (>10 km)'

    census_gdf_merged['access_category'] = census_gdf_merged.apply(categorize_access, axis=1)

    # Define colors for access categories
    access_colors = {
        'Excellent (≤1 km)': '#2E7D32',      # Dark green
        'Good (1-3 km)': '#66BB6A',          # Light green
        'Fair (3-5 km)': '#FDD835',          # Yellow
        'Poor (5-10 km)': '#FB8C00',         # Orange
        'Very Poor (>10 km)': '#D32F2F'      # Red
    }

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 24))

    # Plot each category
    for category, color in access_colors.items():
        subset = census_gdf_merged[census_gdf_merged['access_category'] == category]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.2, alpha=0.8)

    # Formatting
    ax.set_xlim(census_gdf.total_bounds[0], census_gdf.total_bounds[2])
    ax.set_ylim(census_gdf.total_bounds[1], census_gdf.total_bounds[3])
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title('SUD Treatment Access Quality by Census Tract\n(Distance to Nearest Facility)',
                 fontsize=18, fontweight='bold', pad=20)

    # Create legend
    legend_elements = []
    for category, color in access_colors.items():
        count = len(census_gdf_merged[census_gdf_merged['access_category'] == category])
        pct = count / len(census_gdf_merged) * 100
        legend_elements.append(Patch(facecolor=color, edgecolor='white',
                                     label=f'{category}: {count} tracts ({pct:.1f}%)',
                                     alpha=0.8))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
              title='Access Quality', title_fontsize=12, framealpha=0.95)

    # Add statistics
    median_dist = census_gdf_merged['nearest_facility_distance_km'].median()
    treatment_deserts = (census_gdf_merged['count_within_5km'] == 0).sum()
    desert_pct = treatment_deserts / len(census_gdf_merged) * 100

    stats_text = f'Median Distance: {median_dist:.2f} km\nTreatment Deserts: {treatment_deserts} ({desert_pct:.1f}%)\n(0 facilities within 5km)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.9))

    ax.set_aspect('equal')
    plt.tight_layout()

    # Save
    output_path = output_dir / 'map_access_quality.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_facility_density_map(census_gdf, census_df, facilities_gdf, output_dir):
    """Create map showing facility density (count within 5km) by census tract."""
    print("\nCreating facility density map...")

    # Ensure CT20 columns have matching types
    census_gdf_copy = census_gdf.copy()
    census_df_copy = census_df.copy()
    census_gdf_copy['CT20'] = census_gdf_copy['CT20'].astype(str)
    census_df_copy['CT20'] = census_df_copy['CT20'].astype(str)

    # Merge proximity data with geographic boundaries
    census_gdf_merged = census_gdf_copy.merge(
        census_df_copy[['CT20', 'count_within_5km']],
        on='CT20',
        how='left'
    )

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 20))

    # Left panel: Choropleth of facility counts
    census_gdf_merged.plot(column='count_within_5km', ax=ax1, cmap='YlOrRd',
                          edgecolor='white', linewidth=0.2, alpha=0.8,
                          legend=True, legend_kwds={'label': 'Number of Facilities within 5km',
                                                     'orientation': 'vertical',
                                                     'shrink': 0.6})

    ax1.set_xlim(census_gdf.total_bounds[0], census_gdf.total_bounds[2])
    ax1.set_ylim(census_gdf.total_bounds[1], census_gdf.total_bounds[3])
    ax1.set_xlabel('Longitude', fontsize=14)
    ax1.set_ylabel('Latitude', fontsize=14)
    ax1.set_title('Facility Density by Census Tract\n(Count within 5km)',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.set_aspect('equal')

    # Add statistics
    mean_count = census_gdf_merged['count_within_5km'].mean()
    median_count = census_gdf_merged['count_within_5km'].median()
    max_count = census_gdf_merged['count_within_5km'].max()

    stats_text = f'Mean: {mean_count:.1f} facilities\nMedian: {median_count:.0f} facilities\nMax: {max_count:.0f} facilities'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round',
             facecolor='white', alpha=0.9))

    # Right panel: Combined view with tracts and facilities
    census_gdf_merged.plot(column='count_within_5km', ax=ax2, cmap='YlOrRd',
                          edgecolor='gray', linewidth=0.2, alpha=0.4)

    # Overlay facilities
    facilities_gdf.plot(ax=ax2, color='darkblue', markersize=15, alpha=0.6, zorder=5)

    ax2.set_xlim(census_gdf.total_bounds[0], census_gdf.total_bounds[2])
    ax2.set_ylim(census_gdf.total_bounds[1], census_gdf.total_bounds[3])
    ax2.set_xlabel('Longitude', fontsize=14)
    ax2.set_ylabel('Latitude', fontsize=14)
    ax2.set_title('Facility Locations Overlay\n(All {0} Facilities)'.format(len(facilities_gdf)),
                  fontsize=16, fontweight='bold', pad=15)
    ax2.set_aspect('equal')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'map_facility_density.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Setup
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    output_dir = base_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    census_df, zip_df = load_data()

    print("\nLoading geographic data for maps...")
    facilities_gdf = load_facilities()
    census_gdf, zip_gdf = load_geographic_units()

    print("\nCreating statistical visualizations...")
    create_distance_distribution_plots(census_df, zip_df, output_dir)
    create_facility_count_plots(census_df, zip_df, output_dir)

    print("\nCreating map visualizations...")
    create_facility_type_map(census_gdf, facilities_gdf, output_dir)
    create_access_quality_map(census_gdf, census_df, output_dir)
    create_facility_density_map(census_gdf, census_df, facilities_gdf, output_dir)

    print("\nGenerating summary statistics...")
    create_summary_statistics_table(census_df, zip_df, output_dir)

    print(f"\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir}/")
    print(f"\nGenerated files:")
    print("  - distance_distributions.png")
    print("  - facility_counts_by_distance.png")
    print("  - map_facilities_by_type.png (NEW)")
    print("  - map_access_quality.png (NEW)")
    print("  - map_facility_density.png (NEW)")
    print("="*80)


if __name__ == '__main__':
    main()
