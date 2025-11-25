#!/usr/bin/env python3
"""
Create visualizations of SUD facility proximity analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load the processed datasets."""
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    data_dir = base_dir / 'data' / 'processed'

    census_df = pd.read_csv(data_dir / 'la_county_census_tracts_with_facility_proximity.csv')
    zip_df = pd.read_csv(data_dir / 'la_county_zip_codes_with_facility_proximity.csv')

    return census_df, zip_df


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


def main():
    # Setup
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    output_dir = base_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    census_df, zip_df = load_data()

    print("\nCreating visualizations...")
    create_distance_distribution_plots(census_df, zip_df, output_dir)
    create_facility_count_plots(census_df, zip_df, output_dir)
    create_summary_statistics_table(census_df, zip_df, output_dir)

    print(f"\nVisualizations saved to: {output_dir}/")


if __name__ == '__main__':
    main()
