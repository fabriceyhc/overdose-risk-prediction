#!/usr/bin/env python3
"""
Compare filtered facility proximity datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_all_datasets():
    """Load all filtered datasets."""
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    data_dir = base_dir / 'data' / 'processed'

    filters = ['all', 'residential', 'men', 'women', 'youth']
    datasets = {}

    for filter_name in filters:
        census_path = data_dir / f'la_county_census_tracts_facility_proximity_{filter_name}.csv'
        zip_path = data_dir / f'la_county_zip_codes_facility_proximity_{filter_name}.csv'

        datasets[filter_name] = {
            'census': pd.read_csv(census_path),
            'zip': pd.read_csv(zip_path)
        }

    return datasets


def create_comparison_plots(datasets, output_dir):
    """Create comparison visualizations."""

    # Plot 1: Median distance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    filters = ['all', 'residential', 'men', 'women', 'youth']
    labels = ['All\nFacilities', 'Residential\nOnly', 'Men', 'Women', 'Youth']
    colors = ['#2E86AB', '#A23B72', '#6A994E', '#F18F01', '#C73E1D']

    # Census tracts
    census_medians = [datasets[f]['census']['nearest_facility_distance_km'].median() for f in filters]
    bars1 = ax1.bar(labels, census_medians, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Median Distance (km)', fontsize=12)
    ax1.set_title('Median Distance to Nearest Facility\nCensus Tracts', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')

    # ZIP codes
    zip_medians = [datasets[f]['zip']['nearest_facility_distance_km'].median() for f in filters]
    bars2 = ax2.bar(labels, zip_medians, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Median Distance (km)', fontsize=12)
    ax2.set_title('Median Distance to Nearest Facility\nZIP Codes', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_median_distances.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'comparison_median_distances.png'}")
    plt.close()

    # Plot 2: Percentage with zero facilities within 5km
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Census tracts
    census_pct_zero = [(datasets[f]['census']['count_within_5km'] == 0).sum() / len(datasets[f]['census']) * 100
                       for f in filters]
    bars1 = ax1.bar(labels, census_pct_zero, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Areas with ZERO Facilities Within 5km\nCensus Tracts', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # ZIP codes
    zip_pct_zero = [(datasets[f]['zip']['count_within_5km'] == 0).sum() / len(datasets[f]['zip']) * 100
                    for f in filters]
    bars2 = ax2.bar(labels, zip_pct_zero, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Areas with ZERO Facilities Within 5km\nZIP Codes', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_zero_facilities.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'comparison_zero_facilities.png'}")
    plt.close()

    # Plot 3: Distribution comparison (census tracts only)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (filter_name, label, color) in enumerate(zip(filters, labels, colors)):
        df = datasets[filter_name]['census']
        distances = df['nearest_facility_distance_km']

        # Remove outliers for better visualization
        p95 = distances.quantile(0.95)
        distances_plot = distances[distances <= p95]

        axes[idx].hist(distances_plot, bins=50, color=color, edgecolor='black', alpha=0.7)
        axes[idx].axvline(distances.median(), color='red', linestyle='--', linewidth=2,
                         label=f'Median: {distances.median():.2f} km')
        axes[idx].set_xlabel('Distance (km)', fontsize=10)
        axes[idx].set_ylabel('Number of Census Tracts', fontsize=10)
        axes[idx].set_title(f'{label.replace(chr(10), " ")} Facilities', fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[5])

    plt.suptitle('Distance Distribution to Nearest Facility by Filter Type\n(95th percentile truncated for clarity)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'comparison_distributions.png'}")
    plt.close()

    # Plot 4: Mean facilities within distance thresholds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    thresholds = ['1km', '5km', '10km']
    x = np.arange(len(thresholds))
    width = 0.15

    # Census tracts
    for idx, (filter_name, label, color) in enumerate(zip(filters, labels, colors)):
        df = datasets[filter_name]['census']
        means = [df[f'count_within_{t}'].mean() for t in thresholds]
        offset = (idx - 2) * width
        ax1.bar(x + offset, means, width, label=label.replace('\n', ' '), color=color,
               edgecolor='black', alpha=0.8)

    ax1.set_ylabel('Average Number of Facilities', fontsize=12)
    ax1.set_title('Average Facility Count by Distance\nCensus Tracts', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(thresholds)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # ZIP codes
    for idx, (filter_name, label, color) in enumerate(zip(filters, labels, colors)):
        df = datasets[filter_name]['zip']
        means = [df[f'count_within_{t}'].mean() for t in thresholds]
        offset = (idx - 2) * width
        ax2.bar(x + offset, means, width, label=label.replace('\n', ' '), color=color,
               edgecolor='black', alpha=0.8)

    ax2.set_ylabel('Average Number of Facilities', fontsize=12)
    ax2.set_title('Average Facility Count by Distance\nZIP Codes', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(thresholds)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_facility_counts.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'comparison_facility_counts.png'}")
    plt.close()


def print_detailed_comparison(datasets):
    """Print detailed comparison statistics."""

    print("\n" + "="*80)
    print("DETAILED COMPARISON - CENSUS TRACTS")
    print("="*80)

    filters = ['all', 'residential', 'men', 'women', 'youth']
    labels = ['All Facilities', 'Residential Only', 'Men', 'Women', 'Youth']

    print(f"\n{'Filter':<20} {'Median (km)':<12} {'Mean (km)':<12} {'Max (km)':<12} {'0 within 5km':<15}")
    print("-" * 80)

    for filter_name, label in zip(filters, labels):
        df = datasets[filter_name]['census']
        median = df['nearest_facility_distance_km'].median()
        mean = df['nearest_facility_distance_km'].mean()
        max_dist = df['nearest_facility_distance_km'].max()
        zero_count = (df['count_within_5km'] == 0).sum()
        zero_pct = zero_count / len(df) * 100

        print(f"{label:<20} {median:>10.2f}  {mean:>10.2f}  {max_dist:>10.2f}  {zero_count:>6} ({zero_pct:>4.1f}%)")

    print("\n" + "="*80)
    print("DETAILED COMPARISON - ZIP CODES")
    print("="*80)

    print(f"\n{'Filter':<20} {'Median (km)':<12} {'Mean (km)':<12} {'Max (km)':<12} {'0 within 5km':<15}")
    print("-" * 80)

    for filter_name, label in zip(filters, labels):
        df = datasets[filter_name]['zip']
        median = df['nearest_facility_distance_km'].median()
        mean = df['nearest_facility_distance_km'].mean()
        max_dist = df['nearest_facility_distance_km'].max()
        zero_count = (df['count_within_5km'] == 0).sum()
        zero_pct = zero_count / len(df) * 100

        print(f"{label:<20} {median:>10.2f}  {mean:>10.2f}  {max_dist:>10.2f}  {zero_count:>6} ({zero_pct:>4.1f}%)")

    # Gender comparison
    print("\n" + "="*80)
    print("GENDER COMPARISON ANALYSIS")
    print("="*80)

    men_census = datasets['men']['census']
    women_census = datasets['women']['census']

    disparity = women_census['nearest_facility_distance_km'] - men_census['nearest_facility_distance_km']

    print(f"\nMean additional distance for women: {disparity.mean():.3f} km")
    print(f"Median additional distance for women: {disparity.median():.3f} km")
    print(f"Tracts where women travel farther: {(disparity > 0).sum()} ({(disparity > 0).sum()/len(disparity)*100:.1f}%)")
    print(f"Tracts where men travel farther: {(disparity < 0).sum()} ({(disparity < 0).sum()/len(disparity)*100:.1f}%)")
    print(f"Tracts with equal distance: {(disparity == 0).sum()} ({(disparity == 0).sum()/len(disparity)*100:.1f}%)")


def main():
    # Setup
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    output_dir = base_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading filtered datasets...")
    datasets = load_all_datasets()

    print("\nCreating comparison visualizations...")
    create_comparison_plots(datasets, output_dir)

    print_detailed_comparison(datasets)

    print(f"\n{'='*80}")
    print("Comparison analysis complete!")
    print(f"Visualizations saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
