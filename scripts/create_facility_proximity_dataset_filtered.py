#!/usr/bin/env python3
"""
Create filtered geospatial datasets showing relationships between SUD treatment
facilities and geographic units with various filters:

1. All facilities (baseline)
2. Residential only (RES, RES-DETOX - facilities with beds)
3. Gender-specific: Men, Women, Co-ed
4. Youth/adolescent facilities

For each filter, compute:
1. Distance to nearest facility
2. Count of facilities within various distance thresholds
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from geopy.distance import geodesic
from shapely.geometry import shape, Point
from typing import Dict, List, Tuple
import geopandas as gpd
from tqdm import tqdm


def load_facilities(facilities_path: str, filter_type: str = 'all') -> gpd.GeoDataFrame:
    """
    Load SUD facilities with optional filtering.

    Filter types:
    - 'all': All facilities (default)
    - 'residential': Only residential facilities (RES, RES-DETOX, DPH-DETOX)
    - 'men': Facilities serving men (CO-ED or MEN ONLY)
    - 'women': Facilities serving women (CO-ED or WOMEN ONLY/WOMEN CHILDREN)
    - 'youth': Youth/adolescent facilities
    """
    print(f"Loading SUD facilities (filter: {filter_type})...")

    with open(facilities_path, 'r') as f:
        facilities_data = json.load(f)

    # Extract facilities
    facilities = []
    for feature in facilities_data['features']:
        props = feature['properties']
        coords = feature['geometry']['coordinates']

        facilities.append({
            'OBJECTID': props.get('OBJECTID'),
            'Facility_Name': props.get('Facility_Name'),
            'CountyName': props.get('CountyName'),
            'Latitude': props.get('Latitude'),
            'Longitude': props.get('Longitude'),
            'lon': coords[0],
            'lat': coords[1],
            'Facility_Address1': props.get('Facility_Address1'),
            'Facility_City': props.get('Facility_City'),
            'Facility_Zip': props.get('Facility_Zip'),
            'Treatment_Capacity': props.get('Treatment_Capacity'),
            'Total_Capacity': props.get('Total_Capacity'),
            'Program_Code': props.get('Program_Code'),
            'Target_Population': props.get('Target_Population')
        })

    df = pd.DataFrame(facilities)

    # Filter to LA County
    df = df[df['CountyName'] == 'Los Angeles County'].copy()
    print(f"LA County facilities before filtering: {len(df)}")

    # Apply filters
    if filter_type == 'residential':
        # Only facilities with residential beds
        residential_codes = ['RES', 'RES-DETOX', 'DPH-DETOX', 'DSS']
        df = df[df['Program_Code'].isin(residential_codes)].copy()

    elif filter_type == 'men':
        # Facilities that serve men (CO-ED or MEN ONLY)
        df = df[df['Target_Population'].str.contains('CO-ED|MEN ONLY|MEN/YOUTH|DUAL DIAGNOSIS',
                                                     case=False, na=False)].copy()

    elif filter_type == 'women':
        # Facilities that serve women (CO-ED, WOMEN ONLY, WOMEN/CHILDREN, etc.)
        df = df[df['Target_Population'].str.contains('CO-ED|WOMEN|DUAL DIAGNOSIS',
                                                     case=False, na=False)].copy()

    elif filter_type == 'youth':
        # Youth/adolescent facilities
        df = df[df['Target_Population'].str.contains('YOUTH|ADOLESCENT',
                                                     case=False, na=False)].copy()

    print(f"Facilities after filtering: {len(df)}")

    if len(df) == 0:
        raise ValueError(f"No facilities found for filter type: {filter_type}")

    # Create GeoDataFrame
    geometry = [Point(row['lon'], row['lat']) for _, row in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    return gdf


def load_geographic_units(geo_path: str, unit_type: str) -> gpd.GeoDataFrame:
    """Load geographic units (census tracts or ZIP codes)."""
    print(f"Loading {unit_type}...")

    gdf = gpd.read_file(geo_path)

    # Calculate centroids in projected CRS
    print(f"Calculating centroids for {len(gdf)} {unit_type}...")
    gdf_projected = gdf.to_crs('EPSG:2229')
    centroids_projected = gdf_projected.geometry.centroid
    centroids_wgs84 = centroids_projected.to_crs('EPSG:4326')

    gdf['centroid_lat'] = centroids_wgs84.y
    gdf['centroid_lon'] = centroids_wgs84.x

    return gdf


def compute_distances_vectorized(geo_units_gdf: gpd.GeoDataFrame,
                                  facilities_gdf: gpd.GeoDataFrame) -> Tuple:
    """Compute distances from all geographic units to all facilities using vectorized operations."""
    print("Computing distance matrix...")

    # Prepare coordinate arrays
    geo_coords = np.array(list(zip(geo_units_gdf['centroid_lat'], geo_units_gdf['centroid_lon'])))
    facility_coords = np.array(list(zip(facilities_gdf['lat'], facilities_gdf['lon'])))

    # Convert to radians for Haversine
    geo_coords_rad = np.radians(geo_coords)
    facility_coords_rad = np.radians(facility_coords)

    # Haversine distance calculation (vectorized)
    lat1 = geo_coords_rad[:, 0:1]
    lon1 = geo_coords_rad[:, 1:2]
    lat2 = facility_coords_rad[:, 0:1].T
    lon2 = facility_coords_rad[:, 1:2].T

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius
    R_km = 6371.0
    R_miles = 3958.8

    distances_km = R_km * c
    distances_miles = R_miles * c

    return distances_km, distances_miles


def compute_proximity_metrics(geo_gdf: gpd.GeoDataFrame,
                              facilities_gdf: gpd.GeoDataFrame,
                              distance_thresholds_km: List[float] = [1, 5, 10, 15, 25],
                              distance_thresholds_miles: List[float] = [1, 3, 5, 10, 15]) -> gpd.GeoDataFrame:
    """Compute proximity metrics for each geographic unit."""
    # Compute all distances at once (vectorized)
    distances_km, distances_miles = compute_distances_vectorized(geo_gdf, facilities_gdf)

    print("Computing proximity metrics...")

    # Find nearest facility for each geo unit
    nearest_indices = np.argmin(distances_km, axis=1)
    nearest_distances_km = np.min(distances_km, axis=1)
    nearest_distances_miles = np.min(distances_miles, axis=1)

    # Get facility info for nearest facilities
    facility_ids = facilities_gdf['OBJECTID'].values
    facility_names = facilities_gdf['Facility_Name'].values

    output_gdf = geo_gdf.copy()
    output_gdf['nearest_facility_id'] = facility_ids[nearest_indices]
    output_gdf['nearest_facility_name'] = facility_names[nearest_indices]
    output_gdf['nearest_facility_distance_km'] = np.round(nearest_distances_km, 3)
    output_gdf['nearest_facility_distance_miles'] = np.round(nearest_distances_miles, 3)

    # Count facilities within distance thresholds
    print("Counting facilities within distance thresholds...")
    for threshold_km in tqdm(distance_thresholds_km, desc="Processing km thresholds"):
        counts = np.sum(distances_km <= threshold_km, axis=1)
        output_gdf[f'count_within_{int(threshold_km)}km'] = counts

    for threshold_mi in tqdm(distance_thresholds_miles, desc="Processing mile thresholds"):
        counts = np.sum(distances_miles <= threshold_mi, axis=1)
        output_gdf[f'count_within_{int(threshold_mi)}mi'] = counts

    return output_gdf


def save_outputs(gdf: gpd.GeoDataFrame, output_prefix: str, output_dir: Path):
    """Save outputs in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as GeoJSON
    geojson_path = output_dir / f"{output_prefix}.geojson"
    print(f"Saving GeoJSON to {geojson_path}...")
    gdf.to_file(geojson_path, driver='GeoJSON')

    # Save as CSV
    csv_path = output_dir / f"{output_prefix}.csv"
    print(f"Saving CSV to {csv_path}...")
    csv_df = gdf.drop(columns=['geometry']).copy()
    csv_df.to_csv(csv_path, index=False)

    print(f"Saved {len(gdf)} records")


def print_summary(gdf: gpd.GeoDataFrame, geo_type: str, filter_name: str):
    """Print summary statistics."""
    print(f"\n{geo_type} - {filter_name}:")
    print(f"  Total units: {len(gdf)}")
    print(f"  Mean distance to nearest: {gdf['nearest_facility_distance_km'].mean():.2f} km ({gdf['nearest_facility_distance_miles'].mean():.2f} miles)")
    print(f"  Median distance to nearest: {gdf['nearest_facility_distance_km'].median():.2f} km ({gdf['nearest_facility_distance_miles'].median():.2f} miles)")
    print(f"  Max distance to nearest: {gdf['nearest_facility_distance_km'].max():.2f} km ({gdf['nearest_facility_distance_miles'].max():.2f} miles)")
    print(f"  Mean facilities within 5km: {gdf['count_within_5km'].mean():.1f}")
    print(f"  Units with 0 facilities within 5km: {(gdf['count_within_5km'] == 0).sum()} ({(gdf['count_within_5km'] == 0).sum()/len(gdf)*100:.1f}%)")


def main():
    # Paths
    base_dir = Path('/data2/fabricehc/overdose-risk-prediction')
    data_dir = base_dir / 'data'

    facilities_path = data_dir / 'facilities' / 'SUD_Recovery_Treatment_Facilities_2025.11.25.geojson'
    census_tracts_path = data_dir / 'geo' / '2020_Census_Tracts_2025.11.25.geojson'
    zip_codes_path = data_dir / 'geo' / 'LA_County_ZIP_Codes_2025.11.25.geojson'

    output_dir = data_dir / 'processed'

    # Define filters to run
    filters = {
        'all': 'All Facilities',
        'residential': 'Residential Facilities Only',
        'men': 'Facilities Serving Men',
        'women': 'Facilities Serving Women',
        'youth': 'Youth/Adolescent Facilities'
    }

    # Load geographic units once
    census_gdf = load_geographic_units(str(census_tracts_path), "census tracts")
    zip_gdf = load_geographic_units(str(zip_codes_path), "ZIP codes")

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Process each filter
    for filter_key, filter_name in filters.items():
        print("\n" + "="*80)
        print(f"PROCESSING: {filter_name.upper()}")
        print("="*80)

        try:
            # Load facilities with filter
            facilities_gdf = load_facilities(str(facilities_path), filter_type=filter_key)

            # Process Census Tracts
            census_with_metrics = compute_proximity_metrics(
                census_gdf,
                facilities_gdf,
                distance_thresholds_km=[1, 5, 10, 15, 25],
                distance_thresholds_miles=[1, 3, 5, 10, 15]
            )

            save_outputs(
                census_with_metrics,
                f'la_county_census_tracts_facility_proximity_{filter_key}',
                output_dir
            )

            print_summary(census_with_metrics, "Census Tracts", filter_name)

            # Process ZIP Codes
            zip_with_metrics = compute_proximity_metrics(
                zip_gdf,
                facilities_gdf,
                distance_thresholds_km=[1, 5, 10, 15, 25],
                distance_thresholds_miles=[1, 3, 5, 10, 15]
            )

            save_outputs(
                zip_with_metrics,
                f'la_county_zip_codes_facility_proximity_{filter_key}',
                output_dir
            )

            print_summary(zip_with_metrics, "ZIP Codes", filter_name)

        except ValueError as e:
            print(f"Skipping {filter_name}: {e}")

    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
