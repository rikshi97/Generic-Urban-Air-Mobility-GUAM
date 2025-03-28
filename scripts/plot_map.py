# Plotting simulation on map

import ipdb
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import json
from shapely.geometry import LineString
from geopy.distance import geodesic
from mpl_toolkits.basemap import Basemap
from loguru import logger
import os
from typing import List, Tuple, Optional
import warnings


def get_waypoints_from_geojson(geojson_file: str) -> np.ndarray:
    """
    Extracts latitude, longitude, and optionally altitude waypoints from a GeoJSON file.
    
    Parameters:
        geojson_file (str): Path to the GeoJSON file.
        
    Returns:
        np.ndarray: An array of waypoints in the format [[lat, lon, alt], ...].
                    If altitude is not available, defaults to 0.
                    
    Raises:
        FileNotFoundError: If the GeoJSON file doesn't exist
        json.JSONDecodeError: If the GeoJSON file is invalid
        ValueError: If no valid waypoints are found
    """
    try:
        with open(geojson_file, 'r') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"GeoJSON file not found: {geojson_file}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid GeoJSON file: {geojson_file}")
        raise
    
    waypoints = []
    
    for feature in geojson_data.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") == "Point":
            coord = geometry.get("coordinates", [])
            if len(coord) >= 2:
                lat, lon = coord[1], coord[0]
                alt = coord[2] if len(coord) > 2 else 0
                waypoints.append([lat, lon, alt])
        elif geometry.get("type") == "LineString":
            for coord in geometry.get("coordinates", []):
                if len(coord) >= 2:
                    lat, lon = coord[1], coord[0]
                    alt = coord[2] if len(coord) > 2 else 0
                    waypoints.append([lat, lon, alt])
    
    if not waypoints:
        raise ValueError("No valid waypoints found in GeoJSON file")
    
    return np.array(waypoints)


def plot_trajectory_on_map(state_file: str, geojson_file: str, 
                         output_file: str = "batch_traj_map.pdf",
                         downsample_factor: int = 8,
                         show_altitude: bool = True) -> None:
    """
    Plot the flight trajectory on a map with waypoints and altitude information.
    
    Parameters:
        state_file (str): Path to the simulation state file (.npz)
        geojson_file (str): Path to the GeoJSON file containing waypoints
        output_file (str): Path to save the output plot
        downsample_factor (int): Factor to downsample the trajectory for plotting
        show_altitude (bool): Whether to show altitude information
    """
    try:
        # Load simulation state
        npz = np.load(state_file)
        bT_state = npz["aircraft"]
        
        # Downsample and extract position data
        bT_state = bT_state[::downsample_factor]
        bT_pos = bT_state[:, :, 6:9]
        bT_pos = bT_pos[~np.isnan(bT_pos).any(axis=(1,2))]
        
        # Load waypoints
        lla_waypoints_input = get_waypoints_from_geojson(geojson_file)
        
        # Convert ENU to LLA
        ref_lat, ref_lon, ref_alt = lla_waypoints_input[0]
        lla_waypoints = [enu_to_lla(pos[0], pos[1], pos[2], ref_lat, ref_lon, ref_alt) 
                        for pos in bT_pos.reshape(-1, 3)]
        lats, lons, alts = zip(*lla_waypoints)
        
        # Calculate map bounds with padding
        padding = 0.01
        llcrnrlon = min(lons) - padding
        llcrnrlat = min(lats) - padding
        urcrnrlon = max(lons) + padding
        urcrnrlat = max(lats) + padding
        
        # Create figure with two subplots if showing altitude
        if show_altitude:
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax_map = fig.add_subplot(gs[0])
            ax_alt = fig.add_subplot(gs[1])
        else:
            fig, ax_map = plt.subplots(figsize=(10, 6))
        
        # Set up the map
        m = Basemap(projection='merc', 
                   llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                   urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                   resolution='i', ax=ax_map)
        
        # Draw map features
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        m.drawmapboundary(fill_color='aqua')
        m.fillcontinents(color='lightgray', lake_color='aqua')
        
        # Plot trajectory
        x, y = m(lons, lats)
        scatter = m.scatter(x, y, c=alts, cmap='viridis', 
                          s=2, alpha=0.7, label='Trajectory')
        
        # Add colorbar for altitude
        if show_altitude:
            plt.colorbar(scatter, ax=ax_map, label='Altitude (m)')
        
        # Plot waypoints
        plot_waypoints(m, lla_waypoints_input)
        
        # Add title and legend
        ax_map.set_title("Flight Trajectory on Map")
        ax_map.legend()
        
        # Plot altitude profile if requested
        if show_altitude:
            times = np.arange(len(alts)) * 0.005 * 10  # Assuming same dt as simulation
            ax_alt.plot(times, alts, 'b-', alpha=0.7)
            ax_alt.set_xlabel('Time (s)')
            ax_alt.set_ylabel('Altitude (m)')
            ax_alt.set_title('Altitude Profile')
            ax_alt.grid(True)
        
        # Save and show plot
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting trajectory: {str(e)}")
        raise

def plot_waypoints(m: Basemap, waypoints: np.ndarray) -> None:
    """
    Plot waypoints as a dotted line with markers on the map.
    
    Parameters:
        m (Basemap): Basemap instance
        waypoints (np.ndarray): Array of waypoints in [lat, lon, alt] format
    """
    w_lats, w_lons, _ = zip(*waypoints)
    wx, wy = m(w_lons, w_lats)
    m.plot(wx, wy, marker='x', linestyle='--', color='blue', 
           markersize=5, label='Waypoints', linewidth=2)

def enu_to_lla(east: float, north: float, up: float, 
                ref_lat: float, ref_lon: float, ref_alt: float) -> Tuple[float, float, float]:
    """
    Convert ENU (East-North-Up) coordinates to LLA (Latitude-Longitude-Altitude).
    Uses geodesic calculations for more accurate conversion.
    
    Parameters:
        east (float): East coordinate in meters
        north (float): North coordinate in meters
        up (float): Up coordinate in meters
        ref_lat (float): Reference latitude in degrees
        ref_lon (float): Reference longitude in degrees
        ref_alt (float): Reference altitude in meters
        
    Returns:
        Tuple[float, float, float]: (latitude, longitude, altitude)
    """
    # Calculate the bearing and distance
    distance = np.sqrt(east**2 + north**2)
    bearing = np.degrees(np.arctan2(east, north))
    
    # Use geodesic to calculate new position
    geod = geodesic(meters=distance)
    new_lat, new_lon = geod.destination((ref_lat, ref_lon), bearing)
    
    # Add altitude
    new_alt = ref_alt + up
    
    return new_lat, new_lon, new_alt


if __name__ == "__main__":
    # Suppress deprecation warnings from basemap
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Define file paths
    state_file = "bT_state.npz"
    geojson_file = "UCB_NASA.json"
    
    # Plot trajectory
    plot_trajectory_on_map(state_file, geojson_file, show_altitude=True)