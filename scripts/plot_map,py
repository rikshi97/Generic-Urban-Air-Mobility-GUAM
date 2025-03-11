# Plotting simulation on map

import ipdb
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
from mpl_toolkits.basemap import Basemap

def plot_trajectory_on_map():
    npz = np.load("bT_state.npz")
    bT_state = npz["aircraft"]

    bT_state = bT_state[::8]  # Downsample for plotting
    bT_pos = bT_state[:, :, 6:9]  # Extract position data (ENU coordinates)
    
    lla_waypoints = np.array([
    [37.42, -122.05, 0],
    [37.42, -122.05, 500], # NASA Ames
    [37.87, -122.27, 500],   # UCB
    [37.87, -122.27, 0]
    ])

    # Convert ENU to LLA (Assuming a known reference LLA)
    ref_lat, ref_lon, ref_alt =  lla_waypoints[0]
    lla_waypoints = [enu_to_lla(pos[0], pos[1], pos[2], ref_lat, ref_lon, ref_alt) for pos in bT_pos.reshape(-1, 3)]
    lats, lons, _ = zip(*lla_waypoints)

    # Set up the map
    fig, ax = plt.subplots(figsize=(10, 6))
    m = Basemap(projection='merc', llcrnrlon=min(lons)-0.01, llcrnrlat=min(lats)-0.01,
                urcrnrlon=max(lons)+0.01, urcrnrlat=max(lats)+0.01, resolution='i', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='lightgray', lake_color='aqua')

    # Convert LLA to map coordinates and plot
    x, y = m(lons, lats)
    m.plot(x, y, marker='o', linestyle='-', markersize=2, color='red', alpha=0.7, label='Trajectory')
    plt.legend()
    plt.title("Flight Trajectory on Map")
    plt.savefig("batch_traj_map.pdf")
    plt.show()

def enu_to_lla(east, north, up, ref_lat, ref_lon, ref_alt):
    """Convert ENU coordinates back to LLA using the reference point."""
    lat = ref_lat + (north / 111320)  # Approximate conversion
    lon = ref_lon + (east / (111320 * np.cos(np.radians(ref_lat))))
    alt = ref_alt + up
    return lat, lon, alt

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        plot_trajectory_on_map()