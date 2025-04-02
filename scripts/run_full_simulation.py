import os
import time
import argparse
import numpy as np
import pandas as pd
from loguru import logger
from jax_guam.utils.logging import set_logger_format
from scripts.run_simulation import run_simulation
from scripts.visualize_kepler import create_kepler_visualization
from scripts.visualize_3d_trajectory import create_3d_trajectory_plot
from scripts.simulate_batch import get_waypoints_from_geojson
from jax_guam.utils.coordinate_transforms import ecef_to_lla
from scipy.spatial.distance import cdist
from keplergl import KeplerGl
from scripts.setup_jax import setup_jax

# Configure JAX environment before any JAX operations
devices = setup_jax()

def calculate_trajectory_metrics(
    state_file: str,
    geojson_file: str,
    results_file: str = "trajectory_metrics.csv"
) -> pd.DataFrame:
    """
    Calculate metrics to measure how well the aircraft follows the ideal trajectory.
    
    Parameters:
        state_file: Path to the simulation state file
        geojson_file: Path to the GeoJSON file with waypoints
        results_file: Path to save the metrics CSV
        
    Returns:
        DataFrame with trajectory metrics
    """
    logger.info("Calculating trajectory metrics...")
    
    # Load simulation state
    state_data = np.load(state_file)
    aircraft_state = state_data['aircraft']
    
    # Load waypoints
    lla_waypoints = get_waypoints_from_geojson(geojson_file)
    
    # Convert ECEF positions to LLA
    ecef_positions = aircraft_state[0, :, 6:9]
    lla_positions = np.array([ecef_to_lla(x, y, z) for x, y, z in ecef_positions])
    
    # Calculate metrics for each timestep
    metrics = []
    
    # Get reference waypoints for interpolation
    ref_waypoints = lla_waypoints
    
    # For each position in the trajectory
    for i, (lat, lon, alt) in enumerate(lla_positions):
        # Find the closest waypoint
        distances = cdist(
            np.array([[lat, lon, alt]]),
            ref_waypoints
        )[0]
        
        nearest_idx = np.argmin(distances)
        min_distance = distances[nearest_idx]
        
        # Calculate more advanced metrics
        cross_track_error = min_distance  # Simplified for now
        
        # Append metrics
        metrics.append({
            'timestep': i,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'min_distance_to_waypoint': min_distance,
            'cross_track_error': cross_track_error,
            'nearest_waypoint_idx': nearest_idx
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Calculate summary statistics
    avg_distance = metrics_df['min_distance_to_waypoint'].mean()
    max_distance = metrics_df['min_distance_to_waypoint'].max()
    avg_cross_track = metrics_df['cross_track_error'].mean()
    
    logger.info(f"Average distance to ideal trajectory: {avg_distance:.2f} meters")
    logger.info(f"Maximum distance to ideal trajectory: {max_distance:.2f} meters")
    logger.info(f"Average cross-track error: {avg_cross_track:.2f} meters")
    
    # Save metrics to CSV
    metrics_df.to_csv(results_file, index=False)
    logger.info(f"Metrics saved to {results_file}")
    
    return metrics_df

def visualize_trajectory_accuracy(
    metrics_df: pd.DataFrame,
    geojson_file: str,
    output_file: str = "trajectory_accuracy.html"
) -> None:
    """
    Create a Kepler.gl visualization showing trajectory accuracy.
    
    Parameters:
        metrics_df: DataFrame with trajectory metrics
        geojson_file: Path to the GeoJSON file with waypoints
        output_file: Path to save the HTML visualization
    """
    logger.info("Creating trajectory accuracy visualization...")
    
    # Load waypoints for reference
    lla_waypoints = get_waypoints_from_geojson(geojson_file)
    
    # Create a DataFrame for waypoints
    waypoints_df = pd.DataFrame({
        'latitude': lla_waypoints[:, 0],
        'longitude': lla_waypoints[:, 1],
        'altitude': lla_waypoints[:, 2],
        'type': 'waypoint'
    })
    
    # Create a DataFrame for the trajectory with color based on distance to ideal path
    trajectory_df = pd.DataFrame({
        'latitude': metrics_df['latitude'],
        'longitude': metrics_df['longitude'],
        'altitude': metrics_df['altitude'],
        'error': metrics_df['min_distance_to_waypoint'],
        'type': 'trajectory'
    })
    
    # Create a DataFrame for trajectory segments with next point info
    segments_data = []
    for i in range(len(trajectory_df) - 1):
        segments_data.append({
            'latitude': trajectory_df.iloc[i]['latitude'],
            'longitude': trajectory_df.iloc[i]['longitude'],
            'altitude': trajectory_df.iloc[i]['altitude'],
            'error': trajectory_df.iloc[i]['error'],
            'type': 'trajectory',
            'next_lat': trajectory_df.iloc[i+1]['latitude'],
            'next_lon': trajectory_df.iloc[i+1]['longitude'],
            'next_alt': trajectory_df.iloc[i+1]['altitude']
        })
    
    segments_df = pd.DataFrame(segments_data)
    
    # Create reference line from waypoints
    waypoint_segments = []
    for i in range(len(waypoints_df) - 1):
        waypoint_segments.append({
            'latitude': waypoints_df.iloc[i]['latitude'],
            'longitude': waypoints_df.iloc[i]['longitude'],
            'altitude': waypoints_df.iloc[i]['altitude'],
            'type': 'reference',
            'next_lat': waypoints_df.iloc[i+1]['latitude'],
            'next_lon': waypoints_df.iloc[i+1]['longitude'],
            'next_alt': waypoints_df.iloc[i+1]['altitude']
        })
    
    reference_df = pd.DataFrame(waypoint_segments)
    
    # Create Kepler map with all data
    map_1 = KeplerGl(height=800, data={
        'waypoints': waypoints_df, 
        'trajectory': segments_df,
        'reference': reference_df
    })
    
    # Save to HTML
    map_1.save_to_html(file_name=output_file)
    logger.info(f"Trajectory accuracy visualization saved to {output_file}")

def run_full_simulation(
    geojson_file: str,
    output_dir: str = "results",
    dt: float = 0.005,
    final_time: float = 20.0 * 60.0,
    speed: float = 20.0,
    downsample_factor: int = 5
):
    """
    Run a full simulation, analyze trajectory accuracy, and generate visualizations.
    
    Parameters:
        geojson_file: Path to the GeoJSON file with waypoints
        output_dir: Directory to save results
        dt: Time step for simulation
        final_time: Total simulation time in seconds
        speed: Aircraft speed in m/s
        downsample_factor: Factor to downsample trajectory for visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    set_logger_format()
    
    # Run the simulation
    simulation_time = run_simulation(
        geojson_file=geojson_file,
        output_dir=output_dir,
        dt=dt,
        final_time=final_time,
        speed=speed,
        downsample_factor=downsample_factor
    )
    
    # Calculate trajectory metrics
    state_file = os.path.join(output_dir, "bT_state.npz")
    metrics_file = os.path.join(output_dir, "trajectory_metrics.csv")
    metrics_df = calculate_trajectory_metrics(
        state_file=state_file,
        geojson_file=geojson_file,
        results_file=metrics_file
    )
    
    # Create accuracy visualization
    accuracy_file = os.path.join(output_dir, "trajectory_accuracy.html")
    visualize_trajectory_accuracy(
        metrics_df=metrics_df,
        geojson_file=geojson_file,
        output_file=accuracy_file
    )
    
    logger.info(f"Full simulation completed in {simulation_time:.2f} seconds")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GUAM simulation and analyze trajectory accuracy")
    parser.add_argument("--geojson", default="scripts/UCB_NASA.json", help="Path to GeoJSON file with waypoints")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step for simulation")
    parser.add_argument("--final-time", type=float, default=20.0*60.0, help="Total simulation time in seconds")
    parser.add_argument("--speed", type=float, default=20.0, help="Aircraft speed in m/s")
    parser.add_argument("--downsample-factor", type=int, default=5, help="Factor to downsample trajectory for visualization")
    
    args = parser.parse_args()
    
    run_full_simulation(
        geojson_file=args.geojson,
        output_dir=args.output_dir,
        dt=args.dt,
        final_time=args.final_time,
        speed=args.speed,
        downsample_factor=args.downsample_factor
    ) 