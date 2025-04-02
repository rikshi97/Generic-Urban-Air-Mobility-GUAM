import os
import time
import functools as ft

import jax
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import tqdm
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import lift_cruise_reference_inputs_from_lla
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format
from jax_guam.functional.vehicle_eom_utils import _ECEF_noacc
from loguru import logger
import jax.numpy as jnp

from scripts.visualize_3d_trajectory import create_3d_trajectory_plot
from scripts.visualize_kepler import create_kepler_visualization
from scripts.simulate_batch import get_waypoints_from_geojson, setup_gpu
from jax_guam.utils.coordinate_transforms import lla_to_ecef, ecef_to_lla

def run_simulation(
    geojson_file: str,
    output_dir: str = "results",
    dt: float = 0.005,
    final_time: float = 20.0 * 60.0,
    speed: float = 20.0,
    downsample_factor: int = 5
):
    """
    Run a simulation with the given parameters.
    
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
    
    # Load waypoints from GeoJSON
    lla_waypoints = get_waypoints_from_geojson(geojson_file)
    logger.info(f"Loaded {len(lla_waypoints)} waypoints from {geojson_file}")
    
    # Get first waypoint
    ref_lat, ref_lon, ref_alt = lla_waypoints[0]
    logger.info(f"First waypoint LLA coordinates: lat={ref_lat:.6f}, lon={ref_lon:.6f}, alt={ref_alt:.2f}m")
    
    # Convert first waypoint from LLA to ECEF for initial position
    ref_ecef_x, ref_ecef_y, ref_ecef_z = lla_to_ecef(ref_lat, ref_lon, ref_alt)
    logger.info(f"First waypoint ECEF position: ({ref_ecef_x}, {ref_ecef_y}, {ref_ecef_z})")
    
    # Create initial state
    initial_state = jnp.zeros(13)  # 13 state variables
    initial_state = initial_state.at[6:9].set(jnp.array([ref_ecef_x, ref_ecef_y, ref_ecef_z]))  # Set initial position
    initial_state = initial_state.at[9:13].set(jnp.array([1.0, 0.0, 0.0, 0.0]))  # Set initial quaternion to identity
    
    # Verify initial position after setting
    initial_pos = initial_state[6:9]
    logger.info(f"Initial aircraft ECEF position: {initial_pos}")
    
    # Convert back to LLA to verify
    initial_lat, initial_lon, initial_alt = ecef_to_lla(initial_pos[0], initial_pos[1], initial_pos[2])
    logger.info(f"Initial aircraft LLA position: lat={initial_lat:.6f}, lon={initial_lon:.6f}, alt={initial_alt:.2f}m")
    
    # Verify these match the first waypoint
    lat_diff = abs(initial_lat - ref_lat)
    lon_diff = abs(initial_lon - ref_lon)
    alt_diff = abs(initial_alt - ref_alt)
    logger.info(f"Position differences from first waypoint:")
    logger.info(f"  Latitude difference: {lat_diff:.6f} degrees")
    logger.info(f"  Longitude difference: {lon_diff:.6f} degrees")
    logger.info(f"  Altitude difference: {alt_diff:.2f} meters")
    
    # Create simulation function
    def simulate_step(state, t):
        # Get current position in ECEF
        pos_bii = state[6:9]  # Position vector in inertial frame
        vel_bIi = state[3:6]  # Velocity vector in inertial frame
        Omega_BIb = state[0:3]  # Angular velocity vector
        Q_i2b = state[9:13]  # Quaternion from inertial to body frame
        
        # Convert current position to LLA for reference input calculation
        current_lat, current_lon, current_alt = ecef_to_lla(pos_bii[0], pos_bii[1], pos_bii[2])
        
        # Get reference inputs based on current position
        ref_inputs = lift_cruise_reference_inputs_from_lla(t, lla_waypoints, speed=speed)
        
        # Earth rotation rate in rad/s
        Omeg_EIi = jnp.array([0.0, 0.0, 7.2921150e-5])
        
        # Update state using ECEF transformation
        Pos_bei, Vel_bEi, Omeg_BEb = _ECEF_noacc(pos_bii, vel_bIi, Omega_BIb, Q_i2b, Omeg_EIi)
        
        # Simple Euler integration for position and velocity
        new_pos = pos_bii + Vel_bEi * dt
        new_vel = vel_bIi + ref_inputs[:3] * dt  # Assuming first 3 elements are accelerations
        
        # Update quaternion using angular velocity
        # q_dot = 0.5 * q * omega (quaternion multiplication)
        omega_quat = jnp.array([0, Omeg_BEb[0], Omeg_BEb[1], Omeg_BEb[2]])
        q_dot = 0.5 * quaternion_multiplication(Q_i2b, omega_quat)
        new_quat = Q_i2b + q_dot * dt
        
        # Normalize quaternion to prevent numerical drift
        new_quat = new_quat / jnp.linalg.norm(new_quat)
        
        # Construct new state vector
        # [Omega_BIb (3), vel_bIi (3), pos_bii (3), Q_i2b (4)]
        new_state = jnp.concatenate([
            Omeg_BEb,    # Angular velocity in body frame
            new_vel,     # Updated velocity
            new_pos,     # Updated position
            new_quat     # Updated quaternion
        ])
        
        return new_state, (new_state, ref_inputs)
    
    # Run simulation
    logger.info("\nStarting simulation...")
    times = jnp.arange(0, final_time, dt)
    final_state, (states, ref_inputs) = jax.lax.scan(simulate_step, initial_state, times)
    
    # Log final position
    final_pos = final_state[6:9]
    final_lat, final_lon, final_alt = ecef_to_lla(final_pos[0], final_pos[1], final_pos[2])
    logger.info(f"\nFinal position:")
    logger.info(f"ECEF: {final_pos}")
    logger.info(f"LLA: lat={final_lat:.6f}, lon={final_lon:.6f}, alt={final_alt:.2f}m")
    
    # Save results
    save_simulation_results(
        states=states,
        ref_inputs=ref_inputs,
        times=times,
        output_dir=output_dir
    )
    
    return final_time

if __name__ == "__main__":
    # Define file paths
    geojson_file = "scripts/UCB_NASA.json"
    output_dir = "results"
    
    # Run simulation
    run_simulation(geojson_file, output_dir) 