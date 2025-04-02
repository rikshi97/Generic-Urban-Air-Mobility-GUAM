import functools as ft

import ipdb
import jax
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import tqdm
import json
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import lift_cruise_reference_inputs_from_lla, lift_cruise_reference_inputs
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format
from jax_guam.functional.vehicle_eom_utils import _ECEF_noacc
from loguru import logger
import jax.numpy as jnp

def setup_gpu():
    """Setup GPU for JAX computation with multiple GPUs support."""
    try:
        # Get available devices
        devices = jax.devices()
        if len(devices) > 0:
            logger.info(f"Found {len(devices)} GPU devices:")
            for i, device in enumerate(devices):
                logger.info(f"GPU {i}: {device}")
            
            # Set up device placement
            jax.config.update('jax_default_device', devices[0])
            return len(devices)
        else:
            logger.warning("No GPU devices found, falling back to CPU")
            return 0
    except Exception as e:
        logger.warning(f"Error setting up GPU: {e}, falling back to CPU")
        return 0

def get_waypoints_from_geojson(geojson_file):
    """
    Extracts latitude, longitude, and optionally altitude waypoints from a GeoJSON file.
    
    Parameters:
        geojson_file (str): Path to the GeoJSON file.
        
    Returns:
        np.ndarray: An array of waypoints in the format [[lat, lon, alt], ...].
                    If altitude is not available, defaults to 0.
    """
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)
    
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
    
    return np.array(waypoints)

def main():
    # Setup GPU if available
    num_gpus = setup_gpu()
    if num_gpus == 0:
        jax_use_cpu()
    
    jax_use_double()
    set_logger_format()

    final_time = 20.0*60.0
    
    # Load waypoints from a GeoJSON file.
    lla_waypoints = get_waypoints_from_geojson("/home/rishi/Berkeley/CITRIS/Generic-Urban-Air-Mobility-GUAM/scripts/UCB_NASA.json")
    
    logger.info("Constructing GUAM...")
    guam = FuncGUAM()
    logger.info("Calling GUAM...")

    # Adjust batch size based on number of GPUs
    base_batch_size = 16
    batch_size = base_batch_size * max(1, num_gpus)
    logger.info(f"Using batch size of {batch_size} across {num_gpus} GPUs")
    
    state = GuamState.create()
    b_state: GuamState = jtu.tree_map(lambda x: np.broadcast_to(x, (batch_size,) + x.shape).copy(), state)
    T = int(final_time / guam.dt)

    # Convert first waypoint to ENU coordinates for initial state
    ref_lat, ref_lon, ref_alt = lla_waypoints[0]
    ref_pos_bii = jnp.array([[ref_lon], [ref_lat], [ref_alt]])
    vel_bIi = jnp.zeros((3, 1))
    Omega_BIb = jnp.zeros((3, 1))
    Q_i2b = jnp.array([[1.0], [0.0], [0.0], [0.0]])
    Omeg_EIi = jnp.zeros((3, 1))
    
    # Calculate reference ECEF position
    ref_Pos_bei, _, _ = _ECEF_noacc(ref_pos_bii, vel_bIi, Omega_BIb, Q_i2b, Omeg_EIi)
    
    # Reshape and broadcast reference position to match batch size
    ref_Pos_bei = jnp.broadcast_to(ref_Pos_bei.T, (batch_size, 3))
    
    # Set initial position to first waypoint
    aircraft_state = jnp.array(b_state.aircraft)
    aircraft_state = aircraft_state.at[:, 6:9].set(ref_Pos_bei)
    b_state = b_state._replace(aircraft=aircraft_state)

    # JIT compile the step function for better performance
    vmap_step = jax.jit(jax.vmap(ft.partial(guam.step, guam.dt), in_axes=(0, None)))

    def simulate_batch(b_state0) -> GuamState:
        Tb_state = [b_state0]
        b_state = b_state0
        
        # Use tqdm for progress bar
        for kk in tqdm.trange(T):
            t = kk * guam.dt
            ref_inputs = lift_cruise_reference_inputs_from_lla(T, lla_waypoints, speed=20.0)
            b_state = vmap_step(b_state, ref_inputs)
            Tb_state.append(jax2np(b_state))
        
        bT_state = jtu.tree_map(lambda *args: np.stack(list(args), axis=1), *Tb_state)
        return bT_state

    # Run simulation
    bT_state = simulate_batch(b_state)

    # Save results
    np.savez("bT_state.npz", aircraft=bT_state.aircraft)
    logger.info("Simulation completed and results saved")

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
