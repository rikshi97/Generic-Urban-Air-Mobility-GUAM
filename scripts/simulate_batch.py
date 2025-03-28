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
from loguru import logger


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
    jax_use_cpu()
    jax_use_double()
    set_logger_format()

    final_time = 20.0*60.0
    
    #Load waypoints from a GeoJSON file.
    lla_waypoints = get_waypoints_from_geojson("/home/rishi/Berkeley/CITRIS/Generic-Urban-Air-Mobility-GUAM/scripts/UCB_NASA.json")
    
    # lla_waypoints = np.array([
    # [37.42, -122.05, 0],
    # [37.42, -122.05, 500], # NASA Ames
    # [37.87, -122.27, 500],   # UCB
    # [37.87, -122.27, 0]
    # ])

    logger.info("Constructing GUAM...")
    guam = FuncGUAM()
    logger.info("Calling GUAM...")

    batch_size = 4096
    # batch_size = 8192
    # batch_size = 16_384
    state = GuamState.create()
    b_state: GuamState = jtu.tree_map(lambda x: np.broadcast_to(x, (batch_size,) + x.shape).copy(), state)
    T = int(final_time / guam.dt)

    # Perturb the initial state in the x and y directions.
    key0, key1 = jr.split(jr.PRNGKey(0))
    b_state.aircraft[:, 6] = jr.uniform(key0, (batch_size,), minval=-20.0, maxval=20.0)
    b_state.aircraft[:, 7] = jr.uniform(key1, (batch_size,), minval=-20.0, maxval=20.0)

    vmap_step = jax.jit(jax.vmap(ft.partial(guam.step, guam.dt), in_axes=(0, None)))

    def simulate_batch(b_state0) -> GuamState:
        Tb_state = [b_state0]
        b_state = b_state0
        for kk in tqdm.trange(T):
            t = kk * guam.dt
            #ref_inputs = lift_cruise_reference_inputs(t)
            ref_inputs = lift_cruise_reference_inputs_from_lla(T, lla_waypoints, speed=20.0)
            b_state = vmap_step(b_state, ref_inputs)
            Tb_state.append(jax2np(b_state))
        bT_state = jtu.tree_map(lambda *args: np.stack(list(args), axis=1), *Tb_state)
        return bT_state

    bT_state = simulate_batch(b_state)

    np.savez("bT_state.npz", aircraft=bT_state.aircraft)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
