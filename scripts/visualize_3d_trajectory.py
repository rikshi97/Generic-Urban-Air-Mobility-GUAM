import numpy as np
import plotly.graph_objects as go
from loguru import logger
from jax_guam.utils.coordinate_transforms import lla_to_enu, batch_lla_to_enu, validate_coordinates, ecef_to_lla
from scripts.simulate_batch import get_waypoints_from_geojson, setup_gpu

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def create_3d_trajectory_plot(
    state_file: str,
    geojson_file: str,
    output_file: str = "trajectory_3d.html",
    downsample_factor: int = 5
) -> None:
    """Create an interactive 3D visualization of the flight trajectory."""
    try:
        # Setup GPU if available
        num_gpus = setup_gpu()
        
        # Load simulation state
        state_data = np.load(state_file)
        aircraft_state = state_data['aircraft']  # Shape: (1, T, 13)
        
        # Load waypoints
        lla_waypoints = get_waypoints_from_geojson(geojson_file)
        
        # Validate waypoints
        for lat, lon, alt in lla_waypoints:
            if not validate_coordinates(lat, lon, alt):
                logger.warning(f"Invalid coordinates detected: lat={lat}, lon={lon}, alt={alt}")
        
        # Get reference point for coordinate conversion
        ref_lat, ref_lon, ref_alt = lla_waypoints[0]
        
        # Convert waypoints to ENU coordinates
        lats, lons, alts = lla_waypoints.T
        enu_waypoints = batch_lla_to_enu(lats, lons, alts, ref_lat, ref_lon, ref_alt)
        
        # Extract position and orientation from aircraft state
        ecef_positions = aircraft_state[0, ::downsample_factor, 6:9]  # Shape: (T/downsample, 3)
        logger.info(f"Number of trajectory points: {len(ecef_positions)}")
        logger.info(f"ECEF positions shape: {ecef_positions.shape}")
        logger.info(f"First few ECEF positions:\n{ecef_positions[:5]}")
        
        # Convert ECEF coordinates to LLA then to ENU for visualization
        lla_positions = []
        for x, y, z in ecef_positions:
            # Use the proper ECEF to LLA conversion
            lat, lon, alt = ecef_to_lla(x, y, z)
            lla_positions.append([lat, lon, alt])
        
        lla_positions = np.array(lla_positions)
        
        # Convert LLA positions to ENU for 3D visualization
        positions = []
        for lat, lon, alt in lla_positions:
            east, north, up = lla_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
            positions.append([east, north, up])
        
        positions = np.array(positions)
        logger.info(f"ENU positions shape: {positions.shape}")
        logger.info(f"First few ENU positions:\n{positions[:5]}")
        
        # Extract quaternions and normalize them
        quaternions = aircraft_state[0, ::downsample_factor, 3:7]  # Shape: (T/downsample, 4)
        logger.info(f"Quaternions shape: {quaternions.shape}")
        logger.info(f"First few quaternions:\n{quaternions[:5]}")
        
        # Normalize quaternions
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        
        # Create figure
        fig = go.Figure()
        
        # Add waypoints
        fig.add_trace(go.Scatter3d(
            x=enu_waypoints[:, 0],
            y=enu_waypoints[:, 1],
            z=enu_waypoints[:, 2],
            mode='markers+lines',
            name='Waypoints',
            marker=dict(size=8, color='red'),
            line=dict(color='red', width=2)
        ))
        
        # Add trajectory with color gradient based on altitude
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            name='Trajectory',
            line=dict(
                color=positions[:, 2],  # Color based on altitude
                colorscale='Viridis',
                width=4
            ),
            marker=dict(
                size=4,
                color=positions[:, 2],
                colorscale='Viridis',
                opacity=0.5
            )
        ))
        
        # Add aircraft orientation indicators (every 20th point)
        for i in range(0, len(positions), 20):
            pos = positions[i]
            q = quaternions[i]
            
            # Convert quaternion to rotation matrix
            rotation_matrix = quaternion_to_rotation_matrix(q)
            
            # Add orientation arrows
            arrow_length = 5.0  # meters
            for axis, color in zip(range(3), ['red', 'green', 'blue']):
                direction = rotation_matrix[:, axis] * arrow_length
                fig.add_trace(go.Scatter3d(
                    x=[pos[0], pos[0] + direction[0]],
                    y=[pos[1], pos[1] + direction[1]],
                    z=[pos[2], pos[2] + direction[2]],
                    mode='lines',
                    name=f'Axis {axis}',
                    line=dict(color=color, width=2),
                    showlegend=False
                ))
        
        # Calculate the center and scale of the visualization
        center_x = np.mean(positions[:, 0])
        center_y = np.mean(positions[:, 1])
        center_z = np.mean(positions[:, 2])
        max_range = np.max([
            np.ptp(positions[:, 0]),
            np.ptp(positions[:, 1]),
            np.ptp(positions[:, 2])
        ])
        
        # Update layout
        fig.update_layout(
            title='3D Flight Trajectory and Aircraft Orientation',
            scene=dict(
                xaxis_title='East (m)',
                yaxis_title='North (m)',
                zaxis_title='Up (m)',
                aspectmode='data',
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255,255,255,0.2)',
                    range=[center_x - max_range/2, center_x + max_range/2]
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255,255,255,0.2)',
                    range=[center_y - max_range/2, center_y + max_range/2]
                ),
                zaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255,255,255,0.2)',
                    range=[center_z - max_range/2, center_z + max_range/2]
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=center_x, y=center_y, z=center_z),
                eye=dict(
                    x=center_x + max_range/2,
                    y=center_y + max_range/2,
                    z=center_z + max_range/2
                )
            )
        )
        
        # Save to HTML
        fig.write_html(output_file)
        logger.info(f"3D trajectory visualization saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating 3D trajectory visualization: {e}")
        raise

if __name__ == "__main__":
    # Define file paths
    state_file = "bT_state.npz"
    geojson_file = "scripts/UCB_NASA.json"
    
    # Create visualization
    create_3d_trajectory_plot(state_file, geojson_file) 