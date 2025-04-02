import numpy as np
import pandas as pd
from keplergl import KeplerGl
from loguru import logger
from jax_guam.utils.coordinate_transforms import lla_to_enu, batch_lla_to_enu, validate_coordinates, ecef_to_lla
from scripts.simulate_batch import get_waypoints_from_geojson, setup_gpu

def create_kepler_visualization(
    state_file: str,
    geojson_file: str,
    output_file: str = "kepler_trajectory.html",
    downsample_factor: int = 5
) -> None:
    """Create an interactive Kepler.gl visualization of the flight trajectory."""
    try:
        # Setup GPU if available
        num_gpus = setup_gpu()
        
        # Load simulation state
        state_data = np.load(state_file)
        aircraft_state = state_data['aircraft']
        
        # Load waypoints
        lla_waypoints = get_waypoints_from_geojson(geojson_file)
        
        # Validate waypoints
        for lat, lon, alt in lla_waypoints:
            if not validate_coordinates(lat, lon, alt):
                logger.warning(f"Invalid coordinates detected: lat={lat}, lon={lon}, alt={alt}")
        
        # Get reference point for coordinate conversion
        ref_lat, ref_lon, ref_alt = lla_waypoints[0]
        
        # Extract position from aircraft state
        # Convert ECEF positions to LLA
        ecef_positions = aircraft_state[0, ::downsample_factor, 6:9]
        logger.info(f"Number of trajectory points: {len(ecef_positions)}")
        logger.info(f"ECEF positions shape: {ecef_positions.shape}")
        logger.info(f"First few ECEF positions:\n{ecef_positions[:5]}")
        
        # Convert ECEF coordinates to LLA properly
        lla_positions = []
        for x, y, z in ecef_positions:
            # Use the proper ECEF to LLA conversion
            lat, lon, alt = ecef_to_lla(x, y, z)
            lla_positions.append([lat, lon, alt])
        
        lla_positions = np.array(lla_positions)
        logger.info(f"LLA positions shape: {lla_positions.shape}")
        logger.info(f"First few LLA positions:\n{lla_positions[:5]}")
        
        # Create DataFrames for Kepler.gl
        waypoints_df = pd.DataFrame({
            'latitude': lla_waypoints[:, 0],
            'longitude': lla_waypoints[:, 1],
            'altitude': lla_waypoints[:, 2],
            'type': 'waypoint'
        })
        logger.info(f"Number of waypoints: {len(waypoints_df)}")
        
        # Create trajectory DataFrame with proper line segments
        trajectory_data = []
        for i in range(len(lla_positions) - 1):
            trajectory_data.append({
                'latitude': lla_positions[i, 0],
                'longitude': lla_positions[i, 1],
                'altitude': lla_positions[i, 2],
                'type': 'trajectory',
                'next_lat': lla_positions[i + 1, 0],
                'next_lon': lla_positions[i + 1, 1],
                'next_alt': lla_positions[i + 1, 2]
            })
        
        trajectory_df = pd.DataFrame(trajectory_data)
        logger.info(f"Number of trajectory segments: {len(trajectory_df)}")
        logger.info(f"First few trajectory points:\n{trajectory_df.head()}")
        
        # Create Kepler.gl map
        map_1 = KeplerGl(height=800, data={'waypoints': waypoints_df, 'trajectory': trajectory_df})
        
        # Configure map
        config = {
            'version': 'v1',
            'config': {
                'visState': {
                    'filters': [],
                    'layers': [
                        {
                            'id': 'waypoints',
                            'type': 'point',
                            'config': {
                                'dataId': 'waypoints',
                                'label': 'Waypoints',
                                'color': [255, 0, 0],
                                'columns': {
                                    'lat': 'latitude',
                                    'lng': 'longitude',
                                    'altitude': 'altitude'
                                },
                                'isVisible': True,
                                'visConfig': {
                                    'radius': 8,
                                    'fixedRadius': True,
                                    'opacity': 0.8,
                                    'outline': False,
                                    'thickness': 2,
                                    'colorRange': {
                                        'name': 'Custom',
                                        'type': 'sequential',
                                        'category': 'Custom',
                                        'colors': ['#ff0000']
                                    },
                                    'hi-precision': False
                                },
                                'hidden': False,
                                'textLabel': {
                                    'field': {'name': 'type', 'type': 'string'},
                                    'color': [255, 255, 255],
                                    'size': 18,
                                    'offset': [0, 0],
                                    'anchor': 'start',
                                    'alignment': 'center'
                                }
                            },
                            'visualChannels': {
                                'colorField': None,
                                'colorScale': 'ordinal',
                                'sizeField': None,
                                'sizeScale': 'linear',
                                'strokeColorField': None,
                                'strokeColorScale': 'ordinal',
                                'heightField': None,
                                'heightScale': 'linear',
                                'radiusField': None,
                                'radiusScale': 'linear',
                                'angleField': None,
                                'angleScale': 'linear'
                            }
                        },
                        {
                            'id': 'trajectory',
                            'type': 'line',
                            'config': {
                                'dataId': 'trajectory',
                                'label': 'Trajectory',
                                'color': [0, 255, 0],
                                'columns': {
                                    'lat0': 'latitude',
                                    'lng0': 'longitude',
                                    'lat1': 'next_lat',
                                    'lng1': 'next_lon',
                                    'altitude': 'altitude'
                                },
                                'isVisible': True,
                                'visConfig': {
                                    'opacity': 0.8,
                                    'thickness': 2,
                                    'colorRange': {
                                        'name': 'Custom',
                                        'type': 'sequential',
                                        'category': 'Custom',
                                        'colors': ['#00ff00']
                                    },
                                    'hi-precision': True,
                                    'elevationScale': 1,
                                    'enable3d': True,
                                    'striped': False,
                                    'dashed': False
                                },
                                'hidden': False,
                                'textLabel': {
                                    'field': {'name': 'type', 'type': 'string'},
                                    'color': [255, 255, 255],
                                    'size': 18,
                                    'offset': [0, 0],
                                    'anchor': 'start',
                                    'alignment': 'center'
                                }
                            },
                            'visualChannels': {
                                'colorField': None,
                                'colorScale': 'ordinal',
                                'sizeField': None,
                                'sizeScale': 'linear',
                                'strokeColorField': None,
                                'strokeColorScale': 'ordinal',
                                'heightField': None,
                                'heightScale': 'linear',
                                'radiusField': None,
                                'radiusScale': 'linear',
                                'angleField': None,
                                'angleScale': 'linear'
                            }
                        }
                    ],
                    'interactionConfig': {
                        'tooltip': {
                            'fieldsToShow': {
                                'waypoints': ['type', 'altitude'],
                                'trajectory': ['type', 'altitude']
                            },
                            'enabled': True
                        },
                        'brush': {
                            'size': 0.5,
                            'enabled': False
                        },
                        'geocoder': {
                            'enabled': True
                        },
                        'coordinate': {
                            'enabled': True
                        }
                    },
                    'layerBlending': 'normal',
                    'splitMaps': [],
                    'animationConfig': {
                        'currentTime': None,
                        'speed': 1
                    }
                },
                'mapState': {
                    'bearing': 0,
                    'latitude': ref_lat,
                    'longitude': ref_lon,
                    'pitch': 60,
                    'zoom': 11,
                    'isSplit': False
                },
                'mapStyle': {
                    'styleType': 'dark',
                    'topLayerGroups': {},
                    'visibleLayerGroups': {
                        'label': True,
                        'road': True,
                        'building': True,
                        'water': True,
                        'land': True,
                        '3d building': True
                    },
                    'mapStyles': {}
                }
            }
        }
        
        # Apply configuration
        map_1.config = config
        
        # Save to HTML
        map_1.save_to_html(file_name=output_file)
        logger.info(f"Kepler visualization saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating Kepler visualization: {e}")
        raise

if __name__ == "__main__":
    # Define file paths
    state_file = "bT_state.npz"
    geojson_file = "scripts/UCB_NASA.json"
    
    # Create visualization
    create_kepler_visualization(state_file, geojson_file) 