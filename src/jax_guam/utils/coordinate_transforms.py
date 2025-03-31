import jax.numpy as jnp
import numpy as np
from typing import Tuple, Union
from geopy.distance import geodesic

# Earth model parameters
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_F = 1/298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis (m)
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # First eccentricity squared

def lla_to_ecef(lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
    """
    Convert LLA coordinates to ECEF coordinates using WGS84 ellipsoid.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters
        
    Returns:
        Tuple of (x, y, z) ECEF coordinates in meters
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Calculate N (prime vertical radius of curvature)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad)**2)
    
    # Calculate ECEF coordinates
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)
    
    return x, y, z

def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert ECEF coordinates to LLA coordinates using WGS84 ellipsoid.
    
    Args:
        x, y, z: ECEF coordinates in meters
        
    Returns:
        Tuple of (lat, lon, alt) in degrees and meters
    """
    # Calculate longitude
    lon = np.degrees(np.arctan2(y, x))
    
    # Calculate latitude and altitude using iterative method
    p = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, p * (1 - WGS84_E2)))
    
    # Iterative calculation of latitude and altitude
    for _ in range(10):  # Usually converges in 3-4 iterations
        lat_rad = np.radians(lat)
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad)**2)
        alt = p / np.cos(lat_rad) - N
        lat_new = np.degrees(np.arctan2(z, p * (1 - WGS84_E2 * (N / (N + alt)))))
        
        if abs(lat_new - lat) < 1e-10:
            break
        lat = lat_new
    
    return lat, lon, alt

def lla_to_enu(lat: float, lon: float, alt: float,
                ref_lat: float, ref_lon: float, ref_alt: float) -> Tuple[float, float, float]:
    """
    Convert LLA coordinates to ENU coordinates relative to a reference point.
    Uses geodesic calculations for accurate distance and bearing.
    
    Args:
        lat, lon, alt: Target point coordinates
        ref_lat, ref_lon, ref_alt: Reference point coordinates
        
    Returns:
        Tuple of (east, north, up) coordinates in meters
    """
    # Calculate distances using geodesic
    d_east, d_north = geodesic((ref_lat, ref_lon), (ref_lat, lon)).meters, \
                      geodesic((ref_lat, ref_lon), (lat, ref_lon)).meters
    
    # Calculate up component
    d_up = alt - ref_alt
    
    return d_east, d_north, d_up

def enu_to_lla(east: float, north: float, up: float,
                ref_lat: float, ref_lon: float, ref_alt: float) -> Tuple[float, float, float]:
    """
    Convert ENU coordinates to LLA coordinates relative to a reference point.
    
    Args:
        east, north, up: ENU coordinates in meters
        ref_lat, ref_lon, ref_alt: Reference point coordinates
        
    Returns:
        Tuple of (lat, lon, alt) in degrees and meters
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

def validate_coordinates(lat: float, lon: float, alt: float) -> bool:
    """
    Validate LLA coordinates are within valid ranges.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters
        
    Returns:
        bool: True if coordinates are valid
    """
    return (-90 <= lat <= 90 and
            -180 <= lon <= 180 and
            -1000 <= alt <= 100000)  # Reasonable altitude range in meters

def batch_lla_to_enu(lats: np.ndarray, lons: np.ndarray, alts: np.ndarray,
                     ref_lat: float, ref_lon: float, ref_alt: float) -> np.ndarray:
    """
    Convert a batch of LLA coordinates to ENU coordinates.
    
    Args:
        lats, lons, alts: Arrays of coordinates
        ref_lat, ref_lon, ref_alt: Reference point coordinates
        
    Returns:
        Array of (east, north, up) coordinates
    """
    enu_coords = np.zeros((len(lats), 3))
    for i in range(len(lats)):
        enu_coords[i] = lla_to_enu(lats[i], lons[i], alts[i],
                                  ref_lat, ref_lon, ref_alt)
    return enu_coords 