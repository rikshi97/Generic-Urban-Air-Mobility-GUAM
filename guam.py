import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

class GuamSimulation:
    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self._setup_paths()
        
    def _setup_paths(self):
        """Configure MATLAB path and required toolboxes"""
        self.eng.addpath(self.eng.genpath('/home/rishi/Berkeley/CITRIS/Generic-Urban-Air-Mobility-GUAM-waypoint-visualiser'))  # Update with actual GUAM path
        self.eng.eval("addpath(genpath('GUAM/Libraries'))", nargout=0)
        self.eng.eval("checkSetup();", nargout=0)  # GUAM's setup validation

    def convert_coordinates(self, points, src_crs=3993, dest_crs=4326):
        """
        Convert coordinates between CRS systems
        :param points: Nx3 array of [x,y,z] or [lat,lon,alt]
        :param src_crs: EPSG code of source CRS
        :param dest_crs: EPSG code of target CRS
        """
        lat = matlab.double(points[:,0].tolist())
        lon = matlab.double(points[:,1].tolist())
        alt = matlab.double(points[:,2].tolist())
        
        converted = self.eng.projfwd(src_crs, lat, lon, alt, nargout=3)
        return np.array(converted).T

    def generate_trajectory(self, waypoints, convert_crs=True):
        """
        Create GUAM trajectory object from waypoints
        :param waypoints: Nx3 numpy array of [lat, lon, alt]
        :param convert_crs: Convert from Guam CRS 3993 to WGS84
        """
        if convert_crs:
            waypoints = self.convert_coordinates(waypoints)
            
        mat_waypoints = matlab.double(waypoints.tolist())
        self.eng.workspace['waypoints'] = mat_waypoints
        self.eng.eval("traj = geoTrajectory(waypoints, 'PositionInputFormat', 'Geodetic');", nargout=0)
        return self.eng.workspace['traj']

    def configure_simulation(self, config):
        """
        Set simulation parameters
        :param config: Dictionary of simulation parameters
        Example config:
        {
            'wind_model': 'DrydenGust',
            'wind_speed': [5, 30, 0],
            'sensor_noise': True,
            'visualization': '3D'
        }
        """
        self.eng.workspace['SimIn'] = self.eng.struct()
        for key, value in config.items():
            self.eng.setfield(self.eng.workspace['SimIn'], key, matlab.double(value))
            
        self.eng.eval("SimIn = setupSimulation(SimIn);", nargout=0)

    def run_simulation(self, trajectory):
        """Execute GUAM simulation with configured parameters"""
        self.eng.workspace['traj'] = trajectory
        try:
            self.eng.eval("SimOut = runSimulation(traj, SimIn);", nargout=0)
            return self.eng.workspace['SimOut']
        except Exception as e:
            print(f"Simulation error: {e}")
            return None

    def analyze_results(self, SimOut):
        """Extract and visualize key performance metrics"""
        results = {
            'position': np.array(self.eng.getfield(SimOut, 'Position_truth')),
            'velocity': np.array(self.eng.getfield(SimOut, 'Velocity_truth')),
            'energy': np.array(self.eng.getfield(SimOut, 'BatteryEnergy')),
            'time': np.array(self.eng.getfield(SimOut, 'Time'))
        }
        
        # Plot trajectory
        plt.figure(figsize=(12, 6))
        plt.plot(results['position'][:,1], results['position'][:,0])
        plt.xlabel('Longitude (°)')
        plt.ylabel('Latitude (°)')
        plt.title('Flight Trajectory')
        plt.grid(True)
        
        return results