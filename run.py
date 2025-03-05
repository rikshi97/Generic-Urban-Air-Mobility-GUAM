from guam import GuamSimulation

sim = GuamSimulation()

# Define waypoints in Guam State Plane CRS (EPSG:3993)
waypoints = np.array([
    [13.5123, 144.8231, 100],  # Start point
    [13.5345, 144.8012, 150],  # Intermediate point
    [13.5421, 144.7915, 200]   # Destination
])

# Configure simulation parameters
config = {
    'WindModel': 'DrydenGust',
    'WindSpeed': [5, 30, 0],
    'SensorNoise': True,
    'Visualization': '3D'
}

# Run simulation pipeline
trajectory = sim.generate_trajectory(waypoints)
sim.configure_simulation(config)
sim_out = sim.run_simulation(trajectory)

if sim_out:
    results = sim.analyze_results(sim_out)
    plt.show()

# Close MATLAB engine
sim.eng.quit()
