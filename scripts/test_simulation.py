import os
import time
from loguru import logger
from scripts.run_simulation import run_simulation

def test_simulation_parameters():
    """Test different simulation parameters to find optimal settings."""
    geojson_file = "scripts/UCB_NASA.json"
    output_dir = "results"
    
    # Test cases with different parameters
    test_cases = [
        {
            "name": "Fast Iteration",
            "dt": 0.1,
            "base_batch_size": 32,
            "downsample_factor": 5
        },
        {
            "name": "High Accuracy",
            "dt": 0.05,
            "base_batch_size": 16,
            "downsample_factor": 2
        },
        {
            "name": "Balanced",
            "dt": 0.075,
            "base_batch_size": 24,
            "downsample_factor": 3
        }
    ]
    
    results = []
    for case in test_cases:
        logger.info(f"\nTesting {case['name']} configuration:")
        logger.info(f"dt: {case['dt']}")
        logger.info(f"base_batch_size: {case['base_batch_size']}")
        logger.info(f"downsample_factor: {case['downsample_factor']}")
        
        # Create case-specific output directory
        case_dir = os.path.join(output_dir, case['name'].lower().replace(' ', '_'))
        
        # Run simulation
        start_time = time.time()
        simulation_time = run_simulation(
            geojson_file=geojson_file,
            output_dir=case_dir,
            dt=case['dt'],
            base_batch_size=case['base_batch_size'],
            downsample_factor=case['downsample_factor']
        )
        
        # Record results
        results.append({
            "name": case['name'],
            "simulation_time": simulation_time,
            "parameters": case
        })
        
        logger.info(f"Simulation completed in {simulation_time:.2f} seconds")
    
    # Print summary
    logger.info("\nTest Results Summary:")
    for result in results:
        logger.info(f"\n{result['name']}:")
        logger.info(f"Simulation time: {result['simulation_time']:.2f} seconds")
        logger.info(f"Parameters: {result['parameters']}")

if __name__ == "__main__":
    test_simulation_parameters() 