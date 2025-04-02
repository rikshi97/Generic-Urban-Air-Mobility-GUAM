import gymnasium as gym
import numpy as np
from gymnasium import spaces
import jax.numpy as jnp

from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.guam_types import RefInputs
from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import lift_cruise_reference_inputs

class GuamEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(GuamEnv, self).__init__()
        
        # Initialize GUAM system
        self.guam = FuncGUAM()
        self.dt = self.guam.dt
        
        # Define action space
        # Actions are the reference inputs: [forward velocity, side velocity, upward velocity, yaw rate]
        self.action_space = spaces.Box(
            low=np.array([-20.0, -10.0, -10.0, -np.pi/2]),  # Minimum values
            high=np.array([20.0, 10.0, 10.0, np.pi/2]),      # Maximum values
            dtype=np.float32
        )
        
        # Define observation space
        # State includes: [position (3), velocity (3), angular velocity (3), quaternion (4)]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 13),  # 13 state variables
            high=np.array([np.inf] * 13),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = GuamState.create()
        
        # Define goal state with reasonable target position and zero velocities
        self.goal_state = np.zeros(13)
        self.goal_state[6:9] = np.array([50.0, 0.0, 100.0])  # Target position: 50m forward, 100m altitude
        
        # Define reward weights
        self.position_weight = 1.0
        self.velocity_weight = 0.5
        self.angular_weight = 0.3
        self.quaternion_weight = 0.2
        
        # For gymnasium compatibility
        self.render_mode = None

    def step(self, action):
        # Convert action to reference inputs
        ref_inputs = RefInputs(
            Vel_bIc_des=jnp.array([action[0], action[1], action[2]]),
            Pos_des=jnp.zeros(3),  # Target position (can be modified based on task)
            Chi_des=jnp.array(0.0),  # Target heading (can be modified based on task)
            Chi_dot_des=action[3]
        )
        
        # Step the system
        self.state = self.guam.step(self.dt, self.state, ref_inputs)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self._is_done()
        truncated = False
        
        # Additional info
        info = {}
        
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # Reset with random initial state
        super().reset(seed=seed)
        self.state = GuamState.create()
        
        # Randomize initial position within reasonable bounds
        self.state.aircraft[6:9] = np.array([
            np.random.uniform(-20.0, 20.0),   # x position
            np.random.uniform(-20.0, 20.0),   # y position
            np.random.uniform(50.0, 150.0)    # z position (altitude)
        ])
        
        # Small random initial velocities
        self.state.aircraft[0:3] = np.random.uniform(-2.0, 2.0, 3)
        
        # Small random initial angular velocities
        self.state.aircraft[3:6] = np.random.uniform(-0.1, 0.1, 3)
        
        # Keep quaternion as identity (no initial rotation)
        self.state.aircraft[9:13] = np.array([0.0, 0.0, 0.0, 1.0])
        
        return self._get_observation(), {}

    def render(self):
        # Implement rendering if needed
        pass

    def close(self):
        # Clean up resources if needed
        pass

    def _get_observation(self):
        """Convert state to observation vector"""
        return np.array(self.state.aircraft, dtype=np.float32)

    def _calculate_reward(self):
        """Calculate reward based on state error"""
        # Position error (penalize distance from goal)
        pos_error = np.linalg.norm(self.state.aircraft[6:9] - self.goal_state[6:9])
        pos_reward = -0.1 * min(pos_error, 100.0)  # Cap the position error penalty
        
        # Velocity error (encourage matching desired velocity)
        vel_error = np.linalg.norm(self.state.aircraft[0:3] - self.goal_state[0:3])
        vel_reward = -0.05 * min(vel_error, 50.0)  # Cap the velocity error penalty
        
        # Angular velocity error (penalize excessive rotation)
        ang_error = np.linalg.norm(self.state.aircraft[3:6] - self.goal_state[3:6])
        ang_reward = -0.02 * min(ang_error, 20.0)  # Cap the angular velocity error penalty
        
        # Quaternion error (maintain desired orientation)
        quat_error = np.linalg.norm(self.state.aircraft[9:13] - self.goal_state[9:13])
        quat_reward = -0.01 * min(quat_error, 10.0)  # Cap the quaternion error penalty
        
        # Success reward (when close to goal)
        success_threshold = 5.0  # meters
        if pos_error < success_threshold and vel_error < 2.0:
            success_reward = 10.0
        else:
            success_reward = 0.0
        
        # Living bonus (encourage longer episodes)
        living_bonus = 0.1
        
        # Total reward
        reward = pos_reward + vel_reward + ang_reward + quat_reward + success_reward + living_bonus
        
        return reward

    def _is_done(self):
        """Check if episode should end"""
        # 1. Position error too large
        pos_error = np.linalg.norm(self.state.aircraft[6:9] - self.goal_state[6:9])
        if pos_error > 200.0:  # Increased to 200 meters
            return True
            
        # 2. Angular velocity too large
        ang_vel = np.linalg.norm(self.state.aircraft[3:6])
        if ang_vel > 10.0:  # Increased to 10 rad/s
            return True
            
        # 3. Altitude bounds
        altitude = self.state.aircraft[8]
        if altitude < 0.0 or altitude > 500.0:  # More reasonable altitude bounds
            return True
            
        # 4. Episode length limit (handled by the training wrapper)
        return False 