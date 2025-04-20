function generate_ecef_trajectory(csv_file, flight_duration)
% GENERATE_ECEF_TRAJECTORY - Generates a trajectory in ECEF frame from CSV waypoints
%
% Inputs:
%   csv_file - Path to the CSV file containing waypoints
%   flight_duration - Total flight duration in seconds
%
% Example usage:
%   generate_ecef_trajectory('wp_BN.csv', 2400)

% First run setupPath to get basic paths set up
setupPath;

% Default values if not provided
if nargin < 1
    csv_file = 'wp_BN.csv';
end
if nargin < 2
    flight_duration = 40 * 60; % 40 minutes flight
end

% Constants
altitude_scaling = 3.28084; % Convert meters to feet (GUAM uses feet)

try
    fprintf('Reading CSV file and generating ECEF trajectory...\n');
    
    % Read the CSV file
    data = readtable(csv_file);
    
    % Extract coordinates from CSV
    lla = [data.latitude, data.longitude, data.altitude];
    
    % Convert altitude from meters to feet
    lla(:,3) = lla(:,3) * altitude_scaling;
    
    % Store additional waypoint information
    waypoint_info = struct();
    waypoint_info.ids = data.waypoint_id;
    waypoint_info.flight_modes = data.flight_mode;
    waypoint_info.origins = data.origin;
    waypoint_info.destinations = data.destination;
    
    % Initialize ECEF matrix
    num_points = size(lla, 1);
    ecef = zeros(num_points, 3);
    
    % Initialize SimIn structure with required fields
    SimIn = struct();
    
    % Setup model types
    SimIn.fmType = 'Polynomial'; % Force model type
    SimIn.atmosType = 'US_STD_ATMOS_76'; % Atmosphere model
    SimIn.turbType = 'None'; % Turbulence model
    SimIn.ctrlType = 'BASELINE'; % Controller type
    SimIn.actType = 'FirstOrder'; % Actuator type
    SimIn.propType = 'FirstOrder'; % Propulsion model
    SimIn.eomType = 'Simple'; % Equations of motion type
    SimIn.sensorType = 'None'; % Sensor model
    
    % Setup model parameters
    SimIn.Model = struct();
    SimIn.Model.b = 30; % Wingspan in meters (example value)
    SimIn.Model.mass = 2000; % Mass in kg (example value)
    SimIn.Model.I = eye(3); % Inertia matrix (example value)
    
    % Setup units
    SimIn.Units = struct();
    
    % Basic units
    SimIn.Units.m = 1;
    SimIn.Units.ft = 0.3048;
    SimIn.Units.s = 1;
    SimIn.Units.g0 = 9.80665; % Standard gravity in m/s^2
    SimIn.Units.deg = pi/180;
    SimIn.Units.knot = 0.514444; % m/s
    SimIn.Units.rad = 1;
    
    % Temperature units
    SimIn.Units.K = 1; % Kelvin
    SimIn.Units.degR = 5/9; % Degrees Rankine to Kelvin conversion
    SimIn.Units.degC = 1; % Celsius
    SimIn.Units.degF = 5/9; % Fahrenheit to Celsius conversion
    
    % Force and mass units
    SimIn.Units.N = 1; % Newton
    SimIn.Units.kg = 1; % Kilogram
    SimIn.Units.lbf = 4.448222; % Pound force
    SimIn.Units.slug = 14.5939; % Slug
    
    % Pressure units
    SimIn.Units.Pa = 1; % Pascal
    SimIn.Units.atm = 101325; % Standard atmosphere
    SimIn.Units.psi = 6894.757; % Pounds per square inch
    
    % Other base units
    SimIn.Units.mol = 1; % Mole
    SimIn.Units.J = 1; % Joule
    SimIn.Units.W = 1; % Watt
    SimIn.Units.A = 1; % Ampere
    SimIn.Units.V = 1; % Volt
    SimIn.Units.Hz = 1; % Hertz
    
    % Compound units
    SimIn.Units.mps = SimIn.Units.m / SimIn.Units.s; % meters per second
    SimIn.Units.mps2 = SimIn.Units.mps / SimIn.Units.s; % meters per second squared
    SimIn.Units.kgm2 = SimIn.Units.kg * SimIn.Units.m^2; % kilogram meter squared
    SimIn.Units.Nm = SimIn.Units.N * SimIn.Units.m; % Newton meter
    
    % Get Earth parameters from setupEnvironment
    env = setupEnvironment(SimIn);
    
    % Convert LLA to ECEF iteratively
    fprintf('Converting coordinates to ECEF frame...\n');
    for i = 1:num_points
        try
            % Convert current point to ECEF
            current_lla = lla(i,:);
            
            % Debug print for first point
            if i == 1
                fprintf('First point conversion:\n');
                fprintf('Current: lat=%.6f, lon=%.6f, alt=%.6f\n', current_lla(1), current_lla(2), current_lla(3));
            end
            
            % Convert to ECEF using WGS84 parameters
            lat = current_lla(1) * pi/180; % Convert to radians
            lon = current_lla(2) * pi/180;
            alt = current_lla(3);
            
            % WGS84 parameters
            a = env.Earth.RadiusEquator;
            e = env.Earth.Eccentricity;
            
            % Calculate N (radius of curvature in prime vertical)
            N = a / sqrt(1 - e^2 * sin(lat)^2);
            
            % Calculate ECEF coordinates
            x = (N + alt) * cos(lat) * cos(lon);
            y = (N + alt) * cos(lat) * sin(lon);
            z = (N * (1 - e^2) + alt) * sin(lat);
            
            ecef(i,:) = [x, y, z];
            
            % Print progress every 10 points
            if mod(i, 10) == 0
                fprintf('Converted point %d of %d\n', i, num_points);
            end
        catch ME
            fprintf('Error converting point %d: %s\n', i, ME.message);
            fprintf('Current point: lat=%.6f, lon=%.6f, alt=%.6f\n', current_lla(1), current_lla(2), current_lla(3));
            rethrow(ME);
        end
    end
    fprintf('Coordinate conversion complete.\n');
    
    % Convert ECEF to NED using first point as reference
    ref_lla = lla(1,:); % Reference point (first waypoint)
    ned = zeros(size(ecef));
    
    fprintf('Converting ECEF to NED coordinates...\n');
    for i = 1:size(ecef,1)
        % Convert current ECEF point to NED
        current_lla = lla(i,:);
        ned(i,:) = lla2ned(current_lla, ref_lla, 'flat');
    end
    
    % Convert to feet for consistency with GUAM
    ned_ft = ned * 3.28084; % Convert meters to feet
    
    % Create waypoints for Bezier curve
    % Each waypoint needs [position, velocity, acceleration]
    % Format: 3 x N matrix where each column is [pos; vel; acc]
    num_states = 3; % [position, velocity, acceleration]
    wptsX = zeros(num_states, num_points);
    wptsY = zeros(num_states, num_points);
    wptsZ = zeros(num_states, num_points);
    
    % Position components in NED (feet)
    wptsX(1,:) = ned_ft(:,1)'; % North positions
    wptsY(1,:) = ned_ft(:,2)'; % East positions
    wptsZ(1,:) = -ned_ft(:,3)'; % Down positions (negative for altitude)
    
    % Define speeds in feet/s
    cruise_speed = 5 * 3.28084; % ft/s
    climb_speed = 2 * 3.28084;  % ft/s
    descent_speed = 2 * 3.28084; % ft/s
    
    % Calculate distances between waypoints
    distances = zeros(num_points-1, 1);
    for i = 1:num_points-1
        distances(i) = norm(ned_ft(i+1,:) - ned_ft(i,:));
    end
    
    % Calculate total distance
    total_distance = sum(distances);
    
    % Calculate time intervals based on cruise speed
    dt = distances / cruise_speed;
    
    % Add extra time for acceleration/deceleration
    dt = dt * 2.0; % Add 100% more time for even smoother transitions
    
    % Reconstruct time points based on calculated intervals
    time_points = [0; cumsum(dt)];
    
    % Set velocities for specific waypoints
    % First waypoint - zero velocity
    wptsX(2,1) = 0;
    wptsY(2,1) = 0;
    wptsZ(2,1) = 0;
    
    % Second waypoint - initial climb velocity (very smooth)
    direction = (ned_ft(2,:) - ned_ft(1,:)) / norm(ned_ft(2,:) - ned_ft(1,:));
    wptsX(2,2) = direction(1) * climb_speed * 0.5; % Further reduce initial velocity
    wptsY(2,2) = direction(2) * climb_speed * 0.5;
    wptsZ(2,2) = -direction(3) * climb_speed * 0.5; % Negative for altitude
    
    % Second to last waypoint - descent velocity (very smooth)
    direction = (ned_ft(end-1,:) - ned_ft(end-2,:)) / norm(ned_ft(end-1,:) - ned_ft(end-2,:));
    wptsX(2,end-1) = direction(1) * descent_speed * 0.5;
    wptsY(2,end-1) = direction(2) * descent_speed * 0.5;
    wptsZ(2,end-1) = -direction(3) * descent_speed * 0.5; % Negative for altitude
    
    % Last waypoint - zero velocity
    wptsX(2,end) = 0;
    wptsY(2,end) = 0;
    wptsZ(2,end) = 0;
    
    % Set cruise velocities for remaining waypoints with smooth transitions
    for i = 3:num_points-2
        % Calculate direction vector using previous and next points for smoother path
        prev_dir = (ned_ft(i,:) - ned_ft(i-1,:)) / norm(ned_ft(i,:) - ned_ft(i-1,:));
        next_dir = (ned_ft(i+1,:) - ned_ft(i,:)) / norm(ned_ft(i+1,:) - ned_ft(i,:));
        
        % Average the directions for smoother transition
        direction = (prev_dir + next_dir) / 2;
        direction = direction / norm(direction);
        
        % Calculate speed factor based on turn angle
        turn_angle = acos(dot(prev_dir, next_dir));
        speed_factor = cos(turn_angle/2)^2; % More aggressive speed reduction in turns
        
        % Set velocity components with smooth transition
        wptsX(2,i) = direction(1) * cruise_speed * speed_factor;
        wptsY(2,i) = direction(2) * cruise_speed * speed_factor;
        wptsZ(2,i) = -direction(3) * cruise_speed * speed_factor; % Negative for altitude
    end
    
    % Set accelerations for smoother transitions
    % Calculate accelerations based on velocity changes
    for i = 2:num_points-1
        dt_i = time_points(i+1) - time_points(i);
        
        % X acceleration
        v_diff_x = wptsX(2,i+1) - wptsX(2,i);
        wptsX(3,i) = v_diff_x / dt_i * 0.5; % Reduce acceleration magnitude
        
        % Y acceleration
        v_diff_y = wptsY(2,i+1) - wptsY(2,i);
        wptsY(3,i) = v_diff_y / dt_i * 0.5;
        
        % Z acceleration
        v_diff_z = wptsZ(2,i+1) - wptsZ(2,i);
        wptsZ(3,i) = v_diff_z / dt_i * 0.5;
    end
    
    % Set first and last point accelerations to zero
    wptsX(3,1) = 0;
    wptsY(3,1) = 0;
    wptsZ(3,1) = 0;
    wptsX(3,end) = 0;
    wptsY(3,end) = 0;
    wptsZ(3,end) = 0;
    
    % Ensure time points are row vectors for Simulink
    time_points = time_points';
    
    % Create the trajectory structure
    pwcurve.waypoints = {wptsX, wptsY, wptsZ};
    pwcurve.time_wpts = {time_points, time_points, time_points};
    
    % Save the trajectory
    save('ecef_trajectory.mat', 'pwcurve', '-v7.3');
    fprintf('Successfully generated and saved ECEF trajectory.\n');
    
    % Plot the trajectory for verification in NED coordinates
    try
        % Create a new figure with specific size
        fig = figure('Visible', 'off', 'Position', [100 100 1200 800]);
        
        % Create the 3D plot
        plot3(ned_ft(:,2), ned_ft(:,1), -ned_ft(:,3), 'b-', 'LineWidth', 2);
        hold on;
        scatter3(ned_ft(1,2), ned_ft(1,1), -ned_ft(1,3), 100, 'g', 'filled'); % Start point
        scatter3(ned_ft(end,2), ned_ft(end,1), -ned_ft(end,3), 100, 'r', 'filled'); % End point
        
        % Add labels and title
        xlabel('East (ft)', 'FontSize', 12);
        ylabel('North (ft)', 'FontSize', 12);
        zlabel('Up (ft)', 'FontSize', 12);
        title('3D Trajectory in NED Coordinates', 'FontSize', 14);
        
        % Customize the plot
        grid on;
        axis equal;
        
        % Calculate appropriate view angle based on trajectory
        if max(ned_ft(:,3)) - min(ned_ft(:,3)) < 100
            % If trajectory is mostly horizontal, use top-down view
            view(0, 90);
        else
            % Otherwise use 3D view
            view(45, 30);
        end
        
        % Add legend
        legend('Trajectory', 'Start Point', 'End Point', 'Location', 'best');
        
        % Add ground reference
        x_lim = xlim;
        y_lim = ylim;
        [X,Y] = meshgrid(linspace(x_lim(1), x_lim(2), 20), linspace(y_lim(1), y_lim(2), 20));
        Z = zeros(size(X));
        surf(X, Y, Z, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', 'k');
        
        % Save the figure
        print('ned_trajectory_verification', '-dpng', '-r300');
        fprintf('Successfully saved NED trajectory verification plot.\n');
        
        % Close the figure
        close(fig);
    catch ME
        fprintf('Warning: Could not create verification plot: %s\n', ME.message);
    end
    
catch ME
    fprintf('Error in generate_ecef_trajectory: %s\n', ME.message);
    rethrow(ME);
end
end 