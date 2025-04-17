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
    flight_duration = 2400; % 40 minutes flight
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
    SimIn.propType = 'None'; % Propulsion model
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
    
    % Create time points for the trajectory
    time_points = linspace(0, flight_duration, num_points);
    
    % Create waypoints for Bezier curve
    % Each waypoint needs [position, velocity, acceleration]
    wptsX = zeros(num_points, 3);
    wptsY = zeros(num_points, 3);
    wptsZ = zeros(num_points, 3);
    
    % Position components in ECEF
    wptsX(:,1) = ecef(:,1); % X
    wptsY(:,1) = ecef(:,2); % Y
    wptsZ(:,1) = ecef(:,3); % Z
    
    % Calculate velocities using central differences
    dt = diff(time_points);
    for i = 2:num_points-1
        wptsX(i,2) = (wptsX(i+1,1) - wptsX(i-1,1)) / (dt(i-1) + dt(i));
        wptsY(i,2) = (wptsY(i+1,1) - wptsY(i-1,1)) / (dt(i-1) + dt(i));
        wptsZ(i,2) = (wptsZ(i+1,1) - wptsZ(i-1,1)) / (dt(i-1) + dt(i));
    end
    
    % Handle endpoints
    wptsX(1,2) = (wptsX(2,1) - wptsX(1,1)) / dt(1);
    wptsY(1,2) = (wptsY(2,1) - wptsY(1,1)) / dt(1);
    wptsZ(1,2) = (wptsZ(2,1) - wptsZ(1,1)) / dt(1);
    
    wptsX(end,2) = (wptsX(end,1) - wptsX(end-1,1)) / dt(end);
    wptsY(end,2) = (wptsY(end,1) - wptsY(end-1,1)) / dt(end);
    wptsZ(end,2) = (wptsZ(end,1) - wptsZ(end-1,1)) / dt(end);
    
    % Set accelerations to zero (can be improved if needed)
    wptsX(:,3) = 0;
    wptsY(:,3) = 0;
    wptsZ(:,3) = 0;
    
    % Create the trajectory structure
    pwcurve.waypoints = {wptsX, wptsY, wptsZ};
    pwcurve.time_wpts = {time_points, time_points, time_points};
    
    % Save the trajectory
    save('ecef_trajectory.mat', 'pwcurve', '-v7.3');
    fprintf('Successfully generated and saved ECEF trajectory.\n');
    
    % Plot the trajectory for verification in 3D ECEF coordinates
    try
        % Create a new figure with specific size
        fig = figure('Visible', 'off', 'Position', [100 100 800 600]);
        
        % Create the 3D plot
        plot3(wptsX(:,1), wptsY(:,1), wptsZ(:,1), 'b-', 'LineWidth', 2);
        hold on;
        scatter3(wptsX(1,1), wptsY(1,1), wptsZ(1,1), 100, 'g', 'filled'); % Start point
        scatter3(wptsX(end,1), wptsY(end,1), wptsZ(end,1), 100, 'r', 'filled'); % End point
        
        % Add labels and title
        xlabel('X (m)', 'FontSize', 12);
        ylabel('Y (m)', 'FontSize', 12);
        zlabel('Z (m)', 'FontSize', 12);
        title('3D Trajectory in ECEF Coordinates', 'FontSize', 14);
        
        % Customize the plot
        grid on;
        axis equal;
        view(45, 30); % Set view angle
        
        % Add legend
        legend('Trajectory', 'Start Point', 'End Point', 'Location', 'best');
        
        % Save the figure
        print('ecef_trajectory_verification', '-dpng', '-r300');
        fprintf('Successfully saved trajectory verification plot.\n');
        
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