%Script to read waypoints from CSV file, convert to NED and run a simulation
%of the GUAM model using MATLAB's waypointTrajectory function

clear all
close all

addpath('./Bez_Functions/');

% Define the Simulink model
model = 'GUAM';

% Setup required structures for simSetup
userStruct.variants.refInputType = 4; % 4=Piecewise Bezier
SimIn = struct(); % Initialize SimIn structure

%Read waypoints from CSV file
waypoints = readtable('wp_BN.csv');

lat = waypoints.latitude;
lon = waypoints.longitude;
alt = waypoints.altitude;

%Target Speeds
cruise_speed = 176.4; %ft/s (120 kts)
takeoff_target_speed = 83; %ft/s (57 kts)
landing_target_speed = 30; %ft/s (57 kts)
flight_time = 40*60; %seconds

%Convert to NED
% First waypoint is used as reference point for NED conversion
lla0 = [lat(1), lon(1), alt(1)];  % Reference point (first waypoint)
ned_waypoints = zeros(length(lat), 3);  % Preallocate NED coordinates

% Convert each waypoint to NED coordinates
for i = 1:length(lat)
    lla = [lat(i), lon(i), alt(i)];
    ned_waypoints(i,:) = lla2ned(lla, lla0, 'flat');  % Using flat Earth approximation
end

%Initialize waypoints matrices - now as Nx3 matrices
num_waypoints = size(ned_waypoints, 1);
wptsX = zeros(num_waypoints, 3);  % [pos vel acc]
wptsY = zeros(num_waypoints, 3);
wptsZ = zeros(num_waypoints, 3);

% Assign positions
wptsX(:,1) = ned_waypoints(:,2);  % East coordinates
wptsY(:,1) = ned_waypoints(:,1);  % North coordinates
wptsZ(:,1) = ned_waypoints(:,3);  % Down coordinates

% Generate time vector
cruise_time = flight_time - 180;  % Subtract 120 seconds for takeoff and landing
num_cruise_waypoints = num_waypoints - 4;  % Exclude first 2 and last 2 waypoints

% Calculate segment lengths for cruise waypoints
segment_lengths = zeros(num_cruise_waypoints + 1, 1);
for i = 3:num_waypoints-2
    dx = wptsX(i+1,1) - wptsX(i,1);
    dy = wptsY(i+1,1) - wptsY(i,1);
    dz = wptsZ(i+1,1) - wptsZ(i,1);
    segment_lengths(i-2) = sqrt(dx^2 + dy^2 + dz^2);
end

% Calculate total cruise distance
total_cruise_distance = sum(segment_lengths);

% Calculate time steps proportional to segment lengths
time_steps = (segment_lengths / total_cruise_distance) * cruise_time;

% Initialize time vector
time_vector = zeros(1, num_waypoints);
time_vector(1) = 0;  % Start time
time_vector(2) = 60;  % 60 seconds for takeoff
time_vector(end-1) = flight_time - 60;  % 60 seconds for landing
time_vector(end) = flight_time;  % End time

% Assign times for cruise waypoints
current_time = time_vector(2);  % Start from end of takeoff
for i = 3:num_waypoints-2
    current_time = current_time + time_steps(i-2);
    time_vector(i) = current_time;
end

% Create waypoint trajectory object
% Combine positions into a single matrix
positions = [wptsX(:,1), wptsY(:,1), wptsZ(:,1)];

% Create time of arrival array
timeOfArrival = time_vector';

% Create waypoint trajectory object with supported properties
trajectory = waypointTrajectory(positions, timeOfArrival, ...
    'SampleRate', 100, ...  % 100 Hz sampling rate
    'ReferenceFrame', 'NED', ...
    'AutoPitch', false, ...  % Don't automatically adjust pitch
    'AutoBank', false);      % Don't automatically adjust bank

% Generate trajectory
[position, orientation, velocity, acceleration, angularVelocity] = trajectory();

% Create timeseries objects for velocity and acceleration
ts_vel_x = timeseries(velocity(:,1), timeOfArrival);
ts_vel_y = timeseries(velocity(:,2), timeOfArrival);
ts_vel_z = timeseries(velocity(:,3), timeOfArrival);

ts_acc_x = timeseries(acceleration(:,1), timeOfArrival);
ts_acc_y = timeseries(acceleration(:,2), timeOfArrival);
ts_acc_z = timeseries(acceleration(:,3), timeOfArrival);

% Resample at waypoint times
wptsX(:,2) = resample(ts_vel_x, time_vector).Data;
wptsY(:,2) = resample(ts_vel_y, time_vector).Data;
wptsZ(:,2) = resample(ts_vel_z, time_vector).Data;

wptsX(:,3) = resample(ts_acc_x, time_vector).Data;
wptsY(:,3) = resample(ts_acc_y, time_vector).Data;
wptsZ(:,3) = resample(ts_acc_z, time_vector).Data;

% Create 3D plot of generated waypoints
figure;
plot3(wptsX(:,1), wptsY(:,1), wptsZ(:,1), 'r-o', 'LineWidth', 2);
grid on;
xlabel('East (ft)');
ylabel('North (ft)');
zlabel('Altitude (ft)');
title('Generated Waypoints in NED Frame');
view(45, 30); % Set view angle for better 3D visualization
saveas(gcf, 'generated_waypoints_plot_trajectory.png');

% Ensure time vectors are row vectors
time_wptsX = time_vector;
time_wptsY = time_vector;
time_wptsZ = time_vector;

% Create the structure and save into the trajectory file
pwcurve.waypoints = {wptsX, wptsY, wptsZ};
pwcurve.time_wpts = {time_vector, time_vector, time_vector};
save('./NED_PW_Bezier_Traj_trajectory','pwcurve','-v7.3');
if exist('target','var')
    clear target;
end

% Create comprehensive waypoint data table
waypoint_data = table();
waypoint_data.latitude = lat;
waypoint_data.longitude = lon;
waypoint_data.altitude = alt;
waypoint_data.x_ned = wptsX(:,1);
waypoint_data.y_ned = wptsY(:,1);
waypoint_data.z_ned = wptsZ(:,1);
waypoint_data.x_vel = wptsX(:,2);
waypoint_data.y_vel = wptsY(:,2);
waypoint_data.z_vel = wptsZ(:,2);
waypoint_data.x_acc = wptsX(:,3);
waypoint_data.y_acc = wptsY(:,3);
waypoint_data.z_acc = wptsZ(:,3);
waypoint_data.time = time_vector';

% Save the comprehensive waypoint data to CSV
writetable(waypoint_data, 'waypoint_data_trajectory.csv');

% Create bus signals for simulation
target.RefInput.Bezier.waypoints = {wptsX, wptsY, wptsZ};
target.RefInput.Bezier.time_wpts = {time_vector, time_vector, time_vector};

% Set desired Initial condition vars
target.RefInput.Vel_bIc_des    = [wptsX(1,2); wptsY(1,2); wptsZ(1,2)];  % Initial velocity in body frame
target.RefInput.pos_des        = [wptsX(1,1); wptsY(1,1); wptsZ(1,1)];  % Initial position
target.RefInput.chi_des        = 0;  % Initial heading angle
target.RefInput.chi_dot_des    = 0;  % Initial heading rate
target.RefInput.trajectory.refTime = [time_vector(1) time_vector(end)];

% Add required target parameters for non-TIMESERIES refInput Type
target.tas = cruise_speed;  % True airspeed in ft/s
target.gndtrack = 0;        % Ground track angle in radians
target.stopTime = flight_time;  % Simulation stop time in seconds

% Initialize the sim
simSetup;

% Open the simulation
open(model);

% Run the simulation
sim(model);

% Save the simulation results
simPlots_GUAM_save;

% Close the simulation
close_system(model); 