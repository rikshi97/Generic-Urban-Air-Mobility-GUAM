%Script to read waypoints from CSV file, convert to NED and run a simulation
%of the GUAM model

clear all
close all

addpath('./Bez_Functions/');

% Define the Simulink model
model = 'GUAM';

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

%Generate waypoints in NED frame
% pos in ft, vel in ft/sec, and acc in ft/sec^2
% within row = pos vel acc, rows are waypoints

%Initialize waypoints matrices - now as Nx3 matrices
num_waypoints = size(ned_waypoints, 1);
wptsX = zeros(num_waypoints, 3);  % [pos vel acc]
wptsY = zeros(num_waypoints, 3);
wptsZ = zeros(num_waypoints, 3);

% Assign positions
wptsX(:,1) = ned_waypoints(:,2);  % East coordinates
wptsY(:,1) = ned_waypoints(:,1);  % North coordinates
wptsZ(:,1) = ned_waypoints(:,3);  % Down coordinates

% Generate time vector first
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

% Initialize time vector as a row vector
time_vector = zeros(1, num_waypoints);
time_vector(1) = 0;  % Start time
time_vector(2) = 60;  % 60 seconds for takeoff
time_vector(end-1) = flight_time - 60;  % 60 seconds for landing
time_vector(end) = flight_time;  % End time

% Assign times for cruise waypoints based on segment lengths
current_time = time_vector(2);  % Start from end of takeoff
for i = 3:num_waypoints-2
    current_time = current_time + time_steps(i-2);
    time_vector(i) = current_time;
end

% Generate velocities with smoother transitions
% Takeoff phase
wptsX(1:2,2) = 0;
wptsY(1:2,2) = 0;
wptsZ(1,2) = 0;
wptsZ(2,2) = -takeoff_target_speed * 0.7;  % Start with 70% speed

% Cruise phase with smooth transitions
for i = 3:num_waypoints-1
    % Calculate direction vector using previous and next points for smoother path
    prev_dir = [wptsX(i,1) - wptsX(i-1,1), wptsY(i,1) - wptsY(i-1,1)];
    next_dir = [wptsX(i+1,1) - wptsX(i,1), wptsY(i+1,1) - wptsY(i,1)];
    
    if norm(prev_dir) > 0 && norm(next_dir) > 0
        prev_dir = prev_dir / norm(prev_dir);
        next_dir = next_dir / norm(next_dir);
        
        % Average the directions for smoother transition
        direction = (prev_dir + next_dir) / 2;
        direction = direction / norm(direction);
        
        % Calculate speed factor based on turn angle
        turn_angle = acos(dot(prev_dir, next_dir));
        speed_factor = cos(turn_angle/2)^2;  % Moderate speed reduction in turns
        
        % Set velocity components with smooth transition
        wptsX(i,2) = direction(1) * cruise_speed * speed_factor;
        wptsY(i,2) = direction(2) * cruise_speed * speed_factor;
        wptsZ(i,2) = 0;  % No vertical velocity during cruise
    else
        wptsX(i,2) = 0;
        wptsY(i,2) = 0;
        wptsZ(i,2) = 0;
    end
end

% Landing phase with smooth transitions
wptsX(end-1:end,2) = 0;
wptsY(end-1:end,2) = 0;
wptsZ(end-1,2) = landing_target_speed * 0.7;  % Start with 70% speed
wptsZ(end,2) = 0;

% Generate accelerations with smoother transitions
dt = diff(time_vector);
for i = 1:num_waypoints
    if i == 1
        % Forward difference with smoothing
        wptsX(i,3) = (wptsX(i+1,2) - wptsX(i,2)) / dt(i) * 0.7;
        wptsY(i,3) = (wptsY(i+1,2) - wptsY(i,2)) / dt(i) * 0.7;
        wptsZ(i,3) = (wptsZ(i+1,2) - wptsZ(i,2)) / dt(i) * 0.7;
    elseif i == num_waypoints
        % Backward difference with smoothing
        wptsX(i,3) = (wptsX(i,2) - wptsX(i-1,2)) / dt(i-1) * 0.7;
        wptsY(i,3) = (wptsY(i,2) - wptsY(i-1,2)) / dt(i-1) * 0.7;
        wptsZ(i,3) = (wptsZ(i,2) - wptsZ(i-1,2)) / dt(i-1) * 0.7;
    else
        % Central difference with smoothing
        dt_avg = (dt(i-1) + dt(i)) / 2;
        wptsX(i,3) = (wptsX(i+1,2) - wptsX(i-1,2)) / dt_avg * 0.7;
        wptsY(i,3) = (wptsY(i+1,2) - wptsY(i-1,2)) / dt_avg * 0.7;
        wptsZ(i,3) = (wptsZ(i+1,2) - wptsZ(i-1,2)) / dt_avg * 0.7;
    end
end

% Create 3D plot of generated waypoints
figure;
plot3(wptsX(:,1), wptsY(:,1), wptsZ(:,1), 'r-o', 'LineWidth', 2);
grid on;
xlabel('East (ft)');
ylabel('North (ft)');
zlabel('Altitude (ft)');
title('Generated Waypoints in NED Frame');
view(45, 30); % Set view angle for better 3D visualization
saveas(gcf, 'generated_waypoints_plot.png');

% Ensure time vectors are row vectors
time_wptsX = time_vector;
time_wptsY = time_vector;
time_wptsZ = time_vector;

% Create the structure and save into the trajectory file
pwcurve.waypoints = {wptsX, wptsY, wptsZ};
pwcurve.time_wpts = {time_vector, time_vector, time_vector};
save('./NED_PW_Bezier_Traj','pwcurve','-v7.3');
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
writetable(waypoint_data, 'waypoint_data.csv');

% Specify the file just created in userStruct for simSetup.m to load
userStruct.trajFile = './NED_PW_Bezier_Traj.mat';

% Plot the sample PW Bezier curve that was created (see visualization of trajectory and derivatives)
Plot_PW_Bezier_SaveFigures;

% Initialize the sim
simSetup;

% Run the simulation
sim(model);

% Open the simulation
open(model);

% Save the simulation results
simPlots_GUAM_save;

% Close the simulation
close_system(model);