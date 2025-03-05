%% Sim parameters
model = 'GUAM';
userStruct.variants.refInputType = 4; % Use PW Bezier input
userStruct.variants.fmType = 1; % 1 = Aerodynamic model, 1 = s-function, 2 = polynomial
PW_Bezier_flag = 0;

clear pwcurve

%% Define LLA waypoints and convert to NED
% Example LLA waypoints (format: [lat_deg, lon_deg, alt_ft])
lla_waypoints = [ 
    37.42, -122.05, 0;      % Start point
    37.41, -122.16, 500;   % Intermediate point
    37.87, -122.27, 0;    % Final point
];

% Reference LLA (first row of lla_waypoints)
lla0 = lla_waypoints(1, :); 

% Convert LLA to NED using "flat" Earth approximation
ned_waypoints = zeros(size(lla_waypoints));
for i = 1:size(lla_waypoints, 1)
    % lla2ned syntax: lla2ned(current_lla, reference_lla, 'flat'/'ellipsoid')
    xyzNED = lla2ned(lla_waypoints(i, :), lla0, 'flat');
    ned_waypoints(i, :) = xyzNED;
end

%% Define PW Bezier parameters using NED waypoints
% Split into X/Y/Z components
wptsX = [ned_waypoints(:,1), zeros(size(ned_waypoints,1),2)]; % [pos, vel, acc]
wptsY = [ned_waypoints(:,2), zeros(size(ned_waypoints,1),2)];
wptsZ = [ned_waypoints(:,3), zeros(size(ned_waypoints,1),2)];

% Time vector (match number of waypoints)
time_wpts = [0, 50, 80]; % Adjust times as needed

%wptsX = [0 100 0; 2000 100 0; 3500 50 0]; % within row = pos vel acc, rows are waypoints
time_wptsX = [0 50 80];
%wptsY = [0 0 0; 0 0 0]; % within row = pos vel acc, rows are waypoints
time_wptsY = [0 50 80];
%wptsZ = [0 0 0; 0 0 0; 83.33 500/60 0]; % within row = pos vel acc, rows are waypoints, NOTE: NED frame -z is up...
time_wptsZ = [0 50 80];
% NOTE each axis is handled seperately and can have different number of rows (waypoints and times), 
% however start and stop times must be consistent across all three axes

% Store the PW Bezier trajectory in the target structure (used in RefInputs)
target.RefInput.Bezier.waypoints = {wptsX, wptsY, wptsZ};
target.RefInput.Bezier.time_wpts = {time_wptsX time_wptsY time_wptsZ};

% Set desired Initial condition vars
target.RefInput.Vel_bIc_des    = [100;0;0];
target.RefInput.pos_des        = zeros(3,1);
target.RefInput.chi_des        = 0;
target.RefInput.chi_dot_des    = 0;
target.RefInput.trajectory.refTime = [0 50 80];

% Plot the sample PW Bezier curve that was created (see visualization of trajectory and derivatives)
Plot_PW_Bezier;
clear wptsX wptsY wptsZ time_wptsX time_wptsY time_wptsZ 
userStruct.trajFile = ''; % Delete user specified PW Bezier file


%% Initialize and run simulation
simSetup;
open(model);