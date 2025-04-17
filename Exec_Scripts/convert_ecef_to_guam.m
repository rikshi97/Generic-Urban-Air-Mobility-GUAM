function [target] = convert_ecef_to_guam(pwcurve)
% CONVERT_ECEF_TO_GUAM - Converts ECEF trajectory to GUAM format
%
% Inputs:
%   pwcurve - ECEF trajectory structure with waypoints and time points
%
% Outputs:
%   target - GUAM-compatible trajectory structure

% Extract trajectory data
wptsX = pwcurve.waypoints{1};
wptsY = pwcurve.waypoints{2};
wptsZ = pwcurve.waypoints{3};
time_wptsX = pwcurve.time_wpts{1};
time_wptsY = pwcurve.time_wpts{2};
time_wptsZ = pwcurve.time_wpts{3};

% Create target structure with Bezier fields
target = struct();
target.RefInput = struct();
target.RefInput.Bezier = struct();
target.RefInput.Bezier.waypoints = {wptsX, wptsY, wptsZ};
target.RefInput.Bezier.time_wpts = {time_wptsX, time_wptsY, time_wptsZ};

% Set initial conditions
target.RefInput.Vel_bIc_des = [wptsX(1,2); wptsY(1,2); wptsZ(1,2)]; % Initial velocity
target.RefInput.pos_des = [wptsX(1,1); wptsY(1,1); wptsZ(1,1)]; % Initial position
target.RefInput.chi_des = 0; % Initial heading
target.RefInput.chi_dot_des = 0; % Initial heading rate

% Set trajectory time
target.RefInput.trajectory = struct();
target.RefInput.trajectory.refTime = [time_wptsX(1) time_wptsX(end)];

% Validate the trajectory
validate_trajectory(target);

end

function validate_trajectory(target)
% VALIDATE_TRAJECTORY - Validates the trajectory structure
%
% Inputs:
%   target - Trajectory structure to validate

% Extract time vectors
time_wptsX = target.RefInput.Bezier.time_wpts{1};
time_wptsY = target.RefInput.Bezier.time_wpts{2};
time_wptsZ = target.RefInput.Bezier.time_wpts{3};

% Check if all time vectors are monotonically increasing
if any(diff(time_wptsX) <= 0) || any(diff(time_wptsY) <= 0) || any(diff(time_wptsZ) <= 0)
    error('Time vectors must be monotonically increasing');
end

% Check if start and end times match across axes
if time_wptsX(1) ~= time_wptsY(1) || time_wptsX(1) ~= time_wptsZ(1)
    error('Start times must match across all axes');
end

if time_wptsX(end) ~= time_wptsY(end) || time_wptsX(end) ~= time_wptsZ(end)
    error('End times must match across all axes');
end

% Check if waypoints have correct dimensions
wptsX = target.RefInput.Bezier.waypoints{1};
wptsY = target.RefInput.Bezier.waypoints{2};
wptsZ = target.RefInput.Bezier.waypoints{3};

if size(wptsX, 2) ~= 3 || size(wptsY, 2) ~= 3 || size(wptsZ, 2) ~= 3
    error('Waypoints must have 3 columns: [position, velocity, acceleration]');
end

end 