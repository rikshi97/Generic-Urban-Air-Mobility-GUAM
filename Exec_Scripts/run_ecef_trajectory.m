%% ECEF Trajectory Simulation Setup
% This script runs the GUAM simulation using an ECEF trajectory converted to
% Bezier format for the Lift+Cruise model

try
    % Clear workspace and close figures
    clear all;
    close all;
    
    % Get the root directory
    root_dir = fileparts(fileparts(mfilename('fullpath')));
    
    % First run setupPath to get basic paths set up
    run(fullfile(root_dir, 'setupPath.m'));
    
    % Add specific required paths
    addpath(fullfile(root_dir, 'Bez_Functions'));
    addpath(fullfile(root_dir, 'vehicles', 'Lift+Cruise'));
    addpath(fullfile(root_dir, 'vehicles', 'Lift+Cruise', 'Utils'));
    addpath(fullfile(root_dir, 'vehicles', 'Lift+Cruise', 'setup'));
    addpath(fullfile(root_dir, 'ClassDef'));
    addpath(fullfile(root_dir, 'vehicles', 'Lift+Cruise', 'ClassDef'));
    
    % Check if ECEF trajectory exists, if not generate it
    ecef_traj_file = fullfile(root_dir, 'ecef_trajectory.mat');
    if ~exist(ecef_traj_file, 'file')
        fprintf('ECEF trajectory file not found. Generating from waypoints...\n');
        wp_file = fullfile(root_dir, 'wp_BN.csv');
        if ~exist(wp_file, 'file')
            error('Waypoint file wp_BN.csv not found in root directory');
        end
        generate_ecef_trajectory(wp_file, 2400);
    end
    
    % Load the ECEF trajectory
    fprintf('Loading ECEF trajectory...\n');
    load(ecef_traj_file);
    
    % Convert ECEF trajectory to GUAM format
    fprintf('Converting trajectory to GUAM format...\n');
    target = convert_ecef_to_guam(pwcurve);
    
    % Extract waypoints for plotting
    wptsX = target.RefInput.Bezier.waypoints{1};
    wptsY = target.RefInput.Bezier.waypoints{2};
    wptsZ = target.RefInput.Bezier.waypoints{3};
    time_wptsX = target.RefInput.Bezier.time_wpts{1};
    time_wptsY = target.RefInput.Bezier.time_wpts{2};
    time_wptsZ = target.RefInput.Bezier.time_wpts{3};
    
    % Set up simulation parameters
    fprintf('Setting up simulation parameters...\n');
    model = 'GUAM';
    
    % Set simulation solver parameters
    userStruct = struct();
    userStruct.variants = struct();
    userStruct.variants.refInputType = 4; % Piecewise Bezier
    
    % Set model variants using enumeration values
    userStruct.variants.fmType = double(ForceMomentEnum.Polynomial); % 2
    userStruct.variants.atmosType = double(AtmosphereEnum.US_STD_ATMOS_76); % 1
    userStruct.variants.turbType = double(TurbulenceEnum.None); % 1
    userStruct.variants.ctrlType = double(CtrlEnum.BASELINE); % 2
    userStruct.variants.actType = double(ActuatorEnum.FirstOrder); % 1
    userStruct.variants.propType = double(PropulsionEnum.None); % 1
    userStruct.variants.eomType = double(EOMEnum.Simple); % 1
    userStruct.variants.sensorType = double(SensorsEnum.None); % 1
    
    % Initialize the simulation
    fprintf('Initializing simulation...\n');
    simSetup;
    
    % Load and configure the model
    load_system(model);
    
    % Use ode45 solver for continuous states
    set_param(model, 'Solver', 'ode45');
    set_param(model, 'MaxStep', '0.005');  % Maximum step size
    set_param(model, 'RelTol', '1e-3');    % Relative tolerance
    set_param(model, 'AbsTol', '1e-6');    % Absolute tolerance
    set_param(model, 'StartTime', '0');
    set_param(model, 'StopTime', num2str(max([time_wptsX(end), time_wptsY(end), time_wptsZ(end)])));
    
    % Plot the trajectory for verification
    fprintf('Plotting trajectory for verification...\n');
    Plot_PW_Bezier;
    
    try
        % Run the simulation
        fprintf('Running simulation...\n');
        simOut = sim(model);
        
        % Plot simulation results
        fprintf('Plotting simulation results...\n');
        simPlots_GUAM(simOut);
        
        % Animate the results
        fprintf('Generating animation...\n');
        Animate_SimOut(simOut, 'ecef_trajectory_animation.mp4');
        
        fprintf('Simulation completed successfully!\n');
    catch ME
        fprintf('Error occurred: %s\n', ME.message);
        fprintf('In file: %s, Line: %d\n', ME.stack(1).file, ME.stack(1).line);
        close_system(model, 0);  % Close without saving
        rethrow(ME);
    end
    
catch ME
    fprintf('Error occurred: %s\n', ME.message);
    fprintf('In file: %s, Line: %d\n', ME.stack(1).file, ME.stack(1).line);
    rethrow(ME);
end 