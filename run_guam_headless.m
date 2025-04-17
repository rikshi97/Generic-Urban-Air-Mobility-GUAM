%% GUAM Headless Simulation Runner - Simplified Version
% This script runs GUAM simulations in headless mode using built-in options

% Add necessary paths
addpath('./Exec_Scripts/');
addpath('./Bez_Functions/');

% Choose which example to run (programmatically instead of with input prompt)
% Options:
% 1: Sinusoidal Timeseries
% 2: Hover to Transition Timeseries  
% 3: Cruise Climbing Turn Timeseries
% 4: Ramp demo
% 5: Piecewise Bezier Trajectory
simulation_type = 2;  % Change this value to select a different simulation

% Create a log of what we're doing
fprintf('Running GUAM simulation in headless mode...\n');
fprintf('Selected simulation type: %d\n', simulation_type);

% Create the userStruct if it doesn't exist
if ~exist('userStruct', 'var')
    userStruct = struct();
end
if ~isfield(userStruct, 'variants')
    userStruct.variants = struct();
end

% Set up the simulation type based on user selection
model = 'GUAM';
switch simulation_type
    case 1
        fprintf('Running Sinusoidal Timeseries simulation\n');
        userStruct.variants.refInputType = 3; % Timeseries
        
        % Run setup code without executing the plotting commands
        time = [0 20 40]';
        pos = [0 0 0; 0 0 0; 0 0 0];
        vel_i = [0 0 0; 5 0 0; 0 5 1];
        chi = atan2(vel_i(:,2), vel_i(:,1));
        chid = gradient(chi)./gradient(time);
        q = QrotZ(chi);
        vel = Qtrans(q, vel_i);
        
        % Create timeseries
        RefInput.Vel_bIc_des = timeseries(vel, time);
        RefInput.pos_des = timeseries(pos, time);
        RefInput.chi_des = timeseries(chi, time);
        RefInput.chi_dot_des = timeseries(chid, time);
        RefInput.vel_des = timeseries(vel_i, time);
        target.RefInput = RefInput;
        
    case 2
        fprintf('Running Hover to Transition Timeseries simulation\n');
        userStruct.variants.refInputType = 3; % Timeseries
        
        time = [0 20 40]';
        pos = [0 0 0; 0 0 -80; 150 0 -100];
        vel_i = [0 0 -8; 0 0 0; 15 0 0];
        chi = atan2(vel_i(:,2), vel_i(:,1));
        chid = gradient(chi)./gradient(time);
        q = QrotZ(chi);
        vel = Qtrans(q, vel_i);
        
        RefInput.Vel_bIc_des = timeseries(vel, time);
        RefInput.pos_des = timeseries(pos, time);
        RefInput.chi_des = timeseries(chi, time);
        RefInput.chi_dot_des = timeseries(chid, time);
        RefInput.vel_des = timeseries(vel_i, time);
        target.RefInput = RefInput;
        
    case 3
        fprintf('Running Cruise Climbing Turn Timeseries simulation\n');
        userStruct.variants.refInputType = 3; % Timeseries
        
        time = [0 20 40]';
        pos = [0 0 -100; 100 0 -150; 170 60 -200];
        vel_i = [0 0 0; 10 0 -5; 7 7 -5];
        chi = atan2(vel_i(:,2), vel_i(:,1));
        chid = gradient(chi)./gradient(time);
        q = QrotZ(chi);
        vel = Qtrans(q, vel_i);
        
        RefInput.Vel_bIc_des = timeseries(vel, time);
        RefInput.pos_des = timeseries(pos, time);
        RefInput.chi_des = timeseries(chi, time);
        RefInput.chi_dot_des = timeseries(chid, time);
        RefInput.vel_des = timeseries(vel_i, time);
        target.RefInput = RefInput;
        
    case 4
        fprintf('Running Ramp demo simulation\n');
        userStruct.variants.refInputType = 1; % FOUR_RAMP
        target = struct('tas', 110, 'gndtrack', 0, 'stopTime', 40);
        
    case 5
        fprintf('Running Piecewise Bezier Trajectory simulation\n');
        userStruct.variants.refInputType = 4; % Piecewise Bezier
        
        % Simple hover to cruise bezier trajectory
        wptsX = [0 0 0; 150 15 0];
        wptsY = [0 0 0; 0 0 0];
        wptsZ = [0 0 0; -100 0 0];
        time_wpts = [0 40];
        
        pwcurve.waypoints = {wptsX, wptsY, wptsZ};
        pwcurve.time_wpts = {time_wpts, time_wpts, time_wpts};
        
        % Create a temporary trajectory file
        temp_traj_file = 'temp_bezier_traj.mat';
        save(temp_traj_file, 'pwcurve');
        userStruct.trajFile = temp_traj_file;
        
    otherwise
        error('Invalid simulation type selected. Choose 1-5.');
end

% Edit simSetup.m to use release mode
fprintf('Setting up simulation in release mode...\n');

% Setup the simulation using the patched setup
% First run setupPath to get basic paths set up
setupPath;

% Use the standard simSetup, but patched for headless operation
% Note: This part normally runs simSetup, but we'll handle it differently
setupVariantStruct;
SimIn = setupTypes(SimIn, userStruct.variants);
setupSwitches;

% Check for Bezier trajectory settings
if SimIn.refInputType == RefInputEnum.BEZIER
    if (exist('userStruct', 'var') && isfield(userStruct, 'trajFile'))
        SimIn.trajFile = userStruct.trajFile;
    end
end

% Set default units
SimIn.Units = setUnits('ft', 'slug');

% Handle various RefInput types
if SimIn.Switches.RefTrajOn
    % Code from simSetup.m for RefInput handling
    % (This is simplified compared to the original)
    % ...
end

% Use release mode in the setup function (this is the key change)
SimIn = setup(SimIn, target, true);  % Use release simulation mode

% Complete the remaining setup functions
setupParameters(SimIn);
setupVariants;
setupBuses;

% Configure Simulink for headless operation
fprintf('Configuring Simulink for headless operation...\n');
warning('off', 'Simulink:Commands:LoadingOlderModel');
load_system(model);
set_param(model, 'SimulationMode', 'rapid');
set_param(model, 'FastRestart', 'on');
set_param(model, 'EnableDisplays', 'off');
set_param(model, 'ScreenSize', 'current');

% Run the simulation without GUI interaction
fprintf('Executing simulation...\n');
options = simset('SrcWorkspace', 'current');
out = sim(model, options);

% Save the simulation results to a file instead of displaying plots
save('simulation_results.mat', 'out');
fprintf('Simulation complete. Results saved to simulation_results.mat\n');

% Run the plotting script but save to file instead of displaying
try
    % Create a figure but make it invisible
    fig = figure('Visible', 'off');
    
    % Run the standard plotting function
    simPlots_GUAM;
    
    % Save all generated figures to files
    all_figs = findall(0, 'Type', 'figure');
    for i = 1:length(all_figs)
        saveas(all_figs(i), sprintf('simulation_plot_%d.png', i));
    end
    
    fprintf('Generated %d plot files\n', length(all_figs));
    close all;
catch e
    fprintf('Error generating plots: %s\n', e.message);
end

% Clean up temporary files
if simulation_type == 5 && exist(temp_traj_file, 'file')
    delete(temp_traj_file);
end

% Close the model when done
close_system(model, 0);

fprintf('Headless simulation complete!\n'); 