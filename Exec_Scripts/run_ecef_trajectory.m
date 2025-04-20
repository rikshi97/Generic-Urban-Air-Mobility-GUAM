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
    addpath(fullfile(root_dir, 'utilities', 'variant_definitions'));
    
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
    
    % Set up simulation parameters
    fprintf('Setting up simulation parameters...\n');
    model = 'GUAM';
    
    % Initialize SimIn structure
    SimIn = struct();
    
    % Setup units
    SimIn.Units = struct();
    
    % Basic units
    SimIn.Units.m = 1;  % meters
    SimIn.Units.ft = 0.3048;  % feet to meters
    SimIn.Units.s = 1;  % seconds
    SimIn.Units.g0 = 9.81;  % gravity
    SimIn.Units.deg = pi/180;  % degrees to radians
    SimIn.Units.knot = 0.514444;  % knots to m/s
    SimIn.Units.rad = 1;  % radians
    
    % Temperature units
    SimIn.Units.K = 1;  % Kelvin
    SimIn.Units.degR = 5/9;  % Degrees Rankine to Kelvin conversion
    SimIn.Units.degC = 1;  % Celsius
    SimIn.Units.degF = 5/9;  % Fahrenheit to Celsius conversion
    
    % Force and mass units
    SimIn.Units.N = 1;  % Newton
    SimIn.Units.kg = 1;  % Kilogram
    SimIn.Units.lbf = 4.448222;  % Pound force
    SimIn.Units.slug = 14.5939;  % Slug
    
    % Pressure units
    SimIn.Units.Pa = 1;  % Pascal
    SimIn.Units.atm = 101325;  % Standard atmosphere
    SimIn.Units.psi = 6894.757;  % Pounds per square inch
    
    % Other base units
    SimIn.Units.mol = 1;  % Mole
    SimIn.Units.J = 1;  % Joule
    SimIn.Units.W = 1;  % Watt
    SimIn.Units.A = 1;  % Ampere
    SimIn.Units.V = 1;  % Volt
    SimIn.Units.Hz = 1;  % Hertz
    
    % Compound units
    SimIn.Units.mps = SimIn.Units.m / SimIn.Units.s;  % meters per second
    SimIn.Units.mps2 = SimIn.Units.mps / SimIn.Units.s;  % meters per second squared
    SimIn.Units.kgm2 = SimIn.Units.kg * SimIn.Units.m^2;  % kilogram meter squared
    SimIn.Units.Nm = SimIn.Units.N * SimIn.Units.m;  % Newton meter
    
    % Initialize model parameters
    SimIn.Model = struct();
    SimIn.Model.mass = 2000;  % Mass in kg
    SimIn.Model.I = diag([1000, 1000, 2000]);  % Inertia matrix
    SimIn.Model.b = 30;  % Wingspan in meters
    SimIn.Model.c = 3;  % Mean aerodynamic chord in meters
    SimIn.Model.S = 90;  % Wing area in square meters
    
    % Set simulation solver parameters
    userStruct = struct();
    userStruct.variants = struct();
    userStruct.variants.vehicleType = double(VehicleEnum.LiftPlusCruise);
    userStruct.variants.refInputType = 4; % Piecewise Bezier
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
    
    % Setup variant structure
    setupVariantStruct;
    
    % Setup types first
    SimIn = setupTypes(SimIn, userStruct.variants);
    
    % Setup environment
    SimIn.Environment = setupEnvironment(SimIn);
    
    % Setup vehicle
    SimIn = setup(SimIn, target);
    
    % Setup switches
    setupSwitches;
    
    % Setup parameters
    setupParameters(SimIn);
    
    % Setup variants
    setupVariants;
    
    % Setup buses
    setupBuses;
    
    % Load and configure the model
    fprintf('Loading model...\n');
    load_system(model);
    
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
        fprintf('Error occurred during simulation: %s\n', ME.message);
        fprintf('In file: %s, Line: %d\n', ME.stack(1).file, ME.stack(1).line);
        close_system(model, 0);  % Close without saving
        rethrow(ME);
    end
    
catch ME
    fprintf('Error occurred: %s\n', ME.message);
    fprintf('In file: %s, Line: %d\n', ME.stack(1).file, ME.stack(1).line);
    rethrow(ME);
end 