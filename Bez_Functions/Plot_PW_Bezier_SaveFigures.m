% Plot_PW_Bezier_SaveFigures.m saves the ownship top-level trajectory 
% information plots as PNG files instead of displaying them.

% Written by: Michael J. Acheson, michael.j.acheson@nasa.gov
% NASA Langley Research Center (LaRC), 
% Dynamics Systems and Control Branch (D-316)

% Versions:
% 4.18.2025: Modified to save figures as PNG files instead of displaying.

% *************************************************************************
% Plots current trajectory
close all
p_int = 0.01;
waypoints = {wptsX, wptsY, wptsZ};
time_wpts = {time_wptsX, time_wptsY, time_wptsZ};
time = linspace(0,time_wptsX(end), time_wptsX(end)/p_int);
pwcurve_plot = genPWCurve(waypoints,time_wpts);

% Plot positions with time
values = evalPWCurve(pwcurve_plot,time,0);
figure(1);
plot3(values(:,1), values(:,2), values(:,3))
title('3D Desired trajectory');
xlabel('X');
ylabel('Y');
zlabel('Z');
grid on;
values2 = evalPWCurve(pwcurve_plot,time_wptsX ,0);
hold on;
plot3(values2(:,1), values2(:,2), values2(:,3),'rx');

% Add trajectory time at beginning of each PW segment position
for loop = 1:length(time_wptsX)
    text(values2(loop,1), values2(loop,2), values2(loop,3),sprintf('%0.1f',round(time_wptsX(loop))));
end
hold off;
saveas(gcf, '3D_Desired_Trajectory.png');

% Plot positions with time
figure(2);
plot(time, values);
title('Positions');
xlabel('Time (sec)');
ylabel('Position (ft)');
grid on;
legend('X', 'Y', 'Z');
saveas(gcf, 'Positions.png');

% Plot velocities with time
values = evalPWCurve(pwcurve_plot,time,1);
figure(3);
plot(time, values)
title('Velocities')
xlabel('Time (sec)');
ylabel('Velocity (ft/sec)');
grid on;
legend('X','Y','Z')
saveas(gcf, 'Velocities.png');

% Plot accelerations with time
values = evalPWCurve(pwcurve_plot,time,2);
figure(4);
plot(time, values)
title('Accelerations')
xlabel('Time (sec)');
ylabel('Acc (ft/sec^2)');
grid on;
legend('X','Y','Z')
saveas(gcf, 'Accelerations.png');