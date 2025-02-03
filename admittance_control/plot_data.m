clc; clear; close all;

% --- Read CSV Data ---
filename = 'logged_data.csv';
data = readmatrix(filename);

% Extract columns
time_log = data(:, 1);  % Column 1: Time

% Raw force (Fx, Fy, Fz) - Columns 2-4
raw_force_log = data(:, 2:4);

% Filtered force (Fx, Fy, Fz) - Columns 8-10
filtered_force_log = data(:, 8:10);

% Offset (position) (offset_Fx, offset_Fy, offset_Fz) - Columns 14-16
offset_log = data(:, 14:16);

% Target position (target_Fx, target_Fy, target_Fz) - Columns 20-22
target_pose_log = data(:, 20:22);

% Reference position (ref_Fx, ref_Fy, ref_Fz) - Columns 32-34
reference_pose_log = data(:, 32:34);

% Convert position data from meters to millimeters
offset_log_mm = offset_log * 1000;
target_pose_log_mm = target_pose_log * 1000;
reference_pose_log_mm = reference_pose_log * 1000;

% --- Define Labels ---
force_labels = {'F_x', 'F_y', 'F_z'};
axis_labels = {'X', 'Y', 'Z'};

% --- Set Figure Properties ---
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 4.13, 2.4]); % Compact figure size
tiledlayout(3, 2, 'TileSpacing', 'tight', 'Padding', 'tight'); % Minimize gaps

% === LEFT COLUMN: Force Plots (Vertically Arranged) ===
% Force X
ax1 = nexttile(1);
plot(time_log, raw_force_log(:, 1), 'k-', 'LineWidth', 1.2);
ylabel('$F_x$ (N)', 'Interpreter', 'latex', 'FontSize', 9);
grid on;
set(gca, 'XTickLabel', []); % Hide X labels for upper plots

% Force Y
ax2 = nexttile(3);
plot(time_log, raw_force_log(:, 2), 'k-', 'LineWidth', 1.2);
ylabel('$F_y$ (N)', 'Interpreter', 'latex', 'FontSize', 9);
grid on;
set(gca, 'XTickLabel', []); % Hide X labels for upper plots

% Force Z
ax3 = nexttile(5);
plot(time_log, raw_force_log(:, 3), 'k-', 'LineWidth', 1.2);
ylabel('$F_z$ (N)', 'Interpreter', 'latex', 'FontSize', 9);
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 9);
grid on;

% === RIGHT COLUMN: Position (Offset & Target) Plots (Vertically Arranged) ===
% Position X
ax4 = nexttile(2);
plot(time_log, offset_log_mm(:, 1), 'b-', 'LineWidth', 1.2, 'DisplayName', 'Offset X'); hold on;
plot(time_log, target_pose_log_mm(:, 1), 'r-', 'LineWidth', 1.2, 'DisplayName', 'Compliant X'); hold on;
plot(time_log, reference_pose_log_mm(:, 1), 'k--', 'LineWidth', 1.2, 'DisplayName', 'Reference X');
ylabel('$X$ (mm)', 'Interpreter', 'latex', 'FontSize', 9);
legend('Location', 'northeast', 'FontSize', 8, 'Interpreter', 'latex');
legend("boxoff");
grid on;
set(gca, 'XTickLabel', []); % Hide X labels for upper plots

% Position Y
ax5 = nexttile(4);
plot(time_log, offset_log_mm(:, 2), 'b-', 'LineWidth', 1.2, 'DisplayName', 'Offset Y'); hold on;
plot(time_log, target_pose_log_mm(:, 2), 'r-', 'LineWidth', 1.2, 'DisplayName', 'Compliant Y'); hold on;
plot(time_log, reference_pose_log_mm(:, 2), 'k--', 'LineWidth', 1.2, 'DisplayName', 'Reference Y');
ylabel('$Y$ (mm)', 'Interpreter', 'latex', 'FontSize', 9);
legend('Location', 'northeast', 'FontSize', 8, 'Interpreter', 'latex');
legend("boxoff");
grid on;
set(gca, 'XTickLabel', []); % Hide X labels for upper plots

% Position Z
ax6 = nexttile(6);
plot(time_log, offset_log_mm(:, 3), 'b-', 'LineWidth', 1.2, 'DisplayName', 'Offset Z'); hold on;
plot(time_log, target_pose_log_mm(:, 3), 'r-', 'LineWidth', 1.2, 'DisplayName', 'Compliant Z'); hold on;
plot(time_log, reference_pose_log_mm(:, 3), 'k--', 'LineWidth', 1.2, 'DisplayName', 'Reference Z');
ylabel('$Z$ (mm)', 'Interpreter', 'latex', 'FontSize', 9);
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 9);
legend("boxoff");
legend('Location', 'northeast', 'FontSize', 8, 'Interpreter', 'latex');
grid on;

% --- Save Figure as High-Resolution PNG Without Margins ---
% exportgraphics(fig, 'VAC_for_gripper_insertion_v2.png', 'Resolution', 800, 'BackgroundColor', 'white');
