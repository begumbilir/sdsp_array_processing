%% Classical Beamformer Estimating DOA with the first data

load('spcom_10sep.mat');

% Create Array Geometry
element_positions = (0:M-1) * Delta; % Positions of array elements (normalized)


% Estimate Sample Covariance Matrix
R_x = (X * X') / N; % Sample covariance matrix

theta_scan = -90:0.5:90; % Scanning angles for beam pattern

% Calculate Beamformer Output Power
beamformer_output = zeros(size(theta_scan));
for i = 1:length(theta_scan)
    theta_scan_rad = deg2rad(theta_scan(i));
    array_response_vector = exp(1i * 2 * pi * element_positions' * sin(theta_scan_rad)); %For uniform linear array
    beamformer_output(i) = abs(array_response_vector' * R_x * array_response_vector) / (array_response_vector' * array_response_vector);
end

% Find DOAs Estimate (Peak of the Beamformer Output)
[~, peak_indices] = findpeaks(beamformer_output, 'NPeaks', 2, 'SortStr', 'descend'); % Since there is two sources
estimated_thetas = theta_scan(peak_indices);

% Plotting
figure;
plot(theta_scan, 10*log10(beamformer_output));
xlabel('Angle (degrees)');
ylabel('Beamformer Output Power (dB)');
title('Classical Beamformer Output w/ 1st Dataset');
grid on;

% Display Estimated DOA
disp(['Estimated DOA: ', num2str(estimated_thetas), ' degrees']);

%% Classical Beamformer Estimating DOA with the second data

load('spcom_50sep.mat');

% Create Array Geometry
element_positions = (0:M-1) * Delta; % Positions of array elements (normalized)


% Estimate Sample Covariance Matrix
R_x = (X * X') / N; % Sample covariance matrix

theta_scan = -90:0.5:90; % Scanning angles for beam pattern

% Calculate Beamformer Output Power
beamformer_output2 = zeros(size(theta_scan));
for i = 1:length(theta_scan)
    theta_scan_rad = deg2rad(theta_scan(i));
    array_response_vector = exp(1i * 2 * pi * element_positions' * sin(theta_scan_rad)); %For uniform linear array
    beamformer_output2(i) = abs(array_response_vector' * R_x * array_response_vector) / (array_response_vector' * array_response_vector);
end

% Find DOA Estimate (Peak of the Beamformer Output)
[~, peak_indices] = findpeaks(beamformer_output2, 'NPeaks', 2, 'SortStr', 'descend'); % Since there is two sources
estimated_thetas = theta_scan(peak_indices);


% Plotting
figure;
plot(theta_scan, 10*log10(beamformer_output2));
xlabel('Angle (degrees)');
ylabel('Beamformer Output Power (dB)');
title('Classical Beamformer Output w/ 2nd Dataset');
grid on;

% Display Estimated DOA
disp(['Estimated DOA: ', num2str(estimated_thetas), ' degrees']);


figure;

% First subplot
subplot(2,1,1); % Create a 2-row, 1-column grid, and place this plot in the 1st position
plot(theta_scan, 10*log10(beamformer_output), 'LineWidth', 2);
xlabel('Angle (degrees)');
ylabel('Beamformer Output (dB)');
xlim([-90 90]); % Set y-axis range from -90 to 90
title('Classical Beamformer Output w/ 1st Dataset');
grid on;


% Second subplot
subplot(2,1,2); % Place this plot in the 2nd position
plot(theta_scan, 10*log10(beamformer_output2), 'LineWidth', 2); % Use a different beamformer output if needed
xlabel('Angle (degrees)');
ylabel('Beamformer Output (dB)');
title('Classical Beamformer Output w/ 2nd Dataset');
grid on;
xlim([-90 90]); % Set y-axis range from -90 to 90

%% MVDR Estimating DOA with the first data

load('spcom_10sep.mat');

% Create Array Geometry
element_positions = (0:M-1) * Delta; % Positions of array elements (normalized)


% Estimate Sample Covariance Matrix
R_x = (X * X') / N; % Sample covariance matrix

theta_scan = -90:0.5:90; % Scanning angles for beam pattern

% Calculate Beamformer Output Power
mvdr_output = zeros(size(theta_scan));
for i = 1:length(theta_scan)
    theta_scan_rad = deg2rad(theta_scan(i));
    array_response_vector = exp(1i * 2 * pi * element_positions' * sin(theta_scan_rad)); %For uniform linear array
    R_x_inv = inv(R_x); % Inverse of covariance matrix
    mvdr_output(i) = 1 / abs(array_response_vector' * R_x_inv * array_response_vector);
end

% Find DOAs Estimate (Peak of the Beamformer Output)
[~, peak_indices] = findpeaks(mvdr_output, 'NPeaks', 2, 'SortStr', 'descend'); % Since there is two sources
estimated_thetas = theta_scan(peak_indices);

% Plotting
figure;
plot(theta_scan, 10*log10(mvdr_output));
xlabel('Angle (degrees)');
ylabel('MVDR Output Power (dB)');
title('MVDR Beamformer DOA Estimation w/ Data 1');
grid on;

% Display Estimated DOA
disp(['Estimated DOAs: ', num2str(estimated_thetas), ' degrees']);

%% MVDR Estimating DOA with the second data

load('spcom_50sep.mat');

% Create Array Geometry
element_positions = (0:M-1) * Delta; % Positions of array elements (normalized)


% Estimate Sample Covariance Matrix
R_x = (X * X') / N; % Sample covariance matrix

theta_scan = -90:0.5:90; % Scanning angles for beam pattern

% Calculate Beamformer Output Power
mvdr_output2 = zeros(size(theta_scan));
for i = 1:length(theta_scan)
    theta_scan_rad = deg2rad(theta_scan(i));
    array_response_vector = exp(1i * 2 * pi * element_positions' * sin(theta_scan_rad)); %For uniform linear array
    R_x_inv = inv(R_x); % Inverse of covariance matrix
    mvdr_output2(i) = 1 / abs(array_response_vector' * R_x_inv * array_response_vector);
end

% Find DOAs Estimate (Peak of the Beamformer Output)
[~, peak_indices] = findpeaks(mvdr_output2, 'NPeaks', 2, 'SortStr', 'descend'); % Since there is two sources
estimated_thetas = theta_scan(peak_indices);

% Plotting
figure;
plot(theta_scan, 10*log10(mvdr_output2));
xlabel('Angle (degrees)');
ylabel('MVDR Output Power (dB)');
title('MVDR Beamformer DOA Estimation w/ Data 2');
grid on;

% Display Estimated DOA
disp(['Estimated DOAs: ', num2str(estimated_thetas), ' degrees']);


figure;

% First subplot
subplot(2,1,1); % Create a 2-row, 1-column grid, and place this plot in the 1st position
plot(theta_scan, 10*log10(mvdr_output), 'LineWidth', 2);
xlabel('Angle (degrees)');
ylabel('Beamformer Output (dB)');
xlim([-90 90]); % Set y-axis range from -90 to 90
title('MVDR Beamformer Output w/ 1st Dataset');
grid on;


% Second subplot
subplot(2,1,2); % Place this plot in the 2nd position
plot(theta_scan, 10*log10(mvdr_output2), 'LineWidth', 2); % Use a different beamformer output if needed
xlabel('Angle (degrees)');
ylabel('Beamformer Output (dB)');
title('MVDR Beamformer Output w/ 2nd Dataset');
grid on;
xlim([-90 90]); % Set y-axis range from -90 to 90

%% Wiener Receiver Estimating DOA with the first data

% In Wiener Receiver, the aim is to minimize the output power w^h R_x w

load('spcom_10sep.mat');

% Create Array Geometry
element_positions = (0:M-1) * Delta; % Positions of array elements (normalized)


% Estimate Sample Covariance Matrix
R_x = (X * X') / N; % Sample covariance matrix

theta_scan = -90:0.5:90; % Scanning angles for beam pattern

% Calculate Beamformer Output Power
wiener_output = zeros(size(theta_scan));
for i = 1:length(theta_scan)
    theta_scan_rad = deg2rad(theta_scan(i));
    array_response_vector = exp(1i * 2 * pi * element_positions' * sin(theta_scan_rad)); %For uniform linear array
    R_x_inv = inv(R_x); % Inverse of covariance matrix
    wiener_output(i) = abs(array_response_vector' * R_x_inv * array_response_vector);
end

% Find DOAs Estimate (Lowest Peaks of the Beamformer Output)
%to get the minimum two global points, the negative of the signal is taken
[~, peak_indices] = findpeaks(-wiener_output, 'NPeaks', 2, 'SortStr', 'descend'); % Since there is two sources
estimated_thetas = theta_scan(peak_indices);

% Plotting
figure;
plot(theta_scan, 10*log10(wiener_output));
xlabel('Angle (degrees)');
ylabel('Wiener Receiver Output Power (dB)');
title('Wiener Receiver DoA Estimation w/ 1st Dataset');
grid on;

% Display Estimated DOA
disp(['Estimated DOAs: ', num2str(estimated_thetas), ' degrees']);

%% Wiener Receiver Estimating DOA with the second data

load('spcom_50sep.mat');

% Create Array Geometry
element_positions = (0:M-1) * Delta; % Positions of array elements (normalized)


% Estimate Sample Covariance Matrix
R_x = (X * X') / N; % Sample covariance matrix

theta_scan = -90:0.5:90; % Scanning angles for beam pattern

% Calculate Beamformer Output Power
wiener_output2 = zeros(size(theta_scan));
for i = 1:length(theta_scan)
    theta_scan_rad = deg2rad(theta_scan(i));
    array_response_vector = exp(1i * 2 * pi * element_positions' * sin(theta_scan_rad)); %For uniform linear array
    R_x_inv = inv(R_x); % Inverse of covariance matrix
    wiener_output2(i) = abs(array_response_vector' * R_x_inv * array_response_vector);
end

% Find DOAs Estimate (Lowest Peaks of the Beamformer Output)
%to get the minimum two global points, the negative of the signal is taken
[~, peak_indices] = findpeaks(-wiener_output2, 'NPeaks', 2, 'SortStr', 'descend'); % Since there is two sources
estimated_thetas = theta_scan(peak_indices);

% Plotting
figure;
plot(theta_scan, 10*log10(wiener_output2));
xlabel('Angle (degrees)');
ylabel('Wiener Receiver Output Power (dB)');
title('Wiener Receiver DoA Estimation w/ 2nd Dataset');
grid on;

% Display Estimated DOA
disp(['Estimated DOAs: ', num2str(estimated_thetas), ' degrees']);



figure;

% First subplot
subplot(2,1,1); % Create a 2-row, 1-column grid, and place this plot in the 1st position
plot(theta_scan, 10*log10(wiener_output), 'LineWidth', 2);
xlabel('Angle (degrees)');
ylabel('Wiener Receiver Output (dB)');
xlim([-90 90]); % Set y-axis range from -90 to 90
title('Wiener Receiver Output w/ 1st Dataset');
grid on;


% Second subplot
subplot(2,1,2); % Place this plot in the 2nd position
plot(theta_scan, 10*log10(wiener_output2), 'LineWidth', 2); % Use a different beamformer output if needed
xlabel('Angle (degrees)');
ylabel('Wiener Receiver Output (dB)');
title('Wiener Receiver Output w/ 2nd Dataset');
grid on;
xlim([-90 90]); % Set y-axis range from -90 to 90

%% MUSIC Estimating DOA with the first data

load('spcom_10sep.mat');
num_sources = 2; %known

% Create Array Geometry
element_positions = (0:M-1) * Delta; % Positions of array elements (normalized)


% Estimate Sample Covariance Matrix
R_x = (X * X') / N; % Sample covariance matrix


% Perform eigenvalue decomposition
[eigenvectors, eigenvalues] = eig(R_x);
eigenvalues = diag(eigenvalues);  % Get the diagonal elements (eigenvalues)

% Sort eigenvalues in ascending order
[eigenvalues_sorted, idx] = sort(eigenvalues, 'descend');
eigenvectors_sorted = eigenvectors(:, idx);

% Select noise subspace (last M-2 eigenvectors correspond to noise)
En = eigenvectors_sorted(:, num_sources+1:end);  % Noise subspace

theta_scan = -90:0.5:90; % Scanning angles for beam pattern
music_output = zeros(size(theta_scan));  % Initialize MUSIC spectrum

% Compute MUSIC spectrum
for i = 1:length(theta_scan)
    theta_scan_rad = deg2rad(theta_scan(i));
    array_response_vector = exp(1i * 2 * pi * element_positions' * sind(theta_scan(i)));
    music_output(i) = 1 / abs(array_response_vector' * (En * En') * array_response_vector);  % Pseudo-spectrum
end

% Find the peaks (DoA estimates)
[~, peak_indices] = findpeaks(music_output, 'NPeaks', 2, 'SortStr', 'descend'); % Since there is two sources
estimated_thetas = theta_scan(peak_indices);

% Plot the MUSIC spectrum
figure;
plot(theta_scan, 10*log10(music_output));
xlabel('Angle (degrees)');
ylabel('Spatial Spectrum (dB)');
title('MUSIC Spectrum');
grid on;


% Display Estimated DOA
disp(['Estimated DOAs: ', num2str(estimated_thetas), ' degrees']);

%% MUSIC Estimating DOA with the second data

load('spcom_50sep.mat');

% Create Array Geometry
element_positions = (0:M-1) * Delta; % Positions of array elements (normalized)


% Estimate Sample Covariance Matrix
R_x = (X * X') / N; % Sample covariance matrix

theta_scan = -90:0.5:90; % Scanning angles for beam pattern

% Perform eigenvalue decomposition
[eigenvectors, eigenvalues] = eig(R_x);
eigenvalues = diag(eigenvalues);  % Get the diagonal elements (eigenvalues)

% Sort eigenvalues in ascending order
[eigenvalues_sorted, idx] = sort(eigenvalues, 'descend');
eigenvectors_sorted = eigenvectors(:, idx);

% Select noise subspace (last M-2 eigenvectors correspond to noise)
En = eigenvectors_sorted(:, num_sources+1:end);  % Noise subspace

theta_scan = -90:0.5:90; % Scanning angles for beam pattern
music_output2 = zeros(size(theta_scan));  % Initialize MUSIC spectrum

% Compute MUSIC spectrum
for i = 1:length(theta_scan)
    theta_scan_rad = deg2rad(theta_scan(i));
    array_response_vector = exp(1i * 2 * pi * element_positions' * sind(theta_scan(i)));
    music_output2(i) = norm(array_response_vector)^2 / norm(array_response_vector' * En)^2;  % Pseudo-spectrum
end

% Plot the MUSIC spectrum
figure;
plot(theta_scan, 10*log10(music_output2));
xlabel('Angle (degrees)');
ylabel('Spatial Spectrum (dB)');
title('MUSIC Spectrum');
grid on;

% Find the peaks (DoA estimates)
[~, peak_indices] = findpeaks(music_output2, 'NPeaks', 2, 'SortStr', 'descend'); % Since there is two sources
estimated_thetas = theta_scan(peak_indices);


% Display Estimated DOA
disp(['Estimated DOAs: ', num2str(estimated_thetas), ' degrees']);


% First subplot
figure,
subplot(2,1,1); % Create a 2-row, 1-column grid, and place this plot in the 1st position
plot(theta_scan, 10*log10(music_output), 'LineWidth', 2);
xlabel('Angle (degrees)');
ylabel('MUSIC Output (dB)');
xlim([-90 90]); % Set y-axis range from -90 to 90
title('MUSIC Output w/ 1st Dataset');
grid on;


% Second subplot
subplot(2,1,2); % Place this plot in the 2nd position
plot(theta_scan, 10*log10(music_output2), 'LineWidth', 2); % Use a different beamformer output if needed
xlabel('Angle (degrees)');
ylabel('MUSIC Output (dB)');
title('MUSIC Output w/ 2nd Dataset');
grid on;
xlim([-90 90]); % Set y-axis range from -90 to 90

