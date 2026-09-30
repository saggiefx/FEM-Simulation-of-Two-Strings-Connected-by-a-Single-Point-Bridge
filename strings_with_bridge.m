% Two strings connected by a bridge

clear;

live_plot = true;
play_audio = true;

% Physical parameters
% String 1 (Plucked)
L1 = 1.0;                   % String length [m]
mu1 = 1.2e-1;               % Mass per unit length [kg/m]

% String 2 (Sympathetic)
L2 = 3.0;                   % String length [m]
mu2 = 1.2e-3;               % Mass per unit length [kg/m]

S = 65;                     % Tension in string [N]

% Bridge parameters
bridge_stiffness = 15000;    % Coupling stiffness [N/m]
bridge_mass = 0.02;         % Bridge mass [kg]

% Discretization
n_elem_per_string = 200;  
le1 = L1 / n_elem_per_string;
le2 = L2 / n_elem_per_string;

% Node numbering: two strings do not share a node
n1 = n_elem_per_string + 1;       % Nodes in String 1
n2 = n_elem_per_string + 1;       % Nodes in String 2
n_nodes = n1 + n2;                % Total nodes

bridge_node_1 = n1;               % Right end of String 1
bridge_node_2 = n1 + 1;           % Left end of String 2

% Node locations
x1 = linspace(0, L1, n1).';
x2 = linspace(0, L2, n2).';

% Elementary matrices
Me1 = mu1 * le1 / 6 * [2 1; 1 2];
Ke1 = S / le1 * [1 -1; -1 1];

Me2 = mu2 * le2 / 6 * [2 1; 1 2];
Ke2 = S / le2 * [1 -1; -1 1];

% Initialize empty system matrices
M = sparse(n_nodes, n_nodes);
K = sparse(n_nodes, n_nodes);
C = sparse(n_nodes, n_nodes);

% System matrix assembly: String 1
for i = 1 : n_elem_per_string
    ind = [i, i+1];
    M(ind, ind) = M(ind, ind) + Me1;
    K(ind, ind) = K(ind, ind) + Ke1;
end

% System matrix assembly: String 2
offset = n1;
for i = 1 : n_elem_per_string
    ind = [offset + i, offset + i + 1];
    M(ind, ind) = M(ind, ind) + Me2;
    K(ind, ind) = K(ind, ind) + Ke2;
end

% Bridge coupling (modelled as a spring between the two string ends)
K([bridge_node_1, bridge_node_2], [bridge_node_1, bridge_node_2]) = ...
    K([bridge_node_1, bridge_node_2], [bridge_node_1, bridge_node_2]) + ...
    bridge_stiffness * [1, -1; -1, 1];

% Bridge mass
M(bridge_node_1, bridge_node_1) = M(bridge_node_1, bridge_node_1) + bridge_mass;
M(bridge_node_2, bridge_node_2) = M(bridge_node_2, bridge_node_2) + bridge_mass;

% Damping (Rayleigh damping: proportional to stiffness)
beta_damp = 1e-7;
C = beta_damp * K;

% Initial Gaussian shape (String 1 only)
x0_left = L1 / 5;               % Plucking position
U0 = 0.01;                      % Plucking amplitude
d = 0.08;                       % Pluck width

u0 = zeros(n_nodes, 1);
u0(1:n1) = U0 * exp(-(x1 - x0_left).^2 / (2*d^2));
u0(1) = 0;                      % Fixed left end
u0(end) = 0;                    % Fixed right end

% Partitioning (fixed ends)
c_dofs = [1, n_nodes];          % Constrained DOFs
u_dofs = 2 : n_nodes-1;         % Unconstrained DOFs

% Time stepping simulation (Newmark-beta)
beta = 1/4;
gamma = 1/2;
fs = 48000;
dt = 1/fs;
Nt = 96000;                     % Number of steps

M_star = M + dt*gamma*C + dt^2 * beta * K;
M_star22 = M_star(u_dofs, u_dofs);
K22 = K(u_dofs, u_dofs);
C22 = C(u_dofs, u_dofs);

u = u0;                         % Initial displacement
du = zeros(size(u0));           % Velocity
ddu = zeros(size(u0));          % Acceleration

% for live plot
if live_plot
    figure('Name', 'Live Simulation');
    subplot(2,1,1);
    h_line1 = plot(x1, u(1:n1), 'b', 'LineWidth', 2);
    xlabel('Position (m)'); ylabel('Displacement');
    title('String 1 (Plucked)');
    grid on; ylim(1.1*U0*[-1 1]);
    
    subplot(2,1,2);
    h_line2 = plot(x2, u(n1+1:end), 'r', 'LineWidth', 2);
    xlabel('Position (m)'); ylabel('Displacement');
    title('String 2 (Sympathetic)');
    grid on; ylim(1.1*U0*[-1 1]);
    drawnow;
end

% Audio signal at midpoints
if play_audio
    mid_idx_1 = round(n1 / 2);
    mid_idx_2 = round(n1 + n2 / 2);
    audio_signal1 = zeros(Nt, 1);
    audio_signal2 = zeros(Nt, 1);
end

fprintf('Running simulation (%d steps)...\n', Nt);
tic;

for it = 1 : Nt
    % Prediction
    tu = u + dt*du + dt^2/2*(1-2*beta) * ddu;
    tdu = du + dt*(1-gamma) * ddu;
    
    % Solution
    rhs = -K22 * tu(u_dofs) - C22 * tdu(u_dofs);
    ddu(u_dofs) = M_star22 \ rhs;
    
    % Correction
    u = tu + dt^2 * beta * ddu;
    du = tdu + dt * gamma * ddu;
    
    % Update live plot
    if live_plot && mod(it, 100) == 0
        if ishandle(h_line1) && ishandle(get(h_line1, 'Parent'))
            set(h_line1, 'YData', u(1:n1));
            set(h_line2, 'YData', u(n1+1:end));
            drawnow;
        else
            live_plot = false;
        end
    end
    
    % Record audio
    if play_audio
        audio_signal1(it) = du(mid_idx_1);
        audio_signal2(it) = du(mid_idx_2);
    end
end

toc;
fprintf('Simulation finished.\n');

% Post-processing
if play_audio
    % Frequency spectrum
    figure('Name', 'Frequency Spectrum (Log Scale)');
    
    NFFT = 2^nextpow2(Nt);
    f_axis = fs/2 * linspace(0, 1, NFFT/2+1);
    
    Y1 = fft(audio_signal1, NFFT);
    Y2 = fft(audio_signal2, NFFT);
    
    mag1 = 2 * abs(Y1(1:NFFT/2+1));
    mag2 = 2 * abs(Y2(1:NFFT/2+1));
    
    mag1 = mag1 / (max(mag1) + 1e-12);
    mag2 = mag2 / (max(mag2) + 1e-12);
    
    mag1_db = 20 * log10(mag1 + 1e-12);
    mag2_db = 20 * log10(mag2 + 1e-12);
    
    [~, idx1] = max(mag1(2:end));
    [~, idx2] = max(mag2(2:end));
    f_fund1 = f_axis(idx1+1);
    f_fund2 = f_axis(idx2+1);
    
    subplot(2,1,1);
    semilogx(f_axis, mag1_db, 'b', 'LineWidth', 1.5);
    xlabel('Frequency (Hz) - Log Scale');
    ylabel('Magnitude (dB)');
    title(sprintf('String 1 - Fundamental = %.2f Hz', f_fund1));
    xlim([20, 5000]); ylim([-80, 0]);
    grid on; grid minor;
    hold on;
    xline(f_fund1, '--k', sprintf(' %.2f Hz', f_fund1));
    hold off;
    xticks([20, 50, 100, 200, 500, 1000, 2000, 5000]);
    set(gca, 'XMinorTick', 'on');
    
    subplot(2,1,2);
    semilogx(f_axis, mag2_db, 'r', 'LineWidth', 1.5);
    xlabel('Frequency (Hz) - Log Scale');
    ylabel('Magnitude (dB)');
    title(sprintf('String 2 - Fundamental = %.2f Hz', f_fund2));
    xlim([20, 5000]); ylim([-80, 0]);
    grid on; grid minor;
    hold on;
    xline(f_fund2, '--k', sprintf(' %.2f Hz', f_fund2));
    hold off;
    xticks([20, 50, 100, 200, 500, 1000, 2000, 5000]);
    set(gca, 'XMinorTick', 'on');
    
    % Play audio
    audio_signal1 = audio_signal1 / (max(abs(audio_signal1)) + 1e-12) * 0.8;
    audio_signal2 = audio_signal2 / (max(abs(audio_signal2)) + 1e-12) * 0.8;
    
    fprintf('\n--- Playing Audio ---\n');
    fprintf('Playing String 1...\n');
    sound(audio_signal1, fs);
    pause(2.5);
    
    fprintf('Playing String 2...\n');
    sound(audio_signal2, fs);
    pause(2.5);
    
    fprintf('Playing both mixed...\n');
    sound((audio_signal1 + audio_signal2) / 2, fs);
    fprintf('Done.\n');
end

% Final displacement state
figure('Name', 'Final State');
subplot(2,1,1);
plot(x1, u(1:n1), 'b', 'LineWidth', 2);
xlabel('Position (m)'); ylabel('Displacement');
title('String 1 - Final');
grid on;

subplot(2,1,2);
plot(x2, u(n1+1:end), 'r', 'LineWidth', 2);
xlabel('Position (m)'); ylabel('Displacement');
title('String 2 - Final');
grid on;

fprintf('\nAll finished.\n');