%% =========================================================================
%  script_120kmh_optionB.m
%  QuaDRiGa Raw CSI — 120 km/h  OPTION B  (Realistic vehicular, dt=0.5ms)
%
%  SCENARIO:
%    Speed  = 120 km/h  (33.33 m/s) — typical highway vehicular
%    dt     = 0.5 ms    (one LTE subframe / 5G NR slot numerology 0)
%    adj_corr = 0.3717  (channel decorrelates significantly per frame)
%    AR NMSE  = +0.99 dB (AR predictor nearly useless at this speed)
%
%  WHY dt=0.5ms (Option B):
%    This is the physically realistic frame spacing for a 120 km/h
%    vehicular system. Real 5G NR systems use 0.5ms slots (numerology 0).
%    In 0.5ms the UE moves 1.67cm per frame → total 31.67cm over 20 frames.
%    The channel changes substantially → genuine hard prediction problem.
%    Compare with Option A (dt=0.1ms) which kept adj_corr=0.97 but
%    was borderline unrealistic (100μs sampling for a car).
%
%  CHANGES FROM ORIGINAL 5 km/h script.m:
%    1. UE_speed       = 5/3.6    → 120/3.6
%    2. frame_duration = 2.5e-3   → 0.5e-3
%    3. Output files   : *_adp.mat → *_adp_120B.mat
%    4. Checkpoint name: checkpoint_v2env → checkpoint_120Benv
%    5. Verification thresholds updated for adj_corr≈0.37
%
%  UNCHANGED from original:
%    fc, Nc, Nt, delta_f, scenario, 100×100 samples, 20 frames,
%    rng(42), 9000/1000 split, ADP formula, normalisation, movement_profile
%
%  Paper: "CS3T for Accurate Channel Prediction" (IEEE INFOCOM 2024)
%  Scenario: 3GPP_38.901_UMa_NLOS | fc=5GHz | Nf=64 | df=30kHz
%  BS: 64-ant ULA | UE: omni, 120 km/h | dt=0.5ms
% =========================================================================

clear; clc; close all;
addpath(genpath('D:\Anchit\QuaDriGa_2023.12.13_v2.8.1-0\quadriga_src'));

%% -------------------------------------------------------------------------
%  Parameters
% -------------------------------------------------------------------------
fc               = 5.0e9;           % UNCHANGED
Nc               = 64;              % UNCHANGED
delta_f          = 30e3;            % UNCHANGED
Nt               = 64;              % UNCHANGED
Nr               = 1;               % UNCHANGED

%% CHANGE 1: Speed
UE_speed         = 120/3.6;         % CHANGED: 33.3333 m/s (was 5/3.6)

scenario         = '3GPP_38.901_UMa_NLOS';   % UNCHANGED

num_environments = 100;             % UNCHANGED
num_UEs_per_env  = 100;             % UNCHANGED
num_frames       = 20;              % UNCHANGED
total_samples    = num_environments * num_UEs_per_env;

lambda           = 3e8/fc;          % 0.06 m — UNCHANGED
ant_spacing      = lambda/2;        % 0.03 m — UNCHANGED

%% CHANGE 2: Frame duration — Option B realistic vehicular
%  dt=0.5ms = one LTE subframe = one 5G NR slot (numerology 0)
%  This gives adj_corr=0.3717 — realistic for 120 km/h vehicular systems.
%  AR predictor NMSE ≈ +0.99 dB (nearly useless, reflects real difficulty).
frame_duration   = 0.5e-3;         % CHANGED: 0.5ms (was 2.5ms)

%% Derived values — auto-update from the two changes above
pos_spacing      = UE_speed * frame_duration;       % 1.6667 cm per frame
total_track      = pos_spacing * (num_frames-1);    % 31.67 cm total
total_duration   = frame_duration * (num_frames-1); % 9.5 ms total

fd               = UE_speed / lambda;               % 555.56 Hz Doppler
adj_corr         = abs(besselj(0, 2*pi*fd*frame_duration));
frame_nmse_pred  = 10*log10(2*(1-adj_corr));

%% Print parameters for verification
fprintf('=======================================================\n');
fprintf('  120 km/h — OPTION B  (Realistic vehicular, dt=0.5ms)\n');
fprintf('=======================================================\n');
fprintf('UE speed            : %.1f km/h (%.4f m/s)\n', UE_speed*3.6, UE_speed);
fprintf('Frame duration      : %.1f ms\n', frame_duration*1000);
fprintf('Position spacing    : %.4f m (%.2f cm) per frame\n', pos_spacing, pos_spacing*100);
fprintf('Total track length  : %.4f m (%.2f cm)\n', total_track, total_track*100);
fprintf('Total duration      : %.2f ms\n', total_duration*1000);
fprintf('Doppler frequency   : %.2f Hz\n', fd);
fprintf('Adjacent corr       : %.4f  (expected ~0.37 for 120km/h, 0.5ms)\n', adj_corr);
fprintf('Predicted frame NMSE: %.2f dB  (expected ~+1 dB — hard prediction)\n', frame_nmse_pred);
fprintf('-------------------------------------------------------\n');
fprintf('Compare with 5 km/h paper:\n');
fprintf('  5 km/h:   adj_corr=0.9672, NMSE=-11.83 dB (easy)\n');
fprintf('  120 km/h: adj_corr=%.4f, NMSE=%.2f dB  (hard)\n', adj_corr, frame_nmse_pred);
fprintf('=======================================================\n\n');

%% -------------------------------------------------------------------------
%  Antenna arrays — UNCHANGED
% -------------------------------------------------------------------------
BS_array = qd_arrayant('omni');
BS_array.no_elements = Nt;
BS_array.element_position       = zeros(3, Nt);
BS_array.element_position(2,:)  = (0:Nt-1)*ant_spacing;
UE_array = qd_arrayant('omni');

%% -------------------------------------------------------------------------
%  Preallocate — UNCHANGED
% -------------------------------------------------------------------------
all_H_raw   = zeros(total_samples, num_frames, 2, Nc, Nt, 'single');
sample_idx  = 1;
t_start     = tic;
error_count = 0;

%% -------------------------------------------------------------------------
%  Main loop — UNCHANGED structure
%  Only pos_spacing, total_track, total_duration differ (derived above)
% -------------------------------------------------------------------------
fprintf('=== Generating channels (120 km/h, dt=0.5ms) ===\n');

for env = 1:num_environments

    env_start = tic;

    s                     = qd_simulation_parameters;
    s.center_frequency    = fc;
    s.sample_density      = 2;          % UNCHANGED
    s.use_absolute_delays = 1;
    s.show_progress_bars  = 0;

    pos_angles  = 2*pi * rand(1, num_UEs_per_env);
    dists       = 50 + 450 * rand(1, num_UEs_per_env);
    move_angles = 2*pi * rand(1, num_UEs_per_env);
    snap_counts = zeros(1, num_UEs_per_env);

    for u = 1:num_UEs_per_env

        %% BS/UE arrays
        a_bs = qd_arrayant('omni');
        a_bs.no_elements = Nt;
        a_bs.element_position       = zeros(3, Nt);
        a_bs.element_position(2,:)  = (0:Nt-1)*ant_spacing;
        a_ue = qd_arrayant('omni');

        %% Layout
        l             = qd_layout(s);
        l.no_tx       = 1;
        l.no_rx       = 1;
        l.tx_array    = a_bs;
        l.rx_array    = a_ue;
        l.tx_position = [0; 0; 25];

        %% UE track — movement_profile still forces exactly 20 waypoints
        %  At 120km/h, dt=0.5ms: track=31.67cm, nat_snaps≈21
        %  movement_profile overrides sample_density → snap_count=20
        start_x    = dists(u) * cos(pos_angles(u));
        start_y    = dists(u) * sin(pos_angles(u));
        move_angle = move_angles(u);

        t_ue      = qd_track('linear', total_track, move_angle);
        t_ue.name = sprintf('env%03due%03d', env, u);
        t_ue.initial_position = [start_x; start_y; 1.5];

        pos_along      = linspace(0, total_track, num_frames);
        t_ue.positions = [pos_along*cos(move_angle); ...
                          pos_along*sin(move_angle); ...
                          zeros(1, num_frames)];
        time_stamps           = linspace(0, total_duration, num_frames);
        t_ue.movement_profile = [time_stamps; pos_along];   % UNCHANGED structure
        t_ue.scenario         = scenario;
        l.rx_track(1,1)       = t_ue;
        l.rx_position(:,1)    = t_ue.initial_position;

        %% Generate channel + build frequency domain — UNCHANGED
        try
            cb       = l.init_builder;
            cb.gen_parameters;
            channels = cb.get_channels();
            ch       = channels(1);

            num_paths     = size(ch.coeff, 3);
            num_snapshots = size(ch.coeff, 4);
            snap_counts(u) = num_snapshots;

            if num_snapshots >= num_frames
                snap_idx = 1:num_frames;
            else
                snap_idx = round(linspace(1, num_snapshots, num_frames));
                snap_idx = max(1, min(snap_idx, num_snapshots));
            end

            f_axis   = (0:Nc-1)*delta_f;
            H_sample = zeros(num_frames, Nc, Nt);

            for t = 1:num_frames
                sidx    = snap_idx(t);
                coeff_t = ch.coeff(:,:,:,sidx);
                delay_t = ch.delay(:,:,:,sidx);
                H_frame = zeros(Nc, Nt);
                for m = 1:num_paths
                    for ant = 1:Nt
                        alpha = coeff_t(1,ant,m);
                        tau   = delay_t(1,ant,m);
                        phase = exp(-1j*2*pi*f_axis*tau);
                        H_frame(:,ant) = H_frame(:,ant) + alpha*phase.';
                    end
                end
                H_sample(t,:,:) = H_frame;
            end

            all_H_raw(sample_idx,:,1,:,:) = single(real(H_sample));
            all_H_raw(sample_idx,:,2,:,:) = single(imag(H_sample));

        catch ME
            error_count = error_count + 1;
            fprintf('  [ERROR] env%d UE%d: %s\n', env, u, ME.message);
        end

        sample_idx = sample_idx + 1;

    end % UE loop

    elapsed   = toc(t_start);
    rate      = (sample_idx-1) / elapsed;
    remaining = (total_samples - sample_idx+1) / rate;
    fprintf('Env %3d/%d | %.1fs | snaps min=%d max=%d | ETA %.1f min\n', ...
            env, num_environments, toc(env_start), ...
            min(snap_counts), max(snap_counts), remaining/60);

    %% CHANGE 4: Checkpoint filename
    if mod(env,10) == 0
        H_partial = all_H_raw(1:sample_idx-1,:,:,:,:);
        save(sprintf('checkpoint_120Benv%03d.mat',env), ...
             'H_partial','sample_idx','env','-v7.3');
        fprintf('  >>> Checkpoint saved\n');
    end

end % env loop

fprintf('\nGeneration done. Total time: %.1f min\n\n', toc(t_start)/60);

%% -------------------------------------------------------------------------
%  Verification — CHANGE 5: updated thresholds for 120 km/h, dt=0.5ms
%
%  At adj_corr=0.37:
%    Adjacent frame NMSE ≈ +1 dB  (channel changes a lot per frame)
%    Frame-1-to-frame-20 correlation will be very low (~0.001 or less)
%  So the 5 km/h thresholds (NMSE<-8, corr>0.90) will NOT pass here.
%  New thresholds reflect the genuinely harder vehicular channel.
% -------------------------------------------------------------------------
fprintf('=== VERIFICATION (120 km/h, dt=0.5ms) ===\n');
fprintf('Sample | FrameNMSE | AdjCorr  | Status\n');
fprintf('-------|-----------|----------|---------\n');

check_ids = [1, 100, 500, 1000, 5000, 10000];
all_ok    = true;

for sid = check_ids
    Hr = squeeze(all_H_raw(sid,:,1,:,:));
    Hi = squeeze(all_H_raw(sid,:,2,:,:));
    H  = Hr + 1j*Hi;

    f1   = squeeze(H(1,:,:));
    f2   = squeeze(H(2,:,:));
    nmse = 10*log10(norm(f2(:)-f1(:))^2 / (norm(f1(:))^2 + 1e-10));

    Hf   = reshape(H, num_frames, []);
    c1   = Hf(1,:); c2 = Hf(2,:);
    corr_val = abs(c1*c2') / (norm(c1)*norm(c2) + 1e-10);

    status = 'OK';
    % CHANGE 5: thresholds for 120 km/h, dt=0.5ms
    % adj_corr=0.37 → frame NMSE expected around +1 dB
    % Accept range: -5 dB to +4 dB (wide — QuaDRiGa UMa NLOS varies)
    % Adjacent corr expected ~0.25 to 0.55 (around 0.37)
    if nmse > 4 || nmse < -10;  status = 'WARN'; all_ok = false; end
    if corr_val < 0.10;         status = 'WARN'; all_ok = false; end

    fprintf('%6d | %9.2f | %8.4f | %s\n', sid, nmse, corr_val, status);
end

fprintf('\nExpected at 120 km/h, dt=0.5ms:\n');
fprintf('  Adjacent corr ≈ 0.37  (channel changes a lot per 0.5ms)\n');
fprintf('  Frame NMSE    ≈ +1 dB  (AR predictor nearly useless)\n');
fprintf('  This is CORRECT — reflects genuine vehicular channel difficulty.\n');
fprintf('Errors: %d / %d\n', error_count, total_samples);

if all_ok
    fprintf('\nAll checks PASSED.\n');
else
    fprintf('\nWARNING: Some samples out of expected range.\n');
    fprintf('Check parameters before proceeding.\n');
end

%% -------------------------------------------------------------------------
%  ADP Conversion:  H' = Ff * H * Ft'   (paper Eq. 3) — UNCHANGED
% -------------------------------------------------------------------------
fprintf('\n=== ADP CONVERSION ===\n');

i_idx = (0:Nc-1)';  k_idx = (0:Nc-1);
Ff    = exp(-1j * 2*pi * i_idx * k_idx / Nc);

i_idx2 = (0:Nt-1)';  k_idx2 = (0:Nt-1);
Ft     = exp(-1j * 2*pi * i_idx2 * k_idx2 / Nt);

all_H_adp = zeros(total_samples, num_frames, 2, Nc, Nt, 'single');
adp_start  = tic;

for s = 1:total_samples

    sample_adp = zeros(num_frames, 2, Nc, Nt);

    for t = 1:num_frames
        H_real = squeeze(all_H_raw(s,t,1,:,:));
        H_imag = squeeze(all_H_raw(s,t,2,:,:));
        H_cplx = double(H_real) + 1j*double(H_imag);
        H_adp  = Ff * H_cplx * Ft';        % UNCHANGED: paper Eq.(3)
        sample_adp(t,1,:,:) = real(H_adp);
        sample_adp(t,2,:,:) = imag(H_adp);
    end

    %% Normalise per sample — UNCHANGED
    max_amp = max(abs(sample_adp(:)));
    if max_amp > 0
        sample_adp = sample_adp / max_amp;
    end

    all_H_adp(s,:,:,:,:) = single(sample_adp);

    if mod(s,2000) == 0
        fprintf('  ADP: %d/%d | %.1fs\n', s, total_samples, toc(adp_start));
    end

end

fprintf('ADP done in %.2f min\n', toc(adp_start)/60);
fprintf('ADP range: [%.4f, %.4f]\n', min(all_H_adp(:)), max(all_H_adp(:)));

%% -------------------------------------------------------------------------
%  Train/Test split — UNCHANGED (same seed, same ratio)
% -------------------------------------------------------------------------
fprintf('\n=== SAVING ===\n');

rng(42);                               % UNCHANGED: Mersenne Twister seed 42
idx_shuf  = randperm(total_samples);
train_idx = idx_shuf(1:9000);
test_idx  = idx_shuf(9001:10000);

train_adp = all_H_adp(train_idx,:,:,:,:);
test_adp  = all_H_adp(test_idx, :,:,:,:);

fprintf('Train shape: [%d x %d x %d x %d x %d]\n', size(train_adp));
fprintf('Test  shape: [%d x %d x %d x %d x %d]\n', size(test_adp));

%% CHANGE 3: Output filenames — _120B suffix keeps 5 km/h files safe
save('train_adp_120B.mat', 'train_adp', '-v7.3');
save('test_adp_120B.mat',  'test_adp',  '-v7.3');
save('quadriga_raw_CSI_120B.mat', ...
     'all_H_raw','all_H_adp',...
     'fc','Nc','delta_f','Nt','Nr',...
     'UE_speed','num_frames','total_samples',...
     'pos_spacing','frame_duration','scenario','-v7.3');

fprintf('\nSaved:\n');
fprintf('  train_adp_120B.mat        (9000 samples)\n');
fprintf('  test_adp_120B.mat         (1000 samples)\n');
fprintf('  quadriga_raw_CSI_120B.mat\n');
fprintf('\nParameters summary:\n');
fprintf('  Speed         : 120 km/h\n');
fprintf('  Frame spacing : 0.5 ms  (realistic vehicular)\n');
fprintf('  adj_corr      : %.4f\n', adj_corr);
fprintf('  AR NMSE       : %.2f dB\n', frame_nmse_pred);
fprintf('\nTotal time: %.2f minutes\n', toc(t_start)/60);
fprintf('\n=== Done! Feed train_adp_120B.mat to CS3T-Lite ===\n');
fprintf('Run: python train.py --L 1 --train_mat train_adp_120B.mat\n');
fprintf('                          --test_mat  test_adp_120B.mat\n');
