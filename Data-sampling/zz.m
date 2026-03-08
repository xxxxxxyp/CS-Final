clear; clc;

% --- Signal Generation ---
fs = 48e3;
volume = 0.8;  
center_freq =17e3;  
zclen = 480*2;  
bandwidth = 6e3;
zc_l = floor(bandwidth/(fs/zclen)) + 1; 
% Ensure zc_l is odd for symmetry if needed, but keeping your logic:
if mod(zc_l, 2) == 0, zc_l = zc_l + 1; end 

zc_u = 1; 
wind = hamming(zc_l);  
zcseq_base = zadoff_chu(zc_l, zc_u); 
zc_fft_base = fftshift(fft(zcseq_base)).' .* wind; 

% Map to Frequency Domain
ofdm = zeros(zclen, 1);
start_idx = floor(center_freq/fs*zclen);
freq_range = (start_idx - floor(zc_l/2)) : (start_idx + floor(zc_l/2));
ofdm(freq_range + 1) = zc_fft_base; % +1 for MATLAB indexing

% Hermetian Symmetry for Real Time Signal
ofdm_full = zeros(zclen, 1);
ofdm_full(2:zclen/2) = ofdm(2:zclen/2);
ofdm_full(zclen/2+2:end) = conj(flipud(ofdm(2:zclen/2)));

zcseq_time = real(ifft(ofdm_full));
zcseq_time = zcseq_time ./ max(abs(zcseq_time)) * volume;

% --- Playback & Capture Setup ---
play_duration = 2000; 
dataout = repmat(zcseq_time, play_duration*fs/zclen, 1);  
% dataout=[zeros(size(dataout)),dataout]; %%右声道

dataout=[dataout,zeros(size(dataout))]; %%左声道


samplesPerFrame = zclen;
deviceReader = audioDeviceReader('SampleRate', fs, 'NumChannels', 1, 'SamplesPerFrame', samplesPerFrame); 
player = audioplayer(dataout, fs);

% --- Visualization & Storage ---
figure('Position', [100, 100, 1000, 600]);
win_len=200;
hImg = imagesc(zeros(win_len, 100)); 
colormap('jet'); colorbar; axis xy;
title('Real-time CIR Monitor (Zadoff-Chu)');
hText = text(0.05, 0.95, 'Initializing...', 'Color', 'white', 'FontSize', 14, 'Units', 'normalized');

max_frames = 4000; 
all_cir_data = zeros(zclen, max_frames); 
sum_cir = zeros(zclen, 1);
lock_shift = [];
frameCount = 0;
validFrameCount = 0;

play(player); 
calib_buffer = zeros(zclen, 20); % 新增：专门存前20帧的原始数据
try
    fprintf('Running... Press Ctrl+C to stop and save.\n');
    while true
        if ~isplaying(player), play(player); end
        
        audioFrame = deviceReader(); 


        % Channel Estimation (Frequency Domain Cross-Correlation)
        recv_fft = fft(audioFrame);
        % Extract the band of interest and multiply by conjugate of sent signal
        est_fft = zeros(zclen, 1);
        est_fft(freq_range + 1) = recv_fft(freq_range + 1) .* conj(zc_fft_base);
        
        % Transform back to time domain to get CIR
        cir_raw = (ifft(est_fft));

        % --- Calibration Phase (First 20 Frames) ---
        if isempty(lock_shift)
            if frameCount < 20
                sum_cir = sum_cir + cir_raw;
                frameCount = frameCount + 1;
                calib_buffer(:,frameCount)=cir_raw;
                set(hText, 'String', sprintf('Calibrating... %d/20', frameCount), 'Color', [1, 0.5, 0]);
                continue; 
            else
                [~, p1] = max(sum_cir);
                target_pos = 1; % Align peak to index 50 for visibility
                lock_shift = target_pos - p1;
                fprintf('Calibration Complete. Peak at: %d. Alignment Shift: %d\n', p1, lock_shift);
                set(hText, 'String', 'Recording...', 'Color', 'r', 'FontSize', 30);
            end
        end
        
        % --- Recording Phase ---
        if ~isempty(lock_shift)
            cir_aligned = circshift(cir_raw, lock_shift);
            
            validFrameCount = validFrameCount + 1;
            if validFrameCount <= max_frames
                all_cir_data(:, validFrameCount) = cir_aligned;
            end
            
            % Update Display
            if mod(validFrameCount, 5) == 0
                cir_buffer = all_cir_data(:, max(1, validFrameCount-99):validFrameCount);
                % Pad buffer if not yet 100 frames
                if size(cir_buffer, 2) < 100
                    cir_buffer = [zeros(zclen, 100-size(cir_buffer,2)), cir_buffer];
                end
                set(hImg, 'CData', (abs(cir_buffer(1:win_len,:)))+1e-6);
                set(hText, 'String', sprintf('Recording Frame: %d', validFrameCount));
                drawnow limitrate;
            end
        end
    end
    
catch ME
    if ~strcmp(ME.identifier, 'MATLAB:interrupt')
        fprintf('\nError: %s\n', ME.message);
    end
end

% --- Cleanup & Save ---
stop(player);
release(deviceReader);
% all_cir_data = all_cir_data(:, 1:validFrameCount);
% save('recorded_cir_data.mat', 'final_data', 'fs');
% fprintf('\nData saved to recorded_cir_data.mat. Total frames: %d\n', validFrameCount);
