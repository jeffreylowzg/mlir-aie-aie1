close all
clear
if (1)
    % Load the data
    data = load("data_('AM-SSB', -20).txt");
    
    % Extract the real and imaginary parts for the entire sequence
    realPart1 = data(:, 1);
    imagPart1 = data(:, 2);
    realPart2 = data(:, 3);
    imagPart2 = data(:, 4);
    
    % Create complex numbers from the sequence
    complexData1 = realPart1 + 1i * imagPart1;
    complexData2 = realPart2 + 1i * imagPart2;
    
    % Interleave the two sets of complex data
    sig_i = reshape([complexData1.'; complexData2.'], 1, []).';
else
    % Generate a chirp signal
    fs = 10000;         % Sampling frequency
    t = 0:1/fs:10-1/fs; % Time vector of 10 seconds
    f0 = 0;            % Start frequency
    f1 = 5000;         % End frequency
    chirp_signal_real = chirp(t, f0, t(end), f1, 'linear');
    
    % Generate the imaginary part (phase-shifted version of the real part)
    % Phase shift by 90 degrees (pi/2 radians)
    phase_shift = pi/2;
    chirp_signal_imag = chirp_signal_real .* sin(2 * pi * f1 * t + phase_shift);
    
    % Combine real and imaginary parts to form a complex chirp signal
    sig_i = chirp_signal_real(:) + 1i * chirp_signal_imag(:);
end

% Plot the original signal
figure;
subplot(2,1,1);
plot(real(sig_i));
title('Real Part of Original Chirp Signal');
xlabel('Sample Number');
ylabel('Amplitude');

subplot(2,1,2);
plot(imag(sig_i));
title('Imaginary Part of Original Chirp Signal');
xlabel('Sample Number');
ylabel('Amplitude');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Gaussian filter with integer values
filter_size = 7;
sigma = 1; % Standard deviation of the Gaussian filter
x = -floor(filter_size/2):floor(filter_size/2);
gaussian_filter = exp(-x.^2 / (2*sigma^2));
gaussian_filter = int32(gaussian_filter / sum(gaussian_filter) * 1000); % Normalize and convert to integer

% Apply filter to real and imaginary parts separately
real_filtered = conv(real(sig_i), gaussian_filter, 'same');
imag_filtered = conv(imag(sig_i), gaussian_filter, 'same');

% Combine filtered real and imaginary parts back into a complex signal
sig_i_filtered = real_filtered + 1i * imag_filtered;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Rest of the original MATLAB code
Nfft = 32;
Ntransform = 512;
backoff_dB = -15;                       % To achieve (-1,+1) range on AWGN signal
Nports = 1;

% Set quantization:
FF = fimath('RoundingMethod','Nearest','OverflowAction','Saturate');
TT = numerictype(1,16,0);

% Downshift (to ensure no clipping occurs in FFT):
dnshift = 3;

% Parameters for the spectrogram
frameLength = 32; % Frame length of 40 samples
overlap = floor(0.9 * frameLength); % 90% overlap
hannWindow_float = hann(frameLength); % Hann window (floating-point)

% Convert Hann window to integer
scalingFactor = 1000;
hannWindow = int32(hannWindow_float * scalingFactor);

% Compute the spectrogram
[s, w, t] = spectrogram(sig_i_filtered, double(hannWindow), overlap, frameLength, 'yaxis');

% Convert the spectrogram to dB scale
spectrogram_dB = 10 * log10(abs(s));

% Plot the spectrogram
figure;
imagesc(t, w, spectrogram_dB);
axis xy;
title('Spectrogram of Combined Complex Data');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters for the spectrogram
hopSize = frameLength - overlap; % Hop size

% Number of frames
numFrames = floor((length(sig_i) - overlap) / hopSize);

% Number of positive frequency components
numFreqComponents = frameLength;

% Initialize the spectrogram matrix
spectrogramMatrix = zeros(numFreqComponents, numFrames, 'int32');

% Compute the spectrogram
% for k = 1:numFrames
%     idx = (k-1) * hopSize + (1:frameLength);
%     segment = sig_i_filtered(idx) .* hannWindow;
%     fftSegment = fft(segment, frameLength);
%     spectrogramMatrix(:, k) = int32(abs(fftSegment(1:numFreqComponents)).^2); % Only take positive frequencies
% end
segments = buffer(sig_i_filtered, frameLength, overlap, 'nodelay');
real_segments = real(segments) .* double(hannWindow);
imag_segments = imag(segments) .* double(hannWindow);
fft_real_segments = fft(real_segments, frameLength);
fft_imag_segments = fft(imag_segments, frameLength);
spectrogramMatrix = int32(abs(fft_real_segments(1:numFreqComponents, :)).^2 + abs(fft_imag_segments(1:numFreqComponents, :)).^2); % Only take positive frequencies

% Convert to dB scale
spectrogramMatrix_dB = 10*log10(double(spectrogramMatrix)); % Conversion to double for log10

% Plot the spectrogram
figure;
imagesc(spectrogramMatrix_dB);
axis xy;
title('Spectrogram of Combined Complex Data');
xlabel('Time');
ylabel('Frequency');
colorbar;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract the first half of the spectrogram
halfSpectrogramMatrix_dB = spectrogramMatrix_dB(:, 1:floor(numFrames/2));

% Resize to 200x200
resizedSpectrogram = imresize(halfSpectrogramMatrix_dB, [200, 200]);

% Normalize the data to [0, 1]
normData = mat2gray(resizedSpectrogram);

% Convert the normalized data to an RGB image using a colormap
colormapData = parula(256);
rgbImage = ind2rgb(im2uint8(normData), colormapData);

% Save the RGB image as a PNG file
imwrite(rgbImage, 'heatmap.png');

% Turn into streams:
data_o = reshape(spectrogramMatrix_dB,1,[]);
data_i_combined = [real(sig_i_filtered), imag(sig_i_filtered)];

% Combine real and imaginary parts into a single matrix with 4 columns
A1 = data_i_combined(1:2:end, :);
A2 = data_i_combined(2:2:end, :);

data_i_reshaped = [A1, A2];

data_o_reshaped = reshape(data_o, 4, []).';

% Write data_i and data_o to text files
% writematrix(data_i_reshaped, 'output/sig_i.txt', 'Delimiter', ' ');
% writematrix(data_o_reshaped, 'output/sig_o.txt', 'Delimiter', ' ');