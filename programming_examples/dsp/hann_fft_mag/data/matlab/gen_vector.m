clear;
close all;

% Parameters
Nfft = 32;
backoff_dB = -15;
Nports = 1;
FF = fimath('RoundingMethod','Nearest','OverflowAction','Saturate');
TT = numerictype(1,16,15);
dnshift = 4;

% Parameters for the spectrogram
frameLength = 128; % Segment length
overlap = 32; % Overlap length
stride = floor(0.9 * overlap); % 90% overlap
num_window = floor((frameLength - overlap)/(overlap - stride)) + 1;
hannWindow_float = hann(overlap); % Hann window (floating-point)
scalingFactor = 50;
hannWindow = int32(hannWindow_float * scalingFactor);
if (1)
    % Input signal:
    data = load("data_('AM-SSB', -20).txt");
    % data = load("sig0_i.txt");
    realPart1 = data(:, 1);
    imagPart1 = data(:, 2);
    realPart2 = data(:, 3);
    imagPart2 = data(:, 4);

    complexData1 = realPart1 + 1i * imagPart1;
    complexData2 = realPart2 + 1i * imagPart2;

    sig_i = reshape([complexData1.'; complexData2.'], 1, []).';
    scale = 0.5^dnshift;
    % scale = 1;
    sig_i = fi(scale*sig_i,TT,FF);
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
    scale = 0.5^dnshift;
    % scale = 1;
    sig_i = fi(scale*sig_i,TT,FF);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Apply Gaussian filter with integer values
if (1)
    filter_size = 7;
    sigma = 1; % Standard deviation of the Gaussian filter
    x = -floor(filter_size/2):floor(filter_size/2);
    gaussian_filter = exp(-x.^2 / (2*sigma^2));
    % gaussian_filter_int = int32(gaussian_filter / sum(gaussian_filter) * 1000); % Normalize and convert to integer
    gaussian_filter_int = int32(gaussian_filter / sum(gaussian_filter) * 10); % Normalize and convert to integer
    
    % Apply filter to real and imaginary parts separately
    real_filtered = conv(real(sig_i), gaussian_filter_int, 'same');
    imag_filtered = conv(imag(sig_i), gaussian_filter_int, 'same');
    
    % Combine filtered real and imaginary parts back into a complex signal
    sig_i_filtered = real_filtered + 1i * imag_filtered;
    sig_i_filtered = fi(sig_i_filtered,TT,FF);
else
    sig_i_filtered = sig_i;
    sig_i_filtered = fi(sig_i_filtered,TT,FF);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Segmenting the signal
segmented_signal = [];
start_idx = 1;

while start_idx + frameLength - 1 <= length(sig_i_filtered)
    segment = sig_i_filtered(start_idx:start_idx + frameLength - 1);
    segments = buffer(double(segment), overlap, stride, 'nodelay');
    segmented_signal = [segmented_signal segments];
    start_idx = start_idx + frameLength; % Move to the next 128-sample segment without overlap
end

% Apply window and perform FFT
real_segments = real(segmented_signal) .* double(hannWindow);
imag_segments = imag(segmented_signal) .* double(hannWindow);

fft_seg = complex(real_segments, imag_segments);
fi_fft_seg = fi(fft_seg,TT,FF);
int_fft = fi_fft_seg.int;
int_sig_i = sig_i_filtered.int;

spectrogramMatrix = fft(double(fi_fft_seg),[],1);
spectrogramMatrix_fi = fi(spectrogramMatrix,TT,FF);

spectrogramMatrix_abs = spectrogramMatrix_fi.int;

spectrogramMatrix_dB = abs(double(spectrogramMatrix_abs)).^2;


% Plot the spectrogram
figure;
n = 10;
imagesc(spectrogramMatrix_dB(:,((n-1)*num_window)+1:((n-1)*num_window)+num_window));
% imagesc(spectrogramMatrix_dB)
axis xy;
title('Spectrogram of Combined Complex Data');
xlabel('Time');
ylabel('Frequency');
colorbar;

% Save to PLIO files (if needed):
 % [~,~,~] = mkdir('data');
  for pp = 1 : Nports
 %    % % Write input files:
     % data_i = reshape(int_sig_i,1,[]);
     % fid_i = fopen(sprintf('data/sig%d_i_gaussian.txt',pp-1),'w');
     % xx_re_1 = real(data_i);
     % xx_im_1 = imag(data_i);
     % for ii = 1 : 2 : length(xx_re_1)
     %   fprintf(fid_i,'%d %d ',xx_re_1(ii  ),xx_im_1(ii  ));
     %   fprintf(fid_i,'%d %d\n',xx_re_1(ii+1),xx_im_1(ii+1));
     % end
     % fclose(fid_i);
       
      % Write output fft files:
      % data_o = reshape(spectrogramMatrix_abs,1,[]);
      % fid_o = fopen(sprintf('data/output_fft_gaussian.txt',pp-1),'w');
      % xx_re = real(data_o(pp:Nports:end));
      % xx_im = imag(data_o(pp:Nports:end));
      % for ii = 1 : 2 : length(xx_re)
      %   fprintf(fid_o,'%d %d ',xx_re(ii  ),xx_im(ii  ));
      %   fprintf(fid_o,'%d %d\n',xx_re(ii+1),xx_im(ii+1));
      % end
      % fclose(fid_o);
      %  disp(sprintf('Done writing PLIO files for port %g',pp-1));
      
      %save abs in int 
      % data_o = reshape(spectrogramMatrix_dB,1,[]);
      % fid_o = fopen(sprintf('data/output_fft_gaussian.txt',pp-1),'w');
      % for ii = 1 : 4 : length(data_o)
      %   fprintf(fid_o,'%d %d ',data_o(ii  ),data_o(ii + 1));
      %   fprintf(fid_o,'%d %d\n',data_o(ii+2),data_o(ii+3));
      % end
      % fclose(fid_o);
      % disp(sprintf('Done writing PLIO files for port %g',pp-1));
  end
