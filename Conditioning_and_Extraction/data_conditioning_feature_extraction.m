% Data Conditioning + Feature Extraction --------------------------------
% Primary feature calculation: mel-frequency cepstral coefficients (MFCCs)
% MATLAB's mfcc function handles the computation of mel-frequency cepstral 
% coefficients given at least the signal input in the frequency domain and
% sampling frequency in Hertz (Hz). This file prioritizes 20 mfcc values
% from the first 100 non-noisy and noisy signals of each word of the 
% Kaggle dataset.

% For the purpose of this project:
% fs = sampling frequency, ~16 kHz
% L = window length
% L_overlap = overlap length, 50% ideal for hanning window
% duration = time duration of each keyword audio signal
% n_mfcc = number of mel-frequency cepstral coefficients
% n_fft = number of FFT points

% ------ Start of Script ------
clear;
clc;
close all;

% 20-25 mel-frequency cepstral coefficients are relevant for vowel emphasis
% and syllable pronunciation
n_mfcc = 20; % number of mel-frequency cepstral coefficients
L = 1024; % in samples
win = hann(L, "periodic"); % creates hanning window of length L
L_overlap = L/2; % in samples
mfcc_result = []; % mfcc matrix placeholder for entire folder
labels = []; % label placeholder

% reads noise and returns sampled data and sampling frequency
[noise, noise_fs] = audioread("_background_noise_/white_noise.wav");

folder = 'archive';
audio_files = dir(fullfile(folder, '**/*.wav')); % count audio files
for k=1:numel(audio_files)
    path = fullfile(audio_files(k).folder, audio_files(k).name);
    % reads audio signal and returns sampled data and sampling frequency
    [audioIn, fs] = audioread(path);
    % zero-padding the signal
    audioIn = add_noise(audioIn, noise); % add noise to signal
    audioIn = [audioIn; zeros(16000-size(audioIn, 1), 1)];
    % computes STFT which windows and zero pads the frames of the signal  
    % given overlap length
    S = stft(audioIn, "Window", win, ...
        "OverlapLength", L_overlap, ...
        "FFTLength", L, ...
        "Centered", true);
    % generates coefficents from the signal of each file
    % size: 98x20
    coeffs = mfcc(audioIn, fs, "LogEnergy", "Ignore", "numCoeffs", n_mfcc);

    if k == 1
        figure;
    end

    if k < 5
        [~, word] = fileparts(audio_files(k).folder); % extract last folder
        word(1) = upper(word(1)); % capitalize folder name
        % shared title
        sgtitle("MFCCs of " + "'" + word + "'", "FontSize", 18);
        subplot(2, 2, k);
        % plotting mfcc heatmap 
        mfcc(audioIn, fs, "LogEnergy", "Ignore", "numCoeffs", n_mfcc);
        title(audio_files(k).name, ...
            "FontSize", 16, "Interpreter", "none"); % file title
    end

    if k == 101
        figure;
    end

    if 100 < k && k < 105
        [~, word] = fileparts(audio_files(k).folder); % extract last folder
        word(1) = upper(word(1)); % capitalize folder name
        % shared title
        sgtitle("MFCCs of " + "'" + word + "'", "FontSize", 18);
        subplot(2, 2, k-100);
        % plotting mfcc heatmap 
        mfcc(audioIn, fs, "LogEnergy", "Ignore", "numCoeffs", n_mfcc);
        title(audio_files(k).name, ...
            "FontSize", 16, "Interpreter", "none"); % file title
    end

    if k == 201
        figure;
    end

    if 200 < k && k < 205
        [~, word] = fileparts(audio_files(k).folder); % extract last folder
        word(1) = upper(word(1)); % capitalize folder name
        % shared title
        sgtitle("MFCCs of " + "'" + word + "'", "FontSize", 18);
        subplot(2, 2, k-200);
        % plotting mfcc heatmap 
        mfcc(audioIn, fs, "LogEnergy", "Ignore", "numCoeffs", n_mfcc);
        title(audio_files(k).name, ...
            "FontSize", 16, "Interpreter", "none"); % file title
    end

    if k == 301
        figure;
    end

    if 300 < k && k < 305
        [~, word] = fileparts(audio_files(k).folder); % extract last folder
        word(1) = upper(word(1)); % capitalize folder name
        % shared title
        sgtitle("MFCCs of " + "'" + word + "'", "FontSize", 18);
        subplot(2, 2, k-300);
        % plotting mfcc heatmap 
        mfcc(audioIn, fs, "LogEnergy", "Ignore", "numCoeffs", n_mfcc);
        title(audio_files(k).name, ...
            "FontSize", 16, "Interpreter", "none"); % file title
    end

    % flattens and appends new coeffs to resulting matrix of coeffs
    % final size per folder: (98*20)x(number of files in a folder)
    % final size per folder: (98*20)x(number of files in a folder*350)
    mfcc_result = [mfcc_result; coeffs(:).'];

    [~, word] = fileparts(audio_files(k).folder); % extract last folder
    labels = [labels; string(word)]; % appends folder label to label column
end

data = [num2cell(labels), num2cell(mfcc_result)]; % append mfccs to labels

% write data to .csv
% writecell(data, "coeffs_not_noisy.csv"); 
% writecell(data, "coeffs_doing_the_dishes.csv");
% writecell(data, "coeffs_dude_miaowing.csv");
% writecell(data, "coeffs_exercise_bike.csv");
% writecell(data, "coeffs_pink_noise.csv");
% writecell(data, "coeffs_running_tap.csv");
writecell(data, "coeffs_white_noise.csv");

function noisy_signal = add_noise(signal, noise)
    % set cap by finding minimum column length between the audio signal and
    % background noise
    cap = min(size(signal, 1), size(noise, 1));
    signal = signal(1:cap, :); % reduce size of audio signal to [CAP]X1
    noise = noise(1:cap, :); % reduce size of background noise to [CAP]X1
    noisy_signal = signal + noise; % mix the audio signal with the noise
end

% References ------------------------------------------------
% [1] "mfcc."[Online].
% Available:https://www.mathworks.com/help/audio/ref/mfcc.html
% [2] "stft." [Online].
% Available:https://www.mathworks.com/help/signal/ref/stft.html
% [3] "Speaker Verification Using i-vectors." [Online].
% Available:https://www.mathworks.com/help/audio/ug/speaker-verification-
% using-ivectors.html
% [4] "Siemens Digital Industries Software Community." [Online]. 
% Available:https://community.sw.siemens.com/s/article/Overlap-What-
% Why-and-How-to-use-it
% [5] Md. R. Hasan, Md. M. Hasan, and M. Z. Hossain, 
% "How many Mel‐frequency cepstral coefficients to be utilized in speech 
% recognition? A study with the Bengali language," The Journal of 
% Engineering, vol. 2021, no. 12, pp. 817–827, Dec. 2021, 
% doi: 10.1049/tje2.12082.
% [6] "fullfile." [Online].
% Available:https://www.mathworks.com/help/matlab/ref/fullfile.html
% [7] "dir." [Online].
% Available: https://www.mathworks.com/help/matlab/ref/dir.html
% [8] "fileparts." [Online].
% Available:https://www.mathworks.com/help/matlab/ref/fileparts.html
% [9] "upper." [Online].
% Available:https://www.mathworks.com/help/matlab/ref/string.upper.html
% [10] "writecell." [Online].
% Available:https://www.mathworks.com/help/matlab/ref/writecell.html
% [11] “Speech Commands Dataset v0.02.” Accessed: Sep. 28, 2026. [Online].
% Available: https://www.kaggle.com/datasets/yashdogra/speech-commands/data

% ------- End of Script -------




