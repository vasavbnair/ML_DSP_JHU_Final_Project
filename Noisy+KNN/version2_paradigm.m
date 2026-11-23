% Signal Processing ML Paradigm + KNN --------------------------------
% Primary feature calculation: mel-frequency cepstral coefficients (MFCCs)
% MATLAB's mfcc function handles the computation of the mean and standard 
% deviation of mel-frequency cepstral coefficients given at least the 
% signal input in the frequency domain and sampling frequency in Hertz 
% (Hz). This file prioritizes 20 mfcc values from 
% non-noisy and noisy versions of signals of 6 words of the Kaggle dataset.

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
mfcc_means = []; % mean mfcc array placeholder for entire folder
mfcc_std = []; % std mfcc array placeholder for entire folder
mfcc_summary = []; % mfcc array placeholder for entire folder

labels = []; % label placeholder

folder = 'archive';
audio_files = dir(fullfile(folder, '**/*.wav')); % count audio files
for k=1:numel(audio_files)
    path = fullfile(audio_files(k).folder, audio_files(k).name);
    % reads audio signal and returns sampled data and sampling frequency
    [audioIn, fs] = audioread(path);
    % zero-padding the signal
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
    mfcc_means = mean(coeffs, 1); % taking the mean of mfccs
    mfcc_std = std(coeffs); % taking the standard deviation of mfccs
    summary = [mfcc_means, mfcc_std]; % form summary matrix of mfccs
    mfcc_summary = [mfcc_summary; summary]; % stack summaries of mfccs

    [~, word] = fileparts(audio_files(k).folder); % extract last folder
    if (string(word) == 'marvin')
        labels = [labels; 1]; % appends target label to label column
    else
        labels = [labels; 0]; % appends non-target label to label column
    end
end

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
    mfcc_means = mean(coeffs, 1); % taking the mean of mfccs
    mfcc_std = std(coeffs); % taking the standard deviation of mfccs
    summary = [mfcc_means, mfcc_std]; % form summary matrix of mfccs
    mfcc_summary = [mfcc_summary; summary]; % stack summaries of mfccs

    [~, word] = fileparts(audio_files(k).folder); % extract last folder
    if (string(word) == 'marvin')
        labels = [labels; 1]; % appends target label to label column
    else
        labels = [labels; 0]; % appends non-target label to label column
    end
end

% Should revise this part since this is more hardcoded but currently
% suitable for testing.

% 1000 positive ("Marvin") and 1000 negative (non "Marvin") training data
x_training = [mfcc_summary(501:1000,:); mfcc_summary(1501:2000,:); ...
    mfcc_summary(2501:3000,:); mfcc_summary(3501:4000,:)];
y_training = [labels(501:1000); labels(1501:2000); ...
    labels(2501:3000); labels(3501:4000)];

% plotting mean vs standard deviation for each MFCC (1-4)
figure(1);
for c=1:4
    subplot(2, 2, c);
    gscatter(x_training(:,c), x_training(:,20+c), y_training, ...
        [], [], [], 'off')
    xlabel("Mean", "FontSize", 14); % x-label
    ylabel("Standard Deviation", "FontSize", 14); % y-label
    title("MFCC " + string(c), "FontSize", 16); % title
end
sgtitle("Sampled MFCC Training Data Given Target", "FontSize", 18); % title

% 300 positive ("Marvin") and 300 negative (non "Marvin") validation data
x_validation = [mfcc_summary(1:150,:); mfcc_summary(1001:1150,:); ...
    mfcc_summary(2001:2150,:); mfcc_summary(3001:3150,:)];
y_validation = [labels(1:150,:); labels(1001:1150,:); ...
    labels(2001:2150,:); labels(3001:3150,:)];

% 300 positive ("Marvin") and 300 negative (non "Marvin") testing data
x_testing = [mfcc_summary(151:300,:); mfcc_summary(1151:1300,:); ...
    mfcc_summary(2151:2300,:); mfcc_summary(3151:3300,:)];
y_testing = [labels(151:300,:); labels(1151:1300,:); ...
    labels(2151:2300,:); labels(3151:3300,:)];

% applying z-score standardization to training, validation, and testing
% datasets
[x_training_norm, mu, sigma] = zscore(x_training);
x_validation_norm = normalize(x_validation, "center", mu, "scale", sigma);
x_testing_norm = normalize(x_testing, "center", mu, "scale", sigma);

% plotting mean vs standard deviation for each MFCC (1-4) given z-score
% standardization
figure(2);
for c=1:4
    subplot(2, 2, c);
    gscatter(x_training_norm(:,c), x_training_norm(:,20+c), y_training, ...
        [], [], [], 'off')
    xlabel("Mean", "FontSize", 14); % x-label
    ylabel("Standard Deviation", "FontSize", 14); % y-label
    title("MFCC " + string(c), "FontSize", 16); % title
end
sgtitle("Sampled MFCC Training Data Given Target " + ...
    "and Z-Score Standardization", "FontSize", 18); % title

% fit k-nearest neighbor classifer with training dataset
mdl = fitcknn(x_training_norm, y_training, NumNeighbors=9);
% calculate loss with validation dataset
L = loss(mdl, x_validation_norm, y_validation);
% results -----------------
% NumNeighbors=1 | L=0.3300 | Accuracy=66.00%
% NumNeighbors=3 | L=0.2967 | Accuracy=70.33%
% NumNeighbors=5 | L=0.2667 | Accuracy=73.33%
% NumNeighbors=7 | L=0.2767 | Accuracy=72.33%
% NumNeighbors=9 | L=0.2633 | Accuracy=73.67%
% NumNeighbors=11 | L=0.2750 | Accuracy=72.50%
% NumNeighbors=13 | L=0.2700 | Accuracy=73.00%
% NumNeighbors=15 | L=0.2733 | Accuracy=72.67%
% NumNeighbors=17 | L=0.2867 | Accuracy=71.33%
% NumNeighbors=19 | L=0.2917 | Accuracy=70.83%
% NumNeighbors=21 | L=0.2817 | Accuracy=71.83%
% NumNeighbors=23 | L=0.2900 | Accuracy=71.00%
% NumNeighbors=25 | L=0.3050 | Accuracy=69.50%
% -------------------------
% predict new labels given validation set
ypred = predict(mdl,x_validation_norm);
figure(3);
cm = confusionchart(y_validation, ypred);
cm.Title = 'Prediction Statistics (KNN - Validation Data)';

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
% Available:
% https://www.mathworks.com/help/audio/ref/mfcc.html
% [2] "stft." [Online].
% Available:
% https://www.mathworks.com/help/signal/ref/stft.html
% [3] "Speaker Verification Using i-vectors." [Online].
% Available:
% https://www.mathworks.com/help/audio/ug/speaker-verification-
% using-ivectors.html
% [4] "Siemens Digital Industries Software Community." [Online]. 
% Available:
% https://community.sw.siemens.com/s/article/Overlap-What-
% Why-and-How-to-use-it
% [5] Md. R. Hasan, Md. M. Hasan, and M. Z. Hossain, 
% "How many Mel‐frequency cepstral coefficients to be utilized in speech 
% recognition? A study with the Bengali language," The Journal of 
% Engineering, vol. 2021, no. 12, pp. 817–827, Dec. 2021, 
% doi: 10.1049/tje2.12082.
% [6] "fullfile." [Online].
% Available:
% https://www.mathworks.com/help/matlab/ref/fullfile.html
% [7] "dir." [Online].
% Available:
% https://www.mathworks.com/help/matlab/ref/dir.html
% [8] "fileparts." [Online].
% Available:
% https://www.mathworks.com/help/matlab/ref/fileparts.html
% [9] "upper." [Online].
% Available:
% https://www.mathworks.com/help/matlab/ref/string.upper.html
% [10] "Speech Commands Dataset v0.02." Accessed: Sep. 28, 2026. [Online].
% Available:https://www.kaggle.com/datasets/yashdogra/speech-commands/data
% [11] "gscatter." [Online].
% Available:
% https://www.mathworks.com/help/stats/gscatter.html
% [12] "normalize." [Online].
% Available:
% https://www.mathworks.com/help/matlab/ref/double.normalize.html
% [13] "zscore." [Online].
% Available:
% https://www.mathworks.com/help/stats/zscore.html
% [14] "loss." [Online].
% Available:
% https://www.mathworks.com/help/stats/regressionneuralnetwork.loss.html
% [15] "fitcknn." [Online].
% Available:
% https://www.mathworks.com/help/stats/fitcknn.html
% [16] "predict." [Online].
% Available:
% https://www.mathworks.com/help/stats/linearmodel.predict.html
% [17] "confusionchart." [Online].
% Available:
% https://www.mathworks.com/help/stats/confusionchart.html

% ------- End of Script -------