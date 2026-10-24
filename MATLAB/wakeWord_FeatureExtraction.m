function [datasetIdx, coeffPath] = wakeWord_FeatureExtraction(numTrainingPositiveCases,numTrainingNegativeCases, numValidationPositiveCases, numValidationNegativeCases, numTestPositiveCases, numTestNegativeCases)
%WAKEWORD_FEATUREEXTRACTION Summary of this function goes here
%   Detailed explanation goes here

%Flags
isTesting = 0; %hardcode stuff so I don't have to select path every time
isMac = 1;
isWindows = ~isMac;
dontSaveForTesting = 0;

%Initialize datasetIdx
datasetIdx = [];
datasetIdx.relativePath = [];
datasetIdx.word = [];
datasetIdx.isWakeWord = [];
%datasetIdx.isNegativeCase = [];
datasetIdx.isTraining = [];
datasetIdx.isValidation = [];
datasetIdx.isTest = [];
datasetIdx.coeffFname = [];
datasetIdx.audioFullPath = [];
wakeWordWithinWordIdx = [];

currCaseNumber = 1;

if isTesting
    if isMac
        datasetPath = '/Volumes/Extreme_Pro/JHU School/Classes/Machine Learning for Signal Processing (EN.525.670.81.FA26)/Group Project/Dataset of Spoken Words';
    elseif isWindows
        datasetPath ='G:\Other computers\MacMini\JHU School\Classes\Machine Learning for Signal Processing (EN.525.670.81.FA26)\Group Project\Dataset of Spoken Words';
    end
else
    datasetPath = uigetdir(pwd, "Select Dataset Folder");
end

scriptPath = mfilename("fullpath"); [scriptPath, ~, ~] = fileparts(scriptPath);
datasetDir = dir(datasetPath);
featurePath = fullfile(scriptPath, "wakeWordFeatures");
coeffPath = fullfile(featurePath, "MFCCcoeff");


if ~exist(featurePath, "dir")
    mkdir(featurePath);
end


% Choose Wake Word (Only make selection if features haven't been extracted)
%if ~exist(fullfile(currFeaturePath, 'datasetIndex.csv'), "file") || forceFeatureExtraction
isWakeWord = [datasetDir.isdir];
potWakeWords = {datasetDir.name}; potWakeWords = potWakeWords(isWakeWord); %exclude non folders
potWakeWords = setdiff(potWakeWords, {'.', '..', '_background_noise_'}, 'stable'); %exclude nonwords
numWakeWords = length(potWakeWords);
%select Wake Word
wakeWordIdx = listdlg('PromptString', "Choose Wake Word for Feature Extraction", 'ListString', potWakeWords);
currWakeWord = potWakeWords{wakeWordIdx};
%end

%Check to see if more selected than available cases
numWakeWordCases = fullfile(datasetPath, currWakeWord);
chosenWordDir = dir(numWakeWordCases);
numChosenWord = length(chosenWordDir) - 2; %exclude directory pointers
if (numTrainingPositiveCases + numValidationPositiveCases + numTestPositiveCases) > numChosenWord
    fprintf('Not enough cases for the chosen wake word, reduce the number of positive selected (numAvailable = %d)\n', numChosenWord);
    return;
end


if ~exist(coeffPath, "dir")
    mkdir(coeffPath);
end

%% Select random permutation of selected word for POSITIVE TRAINING
selectedTrainingPositive = randperm(numChosenWord, numTrainingPositiveCases) + 2; %add 2 to exclude directory pointers
currDir = chosenWordDir;
for i_mi = 1:numTrainingPositiveCases
    % Select randperm values
    currCaseName = strcat("case", string(currCaseNumber), '.mat');
    currCaseIdx = selectedTrainingPositive(i_mi);
    currAudioPath = fullfile(currDir(currCaseIdx).folder, currDir(currCaseIdx).name);

    % Get relative path
    currRelativePath = fullfile(currWakeWord, currDir(currCaseIdx).name);

    %Save to datasetIndex.csv
    datasetIdx.relativePath = [datasetIdx.relativePath; {currRelativePath}];
    datasetIdx.word = [datasetIdx.word; {currWakeWord}];
    datasetIdx.isWakeWord = [datasetIdx.isWakeWord; 1];
    % datasetIdx.isNegativeCase = [datasetIdx.isNegativeCase; 0];
    datasetIdx.isTraining = [datasetIdx.isTraining; 1];
    datasetIdx.isValidation = [datasetIdx.isValidation; 0];
    datasetIdx.isTest = [datasetIdx.isTest; 0];
    datasetIdx.coeffFname = [datasetIdx.coeffFname; {currCaseName}];
    datasetIdx.audioFullPath = [datasetIdx.audioFullPath; {currAudioPath}];
    wakeWordWithinWordIdx = [wakeWordWithinWordIdx; currCaseIdx];

    %Read audioFile
    [currAudio, fs] = audioread(currAudioPath);
    %Extract MFCC Coeff
    currCoeff = mfcc(currAudio, fs);

    currCoeffFullName = fullfile(coeffPath, currCaseName);
    if ~dontSaveForTesting
        save(currCoeffFullName, "currCoeff");
    end



    %Increment CaseNumber
    currCaseNumber = currCaseNumber + 1;
end

%% Select random permutation of non-wake word for NEGATIVE TRAINING
availableNumbers = setdiff(1:numWakeWords, wakeWordIdx);

%Select random words that are not the wakeword for the negative training
selectedTrainingNegativeWord = availableNumbers(randi(length(availableNumbers), 1, numTrainingNegativeCases));
checkReuse = {};

for i_mi = 1:numTrainingNegativeCases
    %Select randperm of the currWord
    currWord = potWakeWords{selectedTrainingNegativeWord(i_mi)};
    currNegativeWordPath = fullfile(datasetPath, currWord);
    currDir = dir(currNegativeWordPath);
    numNegativeWord = length(currDir) - 2; %exclude directory pointers



    % currTrainingNegativeIdx = randperm(numNegativeWord, 1) + 2;
    % % Get relative path
    % currRelativePath = fullfile(currWord, currDir(currTrainingNegativeIdx).name);
    %
    % % Check if previously used
    % while ismember({currRelativePath}, checkReuse)
    %     currTrainingNegativeIdx = randperm(numNegativeWord, 1) + 2;
    %     currRelativePath = fullfile(currWord, currDir(currTrainingNegativeIdx).name);
    % end
    % checkReuse = [checkReuse; {currRelativePath}];

    % Get all file indices, excluding directory pointers
    allFileIdx = 3:length(currDir);

    % Identify unused files
    allRelativePaths = fullfile(currWord, {currDir(allFileIdx).name});
    availableFileIdx = allFileIdx(~ismember(allRelativePaths, checkReuse));

    % Check that unused files are available
    if isempty(availableFileIdx)
        error('No unused files remaining for word: %s', currWord);
    end

    % Randomly select an unused file
    currTrainingNegativeIdx = availableFileIdx(randi(length(availableFileIdx)));

    % Get relative path
    currRelativePath = fullfile(currWord, currDir(currTrainingNegativeIdx).name);

    % Record selected file
    checkReuse = [checkReuse; {currRelativePath}];




    % Select randperm values
    currCaseName = strcat("case", string(currCaseNumber), '.mat');
    currCaseIdx = currTrainingNegativeIdx;
    currAudioPath = fullfile(currDir(currCaseIdx).folder, currDir(currCaseIdx).name);


    %Save to datasetIndex.csv
    datasetIdx.relativePath = [datasetIdx.relativePath; {currRelativePath}];
    datasetIdx.word = [datasetIdx.word; {currWord}];
    datasetIdx.isWakeWord = [datasetIdx.isWakeWord; 0];
    %datasetIdx.isNegativeCase = [datasetIdx.isNegativeCase; 1];
    datasetIdx.isTraining = [datasetIdx.isTraining; 1];
    datasetIdx.isValidation = [datasetIdx.isValidation; 0];
    datasetIdx.isTest = [datasetIdx.isTest; 0];
    datasetIdx.coeffFname = [datasetIdx.coeffFname; {currCaseName}];
    datasetIdx.audioFullPath = [datasetIdx.audioFullPath; {currAudioPath}];

    %Read audioFile
    [currAudio, fs] = audioread(currAudioPath);
    %Extract MFCC Coeff
    currCoeff = mfcc(currAudio, fs);

    currCoeffFullName = fullfile(coeffPath, currCaseName);
    if ~dontSaveForTesting
        save(currCoeffFullName, "currCoeff");
    end


    %Increment CaseNumber
    currCaseNumber = currCaseNumber + 1;
end



%% Select random permutation of selected word for POSITIVE VALIDATION
availableNumbers = setdiff(3:numChosenWord+2, wakeWordWithinWordIdx);

selectedValidationPositive = availableNumbers(randperm(length(availableNumbers), numValidationPositiveCases));

currDir = chosenWordDir;

for i_mi = 1:numValidationPositiveCases
    % Select randperm values
    currCaseName = strcat("case", string(currCaseNumber), '.mat');
    currCaseIdx = selectedValidationPositive(i_mi);
    currAudioPath = fullfile(currDir(currCaseIdx).folder, currDir(currCaseIdx).name);

    % Get relative path
    currRelativePath = fullfile(currWakeWord, currDir(currCaseIdx).name);

    %Save to datasetIndex.csv
    datasetIdx.relativePath = [datasetIdx.relativePath; {currRelativePath}];
    datasetIdx.word = [datasetIdx.word; {currWakeWord}];
    datasetIdx.isWakeWord = [datasetIdx.isWakeWord; 1];
    %datasetIdx.isNegativeCase = [datasetIdx.isNegativeCase; 0];
    datasetIdx.isTraining = [datasetIdx.isTraining; 0];
    datasetIdx.isValidation = [datasetIdx.isValidation; 1];
    datasetIdx.isTest = [datasetIdx.isTest; 0];
    datasetIdx.coeffFname = [datasetIdx.coeffFname; {currCaseName}];
    datasetIdx.audioFullPath = [datasetIdx.audioFullPath; {currAudioPath}];
    wakeWordWithinWordIdx = [wakeWordWithinWordIdx; currCaseIdx];

    %Read audioFile
    [currAudio, fs] = audioread(currAudioPath);
    %Extract MFCC Coeff
    currCoeff = mfcc(currAudio, fs);

    currCoeffFullName = fullfile(coeffPath, currCaseName);
    if ~dontSaveForTesting
        save(currCoeffFullName, "currCoeff");
    end


    %Increment CaseNumber
    currCaseNumber = currCaseNumber + 1;
end


%% Select random permutation of non-wake word for NEGATIVE VALIDATION
availableNumbers = setdiff(1:numWakeWords, wakeWordIdx);

%Select random words that are not the wakeword for the negative training
selectedTrainingNegativeWord = availableNumbers(randi(length(availableNumbers), 1, numValidationNegativeCases));

for i_mi = 1:numValidationNegativeCases
    %Select randperm of the currWord
    currWord = potWakeWords{selectedTrainingNegativeWord(i_mi)};
    currNegativeWordPath = fullfile(datasetPath, currWord);
    currDir = dir(currNegativeWordPath);
    numNegativeWord = length(currDir) - 2; %exclude directory pointers
    % currTrainingNegativeIdx = randperm(numNegativeWord, 1) + 2;
    %
    %
    % % Get relative path
    % currRelativePath = fullfile(currWord, currDir(currTrainingNegativeIdx).name);
    %
    % % Check if previously used
    % while ismember({currRelativePath}, checkReuse)
    %     currTrainingNegativeIdx = randperm(numNegativeWord, 1) + 2;
    %     currRelativePath = fullfile(currWord, currDir(currTrainingNegativeIdx).name);
    % end
    % checkReuse = [checkReuse; {currRelativePath}];
    % Get all file indices, excluding directory pointers
    allFileIdx = 3:length(currDir);

    % Identify unused files
    allRelativePaths = fullfile(currWord, {currDir(allFileIdx).name});
    availableFileIdx = allFileIdx(~ismember(allRelativePaths, checkReuse));

    % Check that unused files are available
    if isempty(availableFileIdx)
        error('No unused files remaining for word: %s', currWord);
    end

    % Randomly select an unused file
    currTrainingNegativeIdx = availableFileIdx(randi(length(availableFileIdx)));

    % Get relative path
    currRelativePath = fullfile(currWord, currDir(currTrainingNegativeIdx).name);

    % Record selected file
    checkReuse = [checkReuse; {currRelativePath}];





    % Select randperm values
    currCaseName = strcat("case", string(currCaseNumber), '.mat');
    currCaseIdx = currTrainingNegativeIdx;
    currAudioPath = fullfile(currDir(currCaseIdx).folder, currDir(currCaseIdx).name);


    %Save to datasetIndex.csv
    datasetIdx.relativePath = [datasetIdx.relativePath; {currRelativePath}];
    datasetIdx.word = [datasetIdx.word; {currWord}];
    datasetIdx.isWakeWord = [datasetIdx.isWakeWord; 0];
    %datasetIdx.isNegativeCase = [datasetIdx.isNegativeCase; 1];
    datasetIdx.isTraining = [datasetIdx.isTraining; 0];
    datasetIdx.isValidation = [datasetIdx.isValidation; 1];
    datasetIdx.isTest = [datasetIdx.isTest; 0];
    datasetIdx.coeffFname = [datasetIdx.coeffFname; {currCaseName}];
    datasetIdx.audioFullPath = [datasetIdx.audioFullPath; {currAudioPath}];

    %Read audioFile
    [currAudio, fs] = audioread(currAudioPath);
    %Extract MFCC Coeff
    currCoeff = mfcc(currAudio, fs);

    currCoeffFullName = fullfile(coeffPath, currCaseName);
    if ~dontSaveForTesting
        save(currCoeffFullName, "currCoeff");
    end


    %Increment CaseNumber
    currCaseNumber = currCaseNumber + 1;
end


%% Select random permutation of selected word for POSITIVE TESTING
availableNumbers = setdiff(3:numChosenWord+2, wakeWordWithinWordIdx);

selectedTestPositive = availableNumbers(randperm(length(availableNumbers), numTestPositiveCases));

currDir = chosenWordDir;

for i_mi = 1:numTestPositiveCases
    % Select randperm values
    currCaseName = strcat("case", string(currCaseNumber), '.mat');
    currCaseIdx = selectedTestPositive(i_mi);
    currAudioPath = fullfile(currDir(currCaseIdx).folder, currDir(currCaseIdx).name);

    % Get relative path
    currRelativePath = fullfile(currWakeWord, currDir(currCaseIdx).name);

    %Save to datasetIndex.csv
    datasetIdx.relativePath = [datasetIdx.relativePath; {currRelativePath}];
    datasetIdx.word = [datasetIdx.word; {currWakeWord}];
    datasetIdx.isWakeWord = [datasetIdx.isWakeWord; 1];
    %datasetIdx.isNegativeCase = [datasetIdx.isNegativeCase; 0];
    datasetIdx.isTraining = [datasetIdx.isTraining; 0];
    datasetIdx.isValidation = [datasetIdx.isValidation; 0];
    datasetIdx.isTest = [datasetIdx.isTest; 1];
    datasetIdx.coeffFname = [datasetIdx.coeffFname; {currCaseName}];
    datasetIdx.audioFullPath = [datasetIdx.audioFullPath; {currAudioPath}];
    wakeWordWithinWordIdx = [wakeWordWithinWordIdx; currCaseIdx];

    %Read audioFile
    [currAudio, fs] = audioread(currAudioPath);
    %Extract MFCC Coeff
    currCoeff = mfcc(currAudio, fs);

    currCoeffFullName = fullfile(coeffPath, currCaseName);
    if ~dontSaveForTesting
        save(currCoeffFullName, "currCoeff");
    end


    %Increment CaseNumber
    currCaseNumber = currCaseNumber + 1;
end


%% Select random permutation of non-wake word for NEGATIVE TESTING
availableNumbers = setdiff(1:numWakeWords, wakeWordIdx);

%Select random words that are not the wakeword for the negative training
selectedTrainingNegativeWord = availableNumbers(randi(length(availableNumbers), 1, numTestNegativeCases));

for i_mi = 1:numTestNegativeCases
    %Select randperm of the currWord
    currWord = potWakeWords{selectedTrainingNegativeWord(i_mi)};
    currNegativeWordPath = fullfile(datasetPath, currWord);
    currDir = dir(currNegativeWordPath);
    numNegativeWord = length(currDir) - 2; %exclude directory pointers
    % currTrainingNegativeIdx = randperm(numNegativeWord, 1) + 2;
    %
    %
    % % Get relative path
    % currRelativePath = fullfile(currWord, currDir(currTrainingNegativeIdx).name);
    %
    % % Check if previously used
    % while ismember({currRelativePath}, checkReuse)
    %     currTrainingNegativeIdx = randperm(numNegativeWord, 1) + 2;
    %     currRelativePath = fullfile(currWord, currDir(currTrainingNegativeIdx).name);
    % end
    % checkReuse = [checkReuse; {currRelativePath}];
    % Get all file indices, excluding directory pointers
    allFileIdx = 3:length(currDir);

    % Identify unused files
    allRelativePaths = fullfile(currWord, {currDir(allFileIdx).name});
    availableFileIdx = allFileIdx(~ismember(allRelativePaths, checkReuse));

    % Check that unused files are available
    if isempty(availableFileIdx)
        error('No unused files remaining for word: %s', currWord);
    end

    % Randomly select an unused file
    currTrainingNegativeIdx = availableFileIdx(randi(length(availableFileIdx)));

    % Get relative path
    currRelativePath = fullfile(currWord, currDir(currTrainingNegativeIdx).name);

    % Record selected file
    checkReuse = [checkReuse; {currRelativePath}];


    % Select randperm values
    currCaseName = strcat("case", string(currCaseNumber), '.mat');
    currCaseIdx = currTrainingNegativeIdx;
    currAudioPath = fullfile(currDir(currCaseIdx).folder, currDir(currCaseIdx).name);


    %Save to datasetIndex.csv
    datasetIdx.relativePath = [datasetIdx.relativePath; {currRelativePath}];
    datasetIdx.word = [datasetIdx.word; {currWord}];
    datasetIdx.isWakeWord = [datasetIdx.isWakeWord; 0];
    %datasetIdx.isNegativeCase = [datasetIdx.isNegativeCase; 1];
    datasetIdx.isTraining = [datasetIdx.isTraining; 0];
    datasetIdx.isValidation = [datasetIdx.isValidation; 0];
    datasetIdx.isTest = [datasetIdx.isTest; 1];
    datasetIdx.coeffFname = [datasetIdx.coeffFname; {currCaseName}];
    datasetIdx.audioFullPath = [datasetIdx.audioFullPath; {currAudioPath}];

    %Read audioFile
    [currAudio, fs] = audioread(currAudioPath);
    %Extract MFCC Coeff
    currCoeff = mfcc(currAudio, fs);

    currCoeffFullName = fullfile(coeffPath, currCaseName);
    if ~dontSaveForTesting
        save(currCoeffFullName, "currCoeff");
    end


    %Increment CaseNumber
    currCaseNumber = currCaseNumber + 1;
end


% Verify that I didn't accidentally reuse any of the cases
allPaths = string(datasetIdx.relativePath);

numDuplicates = numel(allPaths) - numel(unique(allPaths));

fprintf('Total cases: %d\n', numel(allPaths));
fprintf('Duplicate cases: %d\n', numDuplicates);

assert(numDuplicates == 0, 'Duplicate audio files detected!');


% Save datasetIndex.csv
featureIdxTable = struct2table(datasetIdx);
writetable(featureIdxTable, fullfile(featurePath, 'datasetIndex.csv'));
end